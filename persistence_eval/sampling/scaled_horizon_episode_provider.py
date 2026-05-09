"""Scaled-horizon alternate eval — observation-conditioned with per-query causal masking.

For each query at dt_q from anchor, the latest observation visible to that query
is bounded to `t_q - min(recency_cap, dt_q/2)`. This makes the *effective*
prediction horizon `h_eff = min(recency_cap, dt_q/2)` for `dt_q ≤ 2·recency_cap`,
and `dt_q − recency_cap` for longer queries (see plan).

Causal masking is the provider's job — different cutoffs imply different
observation sets, so we group queries that share the same observation prefix
(of the sorted post-anchor obs list within `[anchor, anchor + recency_cap]`)
into one EvaluationContext per group. Each context becomes one filter call.

Observations are bounded to `[anchor_time, anchor_time + recency_cap]` because
the rule's cutoff for any query is at most `anchor + recency_cap`, so obs past
that window are never visible under this rule.
"""

from __future__ import annotations

from bisect import bisect_left
from itertools import groupby
from typing import Iterable

from persistence_eval.core.domain import (
    BinaryObservationEvent,
    EvaluationContext,
    TrackTemporalBounds,
)
from persistence_eval.core.protocols import TrackDataset


class ScaledHorizonEpisodeProvider:
    """Emits one EvaluationContext per (track, anchor_time, observation-prefix) group."""

    def __init__(
        self,
        track_dataset: TrackDataset,
        query_horizons: list[float],
        *,
        recency_cap: float,
        require_disappearance_observed: bool = False,
        name: str | None = None,
    ) -> None:
        if recency_cap <= 0.0:
            raise ValueError(f"recency_cap must be > 0, got {recency_cap}")
        if any(q <= 0.0 for q in query_horizons):
            raise ValueError(
                "ScaledHorizonEpisodeProvider requires strictly positive query horizons "
                "(dt_q = 0 has h_eff = 0 by construction)."
            )
        self._query_horizons = list(query_horizons)
        self._recency_cap = float(recency_cap)
        self.name = name

        self._lookup: dict[str, tuple[list[BinaryObservationEvent], TrackTemporalBounds]] = {}
        for track_id, observations, bounds in track_dataset.iter_tracks():
            if require_disappearance_observed and bounds.first_missing_time is None:
                continue
            self._lookup[track_id] = (observations, bounds)

    def get_contexts(self, track_id: str, anchor_time: float) -> Iterable[EvaluationContext]:
        entry = self._lookup.get(track_id)
        if entry is None:
            return

        observations, bounds = entry
        X = self._recency_cap

        # Bound post-anchor obs to [anchor_time, anchor_time + X], sorted by time.
        post_obs = sorted(
            (
                o for o in observations
                if anchor_time <= o.timestamp <= anchor_time + X
            ),
            key=lambda o: o.timestamp,
        )
        obs_times = [o.timestamp for o in post_obs]

        # Build per-query (prefix_length, dt_q, h_eff) records.
        # h_eff is the *actual* gap from the latest visible obs (or from the
        # anchor when no obs is visible), not the rule's lower bound. This is
        # the value the reporting layer bins on.
        records = []
        for dt_q in self._query_horizons:
            t_q = anchor_time + dt_q
            gap_floor = min(X, dt_q / 2.0)
            cutoff = t_q - gap_floor
            k = bisect_left(obs_times, cutoff)  # strict <
            if k == 0:
                h_eff = dt_q  # no obs visible — h_eff is gap from anchor
            else:
                h_eff = t_q - obs_times[k - 1]
            records.append((k, dt_q, h_eff))

        # Group queries by prefix length k. Each group becomes one EvaluationContext.
        records.sort(key=lambda r: r[0])
        for k, group in groupby(records, key=lambda r: r[0]):
            group_records = list(group)
            qs = [r[1] for r in group_records]
            hs = [r[2] for r in group_records]
            yield EvaluationContext(
                query_times=qs,
                bounds=bounds,
                observations=post_obs[:k],
                effective_horizons=hs,
            )
