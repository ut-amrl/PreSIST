"""Scaled-horizon variant — anchor-relative forward observation window.

For each query at `dt_q` from anchor, observations are visible iff their
timestamp lies in `[anchor, anchor + min(recency_cap, dt_q/2)]`. The window
grows with `dt_q` (so longer horizons see more obs) but is capped by
`recency_cap`. Unlike `ScaledHorizonEpisodeProvider`, the window is anchored
at the filter's start rather than walked back from `t_q`, so for large
`dt_q` the visible set saturates at `[anchor, anchor + recency_cap]`
instead of sliding forward toward `t_q`.

Compared with the sibling providers (same horizon shape, different obs rule):

| Provider                              | Visible obs at query dt_q |
|---------------------------------------|---------------------------|
| ScaledHorizonEpisodeProvider          | strictly before `t_q − min(recency_cap, dt_q/2)` |
| WindowedHorizonEpisodeProvider        | timestamps ≤ `t_q`, capped at `anchor + recency_cap` |
| **this one (anchor-window)**          | `[anchor, anchor + min(recency_cap, dt_q/2)]` |

The visible obs set is monotone non-decreasing in `dt_q`, so queries
sharing a prefix length collapse into one `EvaluationContext` (same
grouping pattern as the scaled / windowed providers).
"""

from __future__ import annotations

from bisect import bisect_right
from itertools import groupby
from typing import Iterable

from persistence_eval.core.domain import (
    BinaryObservationEvent,
    EvaluationContext,
    TrackTemporalBounds,
)
from persistence_eval.core.protocols import TrackDataset


class AnchorWindowScaledHorizonEpisodeProvider:
    """Emits one EvaluationContext per (track, anchor_time, observation-prefix) group.

    Observation rule: an obs at time `t_o` is visible to a query with horizon
    `dt_q` iff `anchor ≤ t_o ≤ anchor + min(recency_cap, dt_q / 2)`.
    """

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
                "AnchorWindowScaledHorizonEpisodeProvider requires strictly "
                "positive query horizons (dt_q = 0 yields an empty window)."
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

        post_obs = sorted(
            (
                o for o in observations
                if anchor_time <= o.timestamp <= anchor_time + X
            ),
            key=lambda o: o.timestamp,
        )
        obs_times = [o.timestamp for o in post_obs]

        # For each query, count obs with timestamp ≤ anchor + min(X, dt_q/2).
        records = []
        for dt_q in self._query_horizons:
            window = min(X, dt_q / 2.0)
            upper = anchor_time + window
            k = bisect_right(obs_times, upper)
            records.append((k, dt_q))

        records.sort(key=lambda r: r[0])
        for k, group in groupby(records, key=lambda r: r[0]):
            qs = [r[1] for r in group]
            yield EvaluationContext(
                query_times=qs,
                bounds=bounds,
                observations=post_obs[:k],
                effective_horizons=qs,
            )
