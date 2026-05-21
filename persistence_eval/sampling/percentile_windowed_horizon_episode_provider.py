"""Percentile-derived horizons + obs-up-to-query-time (no future-obs leakage).

Same horizon-derivation logic as `PercentileScaledHorizonEpisodeProvider`
(percentile of per-track persistence times, sampled across `num_query_times`
points using the chosen spacing), but the observation-feeding rule is
weakened: each query at `dt_q` sees every observation with timestamp in
`[anchor, anchor + min(dt_q, recency_cap)]`. No `dt_q/2` causal buffer like
`ScaledHorizonEpisodeProvider`; the only cutoff is the query time itself
(inclusive) and the global `recency_cap`.

Comparison with the other two scenarios sharing the same percentile-horizon
shape:

| Provider                              | Visible obs at query dt_q |
|---------------------------------------|---------------------------|
| PercentileScaledHorizonEpisodeProvider| strictly before t_q − min(recency_cap, dt_q/2) |
| **this one (windowed)**               | timestamps ≤ t_q, capped at anchor + recency_cap |
| (would-be acausal variant)            | all obs in [anchor, anchor + recency_cap], regardless of dt_q |

Use case: gives mixture filters access to *some* obs at every query (so
argmax-k component selection isn't starving short-horizon queries the way
the strict scaled rule does) without leaking observations from after the
query time. Causal but maximally permissive.

For the multi-query case the visible obs set is monotone non-decreasing in
`dt_q` (longer horizon = more visible obs), so we group queries by their
prefix length and emit one `EvaluationContext` per group (same pattern as
the scaled-horizon provider). For PKLot's 5-min cadence and a 30-min
recency_cap, expect 1–6 distinct prefix-length groups per anchor.

Outputs land under `survival_based__percentile_windowed_horizons/` so they
do not collide with the per-query-masked scenario.
"""

from __future__ import annotations

from bisect import bisect_right
from itertools import groupby
from typing import Iterable

from persistence_eval.core.domain import EvaluationContext
from persistence_eval.sampling.percentile_scaled_horizon_episode_provider import (
    PercentileScaledHorizonEpisodeProvider,
)


class PercentileWindowedHorizonEpisodeProvider(PercentileScaledHorizonEpisodeProvider):
    """Same horizon construction as the scaled variant; observation cutoff is
    `min(t_q, anchor + recency_cap)` per query (inclusive).
    """

    def get_contexts(self, track_id: str, anchor_time: float) -> Iterable[EvaluationContext]:
        entry = self._lookup.get(track_id)
        if entry is None:
            return

        observations, bounds = entry
        X = self._recency_cap

        # Bound observations to [anchor, anchor + recency_cap] up front so the
        # recency_cap remains the global ceiling. Per-query cutoffs are then
        # ≤ t_q within that window.
        post_obs = sorted(
            (
                o for o in observations
                if anchor_time <= o.timestamp <= anchor_time + X
            ),
            key=lambda o: o.timestamp,
        )
        obs_times = [o.timestamp for o in post_obs]

        # For each query, find the number of observations with timestamp ≤ t_q
        # (inclusive). `bisect_right` returns the count of elements ≤ value.
        records = []
        for dt_q in self._query_horizons:
            t_q = anchor_time + dt_q
            k = bisect_right(obs_times, t_q)
            records.append((k, dt_q))

        # Group queries by prefix length k. Queries sharing the same visible
        # observation prefix collapse into one EvaluationContext.
        records.sort(key=lambda r: r[0])
        for k, group in groupby(records, key=lambda r: r[0]):
            qs = [r[1] for r in group]
            yield EvaluationContext(
                query_times=qs,
                bounds=bounds,
                observations=post_obs[:k],
                effective_horizons=qs,
            )
