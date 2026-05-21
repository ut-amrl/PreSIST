"""Percentile-derived horizons + anchor-window observation rule.

Same horizon-derivation logic as `PercentileScaledHorizonEpisodeProvider`
(percentile of per-track persistence times, sampled across `num_query_times`
points using the chosen spacing). The observation-feeding rule, however, is
the anchor-window rule: each query at `dt_q` sees observations with
timestamps in `[anchor, anchor + min(recency_cap, dt_q / 2)]`.

Comparison with the other percentile-horizon scenarios:

| Provider                                | Visible obs at query dt_q |
|-----------------------------------------|---------------------------|
| PercentileScaledHorizonEpisodeProvider  | strictly before `t_q − min(recency_cap, dt_q/2)` |
| PercentileWindowedHorizonEpisodeProvider| timestamps ≤ `t_q`, capped at `anchor + recency_cap` |
| **this one (anchor-window)**            | `[anchor, anchor + min(recency_cap, dt_q/2)]` |

For the multi-query case the visible obs set is monotone non-decreasing in
`dt_q` (longer horizon = wider window = more visible obs), so queries
sharing a prefix length collapse into one `EvaluationContext`.
"""

from __future__ import annotations

from bisect import bisect_right
from itertools import groupby
from typing import Iterable

from persistence_eval.core.domain import EvaluationContext
from persistence_eval.sampling.percentile_scaled_horizon_episode_provider import (
    PercentileScaledHorizonEpisodeProvider,
)


class PercentileAnchorWindowHorizonEpisodeProvider(PercentileScaledHorizonEpisodeProvider):
    """Same horizon construction as the scaled variant; observation rule is
    `[anchor, anchor + min(recency_cap, dt_q / 2)]` (inclusive) per query.
    """

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
