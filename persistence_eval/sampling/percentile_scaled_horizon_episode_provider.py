"""Percentile-derived horizons + scaled-horizon causal masking.

Composes the horizon-derivation logic from `PercentileHorizonEpisodeProvider`
with the per-query observation-cutoff logic from
`ScaledHorizonEpisodeProvider`, so a single self-contained eval_scenario
config can replace the previous "compute horizons, then re-eval via the
override script" workflow (see scripts/run_observation_window_sweep.py).

For each track, the same per-track persistence-time distribution drives the
query-horizon grid as in PercentileHorizonEpisodeProvider. Each query at
`dt_q` then sees only observations strictly earlier than
`t_q − min(recency_cap, dt_q/2)`, matching ScaledHorizonEpisodeProvider's
causal cutoff. Reporting bins by per-query effective horizon.

Requires `t_min > 0` (ScaledHorizonEpisodeProvider rejects dt_q = 0).
"""

from __future__ import annotations

from typing import Literal

from persistence_eval.core.protocols import TrackDataset
from persistence_eval.sampling.percentile_horizon_episode_provider import (
    PercentileHorizonEpisodeProvider,
)
from persistence_eval.sampling.scaled_horizon_episode_provider import (
    ScaledHorizonEpisodeProvider,
)

Spacing = Literal["linear", "log", "power", "huber", "quantile"]


class PercentileScaledHorizonEpisodeProvider(ScaledHorizonEpisodeProvider):
    """Percentile-derived horizons fed into the scaled-horizon causal-masking provider."""

    def __init__(
        self,
        track_dataset: TrackDataset,
        *,
        percentile: float,
        recency_cap: float,
        num_query_times: int = 20,
        spacing: Spacing = "power",
        t_min: float = 1.0,
        max_query_inflation: float = 1.0,
        power_exponent: float = 2.0,
        huber_transition_frac: float = 0.2,
        huber_u_delta: float = 0.5,
        require_disappearance_observed: bool = False,
        name: str | None = None,
    ) -> None:
        if t_min <= 0.0:
            raise ValueError(
                "PercentileScaledHorizonEpisodeProvider requires t_min > 0 "
                f"(ScaledHorizonEpisodeProvider rejects dt_q = 0); got t_min={t_min}"
            )
        if not (0.0 < percentile <= 100.0):
            raise ValueError(f"percentile must be in (0, 100], got {percentile}")
        if num_query_times < 1:
            raise ValueError(f"num_query_times must be >= 1, got {num_query_times}")
        if max_query_inflation <= 0.0:
            raise ValueError(
                f"max_query_inflation must be > 0, got {max_query_inflation}"
            )

        import numpy as np

        persistence_times = [
            bounds.last_observed_time - bounds.first_observed_time
            for _, _, bounds in track_dataset.iter_tracks()
        ]
        if not persistence_times:
            raise ValueError(
                "Cannot compute percentile horizons: track_dataset yielded no tracks."
            )

        percentile_value = float(np.percentile(persistence_times, percentile))
        cap = max_query_inflation * percentile_value
        if cap <= t_min:
            raise ValueError(
                f"Computed cap ({cap:.6g}) is <= t_min ({t_min:.6g}); "
                "the percentile of persistence times is too small for the requested t_min."
            )

        horizons = PercentileHorizonEpisodeProvider._build_horizons(
            spacing=spacing,
            num_query_times=num_query_times,
            t_min=t_min,
            cap=cap,
            persistence_times=persistence_times,
            percentile=percentile,
            max_query_inflation=max_query_inflation,
            power_exponent=power_exponent,
            huber_transition_frac=huber_transition_frac,
            huber_u_delta=huber_u_delta,
        )

        self._percentile = percentile
        self._percentile_value = percentile_value
        self._cap = cap
        self._spacing = spacing

        super().__init__(
            track_dataset=track_dataset,
            query_horizons=[float(h) for h in horizons],
            recency_cap=recency_cap,
            require_disappearance_observed=require_disappearance_observed,
            name=name,
        )
