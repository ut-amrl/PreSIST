from __future__ import annotations

from typing import Literal

import numpy as np

from persistence_eval.core.protocols import TrackDataset
from persistence_eval.sampling.episode_provider import TrackBasedEpisodeProvider

Spacing = Literal["linear", "log", "power", "huber", "quantile"]


class PercentileHorizonEpisodeProvider(TrackBasedEpisodeProvider):
    """
    Like TrackBasedEpisodeProvider, but derives the (single, dataset-wide) query
    horizons list from the distribution of per-track persistence times.

    Persistence time per track is defined as
        bounds.last_observed_time - bounds.first_observed_time
    so every track contributes regardless of whether disappearance was observed.

    The horizon range is [t_min, max_query_inflation * percentile_value], sampled
    at num_query_times points using the chosen spacing. Inherits per-anchor
    EvaluationContext shape from the parent.
    """

    def __init__(
        self,
        track_dataset: TrackDataset,
        *,
        percentile: float,
        num_query_times: int = 20,
        spacing: Spacing = "power",
        t_min: float = 0.1,
        max_query_inflation: float = 1.0,
        power_exponent: float = 2.0,
        huber_transition_frac: float = 0.2,
        huber_u_delta: float = 0.5,
        require_disappearance_observed: bool = False,
        name: str | None = None,
    ) -> None:
        if not (0.0 < percentile <= 100.0):
            raise ValueError(f"percentile must be in (0, 100], got {percentile}")
        if num_query_times < 1:
            raise ValueError(f"num_query_times must be >= 1, got {num_query_times}")
        if max_query_inflation <= 0.0:
            raise ValueError(f"max_query_inflation must be > 0, got {max_query_inflation}")
        if spacing == "log" and t_min <= 0.0:
            raise ValueError(f"spacing='log' requires t_min > 0, got {t_min}")
        if spacing == "huber" and not (0.0 < huber_u_delta < 1.0):
            raise ValueError(f"huber_u_delta must be in (0, 1), got {huber_u_delta}")

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

        horizons = self._build_horizons(
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
            query_horizons=list(horizons),
            require_disappearance_observed=require_disappearance_observed,
        )

    @staticmethod
    def _build_horizons(
        *,
        spacing: Spacing,
        num_query_times: int,
        t_min: float,
        cap: float,
        persistence_times: list[float],
        percentile: float,
        max_query_inflation: float,
        power_exponent: float,
        huber_transition_frac: float,
        huber_u_delta: float,
    ) -> np.ndarray:
        if num_query_times == 1:
            return np.array([cap], dtype=float)

        if spacing == "linear":
            return np.linspace(t_min, cap, num_query_times)

        if spacing == "log":
            return np.logspace(np.log10(t_min), np.log10(cap), num_query_times)

        if spacing == "power":
            u = np.linspace(0.0, 1.0, num_query_times)
            return t_min + (cap - t_min) * u ** power_exponent

        if spacing == "huber":
            delta = huber_transition_frac * cap
            u = np.linspace(0.0, 1.0, num_query_times)
            quad_mask = u <= huber_u_delta
            horizons = np.empty_like(u)
            horizons[quad_mask] = (
                t_min + (delta - t_min) * (u[quad_mask] / huber_u_delta) ** 2
            )
            horizons[~quad_mask] = (
                delta + (cap - delta) * (u[~quad_mask] - huber_u_delta) / (1.0 - huber_u_delta)
            )
            return horizons

        if spacing == "quantile":
            qs = np.linspace(0.0, percentile, num_query_times)
            horizons = np.percentile(persistence_times, qs) * max_query_inflation
            return np.clip(horizons, t_min, None)

        raise ValueError(f"Unknown spacing: {spacing!r}")
