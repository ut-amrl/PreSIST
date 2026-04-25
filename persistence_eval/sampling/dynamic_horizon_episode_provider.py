from __future__ import annotations

from typing import Iterable

from persistence_eval.core.domain import BinaryObservationEvent, EvaluationContext, TrackTemporalBounds
from persistence_eval.core.protocols import TrackDataset


class DynamicHorizonEpisodeProvider:
    """
    Builds EvaluationContexts with per-anchor dynamic query horizons derived from track bounds.

    For each (track_id, anchor_time) pair, yields up to two EvaluationContexts:
      1. dt1 = (last_observed - anchor_time) / 2  (halfway between anchor and last visible)
      2. dt2 = (first_missing - anchor_time) + dt1  (same gap past first absent observation),
         yielded only if first_missing_time is not None.

    Yields nothing if anchor_time >= last_observed_time (no gap to halve).

    prior_observations is always empty (anchor-only evaluation).
    """

    def __init__(
        self,
        track_dataset: TrackDataset,
        *,
        name: str | None = None,
        require_disappearance_observed: bool = False,
    ) -> None:
        self._lookup: dict[str, tuple[list[BinaryObservationEvent], TrackTemporalBounds]] = {}
        for track_id, observations, bounds in track_dataset.iter_tracks():
            if require_disappearance_observed and bounds.first_missing_time is None:
                continue
            self._lookup[track_id] = (observations, bounds)

    def get_contexts(self, track_id: str, anchor_time: float) -> Iterable[EvaluationContext]:
        entry = self._lookup.get(track_id)
        if entry is None:
            return

        _, bounds = entry

        gap = (bounds.last_observed_time - anchor_time) / 2.0
        if gap <= 0.0:
            return

        query_times: list[float] = [gap]

        if bounds.first_missing_time is not None:
            dt2 = (bounds.first_missing_time - anchor_time) + gap
            if dt2 > 0.0:
                query_times.append(dt2)

        yield EvaluationContext(
            prior_observations=[],
            query_times=query_times,
            bounds=bounds,
        )
