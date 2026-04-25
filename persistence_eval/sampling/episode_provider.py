from __future__ import annotations

from typing import Iterable

from persistence_eval.core.domain import BinaryObservationEvent, EvaluationContext, TrackTemporalBounds
from persistence_eval.core.protocols import TrackDataset


class TrackBasedEpisodeProvider:
    """
    Builds EvaluationContexts for inference artifacts by looking up track data.
    Eagerly indexes all tracks at construction so get_contexts() is an O(1) dict lookup.

    For each (track_id, anchor_time) pair the provider yields one EvaluationContext:
      - prior_observations: all track observations with timestamp < anchor_time,
        optionally capped to max_prior_observations most-recent entries.
      - query_times: the fixed list of dt offsets supplied at construction.
      - bounds: the TrackTemporalBounds for the track.

    Artifacts whose track_id is not present in the dataset are silently skipped
    (get_contexts yields nothing).
    """

    def __init__(
        self,
        track_dataset: TrackDataset,
        query_horizons: list[float],
        *,
        require_disappearance_observed: bool = False,
        include_prior_history: bool = True,
        max_prior_observations: int | None = None,
    ) -> None:
        self._query_horizons = list(query_horizons)
        self._include_prior = include_prior_history
        self._max_prior = max_prior_observations

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

        if self._include_prior:
            prior = [o for o in observations if o.timestamp < anchor_time]
            if self._max_prior is not None:
                prior = prior[-self._max_prior:]
        else:
            prior = []

        yield EvaluationContext(
            prior_observations=prior,
            query_times=self._query_horizons,
            bounds=bounds,
        )
