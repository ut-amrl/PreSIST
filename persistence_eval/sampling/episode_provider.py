from __future__ import annotations

import logging
from typing import Iterable

from persistence_eval.core.domain import BinaryObservationEvent, EvaluationContext, TrackTemporalBounds
from persistence_eval.core.protocols import TrackDataset

logger = logging.getLogger(__name__)


def _log_query_horizons(provider_cls_name: str, query_horizons_s: list[float]) -> None:
    """Log a fixed query-horizon list in hours so it's visible before eval starts."""
    hrs = ", ".join(f"{q / 3600.0:.4g}" for q in query_horizons_s)
    logger.info(
        "%s: %d fixed query horizons (hours): [%s]",
        provider_cls_name, len(query_horizons_s), hrs,
    )


class TrackBasedEpisodeProvider:
    """
    Builds EvaluationContexts for inference artifacts by looking up track data.
    Eagerly indexes all tracks at construction so get_contexts() is an O(1) dict lookup.

    For each (track_id, anchor_time) pair the provider yields one EvaluationContext
    with the fixed list of dt offsets supplied at construction and the track's
    TrackTemporalBounds. The persistence filter is anchor-only — observations are
    not passed to it; ground truth comes from bounds.

    Artifacts whose track_id is not present in the dataset are silently skipped
    (get_contexts yields nothing).
    """

    def __init__(
        self,
        track_dataset: TrackDataset,
        query_horizons: list[float],
        *,
        require_disappearance_observed: bool = False,
    ) -> None:
        self._query_horizons = list(query_horizons)
        _log_query_horizons(type(self).__name__, self._query_horizons)

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

        yield EvaluationContext(
            query_times=self._query_horizons,
            bounds=bounds,
        )
