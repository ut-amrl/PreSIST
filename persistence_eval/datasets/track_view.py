from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from persistence_eval.core.domain import BinaryObservationEvent, TrackTemporalBounds
from persistence_eval.datasets.backing import AbsBackingDataset


class AbsTrackDataset(ABC):
    """
    Provides per-track observation sequences and temporal bounds for episode sampling.
    Implemented either directly or via TrackView wrapping a BackingDataset.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def iter_tracks(
        self,
    ) -> Iterable[tuple[str, list[BinaryObservationEvent], TrackTemporalBounds]]:
        """Yields (track_id, observations, temporal_bounds) for each track."""
        ...


class TrackView(AbsTrackDataset):
    """
    Wraps an AbsBackingDataset to expose the TrackDataset protocol.
    Concrete subclasses implement iter_tracks() using the backing data.
    """

    def __init__(self, backing: AbsBackingDataset) -> None:
        self._backing = backing

    @property
    def name(self) -> str:
        return self._backing.name

    @abstractmethod
    def iter_tracks(
        self,
    ) -> Iterable[tuple[str, list[BinaryObservationEvent], TrackTemporalBounds]]: ...
