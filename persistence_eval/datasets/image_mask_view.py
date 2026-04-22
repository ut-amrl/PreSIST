from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from persistence_eval.core.domain import AnchorObservation
from persistence_eval.datasets.backing import AbsBackingDataset


class AbsImageMaskDataset(ABC):
    """
    Provides AnchorObservations for the persistence parameter inferencer.
    Implemented either directly or via ImageMaskView wrapping a BackingDataset.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def iter_anchor_observations(self) -> Iterable[AnchorObservation]: ...


class ImageMaskView(AbsImageMaskDataset):
    """
    Wraps an AbsBackingDataset to expose the ImageMaskDataset protocol.
    Concrete subclasses implement iter_anchor_observations() using the backing data.
    """

    def __init__(self, backing: AbsBackingDataset) -> None:
        self._backing = backing

    @property
    def name(self) -> str:
        return self._backing.name

    @abstractmethod
    def iter_anchor_observations(self) -> Iterable[AnchorObservation]: ...
