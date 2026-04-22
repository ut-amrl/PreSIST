from __future__ import annotations

from abc import ABC, abstractmethod


class AbsBackingDataset(ABC):
    """
    Raw data access for a dataset. Concrete datasets subclass this.
    ImageMaskView and TrackView wrap an AbsBackingDataset instance to provide
    protocol-conformant iteration for inference and evaluation respectively.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...
