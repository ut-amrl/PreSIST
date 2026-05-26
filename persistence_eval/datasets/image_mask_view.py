from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import torch

from persistence_eval.core.domain import AnchorObservation, ImageMaskGroup
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

    def iter_image_groups(self) -> Iterable[ImageMaskGroup]:
        """Iterate observations grouped by image frame.
        Default wraps each AnchorObservation into a single-mask group.
        Override for datasets where multiple objects share frames.
        """
        for obs in self.iter_anchor_observations():
            yield ImageMaskGroup(
                image=obs.image,
                masks=obs.mask.unsqueeze(0),  # (1, H, W)
                track_ids=[obs.track_id],
                anchor_time=obs.anchor_time,
            )

    @abstractmethod
    def num_anchor_observations(self) -> int: ...

    def num_image_groups(self) -> int:
        """Default matches the default iter_image_groups: one group per anchor."""
        return self.num_anchor_observations()


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
