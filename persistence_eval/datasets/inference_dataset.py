from __future__ import annotations

from typing import Iterator, Literal

import torch
from torch.utils.data import IterableDataset

from persistence_eval.core.domain import ImageMaskGroup
from persistence_eval.core.protocols import ImageMaskDataset


class InferenceDataset(IterableDataset):
    """
    Wraps an ImageMaskDataset as a PyTorch IterableDataset. Always yields ImageMaskGroup
    so the inferencer receives a uniform type regardless of mode.

    mode="grouped"    — one item per image frame; masks tensor is (K, H, W) where K is
                        the number of tracked objects visible in that frame.
    mode="individual" — one item per (image, mask) pair; masks tensor is (1, H, W).
    """

    def __init__(
        self,
        view: ImageMaskDataset,
        mode: Literal["grouped", "individual"] = "grouped",
    ) -> None:
        self._view = view
        self._mode = mode

    def __iter__(self) -> Iterator[ImageMaskGroup]:
        if self._mode == "grouped":
            yield from self._view.iter_image_groups()
        else:
            for anchor in self._view.iter_anchor_observations():
                yield ImageMaskGroup(
                    image=anchor.image,
                    masks=anchor.mask.unsqueeze(0),  # (1, H, W)
                    track_ids=[anchor.track_id],
                    anchor_time=anchor.anchor_time,
                )

    def __len__(self) -> int:
        if self._mode == "grouped":
            return self._view.num_image_groups()
        return self._view.num_anchor_observations()
