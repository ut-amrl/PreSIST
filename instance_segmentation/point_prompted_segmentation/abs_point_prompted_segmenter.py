from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PointPromptedSegmentationResults:
    """
    Flat outputs for an image + multiple point prompts.

    point_ids: (N,) int64 - which input point produced each mask
    masks:     (N,H,W) bool
    boxes:     (N,4) float32 XYXY
    scores:    (N,) float32 (0 if unavailable)
    """

    point_ids: torch.Tensor
    masks: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor

    # Add a length method
    def __len__(self) -> int:
        return len(self.masks)


class AbsPointPromptedSegmenter:
    """
    Given an image and point prompts, return 0..K masks per point (flattened).
    """

    def __init__(self, name: str):
        self.name = name

    def getName(self) -> str:
        return self.name

    def segment_from_points(
        self,
        image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: Optional[torch.Tensor] = None,
    ) -> PointPromptedSegmentationResults:
        raise NotImplementedError

