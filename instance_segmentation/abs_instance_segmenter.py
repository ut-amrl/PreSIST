from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np


# =========================
# Core, model-agnostic API
# =========================

@dataclass
class SegmentationResults:
    """
    Results for one image.

    masks:  (N, H, W) bool tensor (False=background, True=instance)
    boxes:  (N, 4) float32 XYXY tensor
    scores: (N,) float32 tensor (model confidence if available)
    """
    ids: torch.Tensor            # int64, shape (N,)
    masks: torch.Tensor          # bool, shape (N, H, W)
    boxes: torch.Tensor          # float32, shape (N, 4) in XYXY
    scores: Optional[torch.Tensor] = None  # float32, shape (N,)

    def __len__(self) -> int:
        return self.masks.shape[0] if self.masks.ndim == 3 else 0


    # Add utility methods to save this to file
    def saveToFile(self, filePath: str, imagePath: Optional[str] = None):
        """Saves segmentation results to a .npz file."""
        np.savez_compressed(
            filePath,
            ids=self.ids.cpu().numpy(),
            masks=self.masks.cpu().numpy(),
            boxes=self.boxes.cpu().numpy(),
            scores=self.scores.cpu().numpy() if self.scores is not None else None,
            imagePath=imagePath
        )

    # Add another utility method to load from file
    @staticmethod
    def loadFromFile(filePath: str) -> SegmentationResults:
        """Loads segmentation results from a .npz file."""
        data = np.load(filePath, allow_pickle=True)
        ids = torch.from_numpy(data['ids'])
        masks = torch.from_numpy(data['masks'])
        boxes = torch.from_numpy(data['boxes'])
        scores = torch.from_numpy(data['scores']) if data['scores'] is not None else None
        imagePath = data['imagePath'].item() if 'imagePath' in data else None
        return SegmentationResults(
            ids=ids,
            masks=masks,
            boxes=boxes,
            scores=scores
        ), imagePath


class AbsInstanceSegmenter:
    """Abstract base class for single-image instance segmentation models."""


    def __init__(self, name: str):
        self.name = name


    def getName(self) -> str:
        """Returns the name of this segmenter."""
        return self.name

    def segmentInstances(self, image: torch.Tensor) -> SegmentationResults:
        """
        Args:
            image: torch.Tensor, shape (C, H, W), dtype=uint8, RGB.

        Returns:
            SegmentationResults
        """
        raise NotImplementedError("Subclasses must implement this method.")


