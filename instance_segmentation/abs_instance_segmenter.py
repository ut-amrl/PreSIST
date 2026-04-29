from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil
from typing import Optional, Tuple
import torch
import numpy as np
from utils.mask_utils import dedupe_mask_indices_by_iou, filter_mask_indices_by_min_area
from dataset_generation.intermediate_results_logger.debug_data import DebugPayload


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

    @staticmethod
    def empty(*, h: int, w: int) -> SegmentationResults:
        return SegmentationResults(
            ids=torch.zeros((0,), dtype=torch.int64),
            masks=torch.zeros((0, h, w), dtype=torch.bool),
            boxes=torch.zeros((0, 4), dtype=torch.float32),
            scores=torch.zeros((0,), dtype=torch.float32),
        )


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
        # data = np.load(filePath, allow_pickle=True)
        with np.load(filePath, allow_pickle=True) as data:
            ids = torch.from_numpy(data['ids'])
            masks = torch.from_numpy(data['masks'])
            boxes = torch.from_numpy(data['boxes'])
            scores = torch.from_numpy(data['scores']) if data['scores'] is not None else None
            imagePath = data['imagePath'].item() if 'imagePath' in data else None


            segmentationResults = SegmentationResults(
                ids=ids,
                masks=masks,
                boxes=boxes,
                scores=scores
            )

            return segmentationResults, imagePath

    def divideIntoSubsets(self, max_subset_size):
        """
        Divides the segmentation results into subsets of at most max_subset_size instances.
        Returns a list of SegmentationResults objects.
        """

        numGroups = ceil(len(self) / max_subset_size)
        targetGroupSize = ceil(len(self) / numGroups)

        subsets = []

        n = len(self)
        for i in range(0, n, targetGroupSize):
            subset = SegmentationResults(
                ids=self.ids[i:i+targetGroupSize],
                masks=self.masks[i:i+targetGroupSize],
                boxes=self.boxes[i:i+targetGroupSize],
                scores=self.scores[i:i+targetGroupSize] if self.scores is not None else None,
            )
            subsets.append(subset)
        return subsets


class AbsInstanceSegmenter(ABC):
    """Abstract base class for single-image instance segmentation models."""


    def __init__(
        self,
        name: str,
        *,
        min_area_perc: float = 0.0,
        dedupe_iou_threshold: float = 1.01,
        confidence_threshold: Optional[float] = None,
    ):
        self.name = name
        self.min_area_perc = float(min_area_perc)
        self.dedupe_iou_threshold = float(dedupe_iou_threshold)
        self.confidence_threshold = None if confidence_threshold is None else float(confidence_threshold)

        if self.min_area_perc < 0:
            raise ValueError("min_area_perc must be >= 0.")
        if self.dedupe_iou_threshold < 0:
            raise ValueError("dedupe_iou_threshold must be >= 0.")


    def getName(self) -> str:
        """Returns the name of this segmenter."""
        return self.name

    def segmentInstances(self, image: torch.Tensor) -> Tuple[SegmentationResults, DebugPayload]:
        """
        Args:
            image: torch.Tensor, shape (C, H, W), dtype=uint8, RGB.

        Returns:
            (SegmentationResults, DebugPayload)
        """
        self._validateImage(image)
        raw_results, debug_payload = self.segmentInstancesRaw(image)
        if not isinstance(raw_results, SegmentationResults):
            raise TypeError("segmentInstancesRaw() must return SegmentationResults.")
        if debug_payload is None:
            debug_payload = DebugPayload.empty()
        processed = self._postprocessSegmentationResults(image=image, results=raw_results)
        return processed, debug_payload

    @abstractmethod
    def segmentInstancesRaw(self, image: torch.Tensor) -> Tuple[SegmentationResults, DebugPayload]:
        """
        Model-specific segmentation implementation.

        Subclasses should return (unfiltered SegmentationResults in image space, DebugPayload).
        Subclasses with no debug data may return (results, DebugPayload.empty()) or (results, None).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _validateImage(self, image: torch.Tensor) -> None:
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        if image.ndim != 3 or image.shape[0] not in (3, 4):
            raise ValueError("Expected image of shape (C,H,W) with C in {3,4}.")
        if image.dtype != torch.uint8:
            raise ValueError("Expected image dtype=torch.uint8.")

    def _postprocessSegmentationResults(
        self,
        *,
        image: torch.Tensor,
        results: SegmentationResults,
    ) -> SegmentationResults:
        _, h, w = image.shape
        processed = self._normalizeSegmentationResults(results, h=h, w=w)

        # Print the number of masks before filtering
        # print(f"Number of masks before filtering: {len(processed)}")

        if len(processed) == 0:
            return processed

        # 1) Area filtering
        min_area_pixels = int(self.min_area_perc * h * w)
        if min_area_pixels > 0:
            keep = torch.tensor(
                filter_mask_indices_by_min_area(
                    masks=processed.masks,
                    min_area_pixels=min_area_pixels,
                ),
                dtype=torch.long,
                device=processed.ids.device,
            )
            processed = self._selectByIndices(processed, keep)
            if len(processed) == 0:
                return SegmentationResults.empty(h=h, w=w)

        # Print the number of masks after area filtering
        # print(f"Number of masks after area filtering: {len(processed)}")

        # 2) IoU-based deduplication
        dedupe_keep = dedupe_mask_indices_by_iou(
            masks=processed.masks,
            scores=processed.scores,
            iou_threshold=self.dedupe_iou_threshold,
        )
        processed = self._selectByIndices(
            processed, torch.tensor(dedupe_keep, dtype=torch.long, device=processed.ids.device)
        )
        # Print the number of masks after deduplication
        print(f"Number of masks after deduplication: {len(processed)}")
        if len(processed) == 0:
            return SegmentationResults.empty(h=h, w=w)

        # 3) Confidence thresholding
        if self.confidence_threshold is not None and processed.scores is not None:
            keep = (processed.scores >= self.confidence_threshold).nonzero(as_tuple=True)[0]
            processed = self._selectByIndices(processed, keep)

        return processed

    def _normalizeSegmentationResults(self, results: SegmentationResults, *, h: int, w: int) -> SegmentationResults:
        ids = results.ids.to(dtype=torch.int64)
        masks = results.masks.to(dtype=torch.bool)
        boxes = results.boxes.to(dtype=torch.float32)
        scores = results.scores.to(dtype=torch.float32) if results.scores is not None else None

        if masks.ndim != 3:
            raise ValueError("Expected masks to have shape (N, H, W).")
        n = masks.shape[0]
        if masks.shape[1] != h or masks.shape[2] != w:
            raise ValueError(f"Expected masks shape (N, {h}, {w}); got {tuple(masks.shape)}.")
        if ids.ndim != 1 or ids.shape[0] != n:
            raise ValueError("Expected ids shape (N,).")
        if boxes.ndim != 2 or boxes.shape[0] != n or boxes.shape[1] != 4:
            raise ValueError("Expected boxes shape (N, 4).")
        if scores is not None and (scores.ndim != 1 or scores.shape[0] != n):
            raise ValueError("Expected scores shape (N,) when provided.")

        return SegmentationResults(ids=ids, masks=masks, boxes=boxes, scores=scores)

    def _selectByIndices(self, results: SegmentationResults, indices: torch.Tensor) -> SegmentationResults:
        return SegmentationResults(
            ids=results.ids[indices],
            masks=results.masks[indices],
            boxes=results.boxes[indices],
            scores=results.scores[indices] if results.scores is not None else None,
        )

