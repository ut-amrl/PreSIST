from __future__ import annotations

from typing import Optional
import torch

from instance_segmentation.abs_instance_segmenter import AbsInstanceSegmenter, SegmentationResults
from instance_segmentation.object_discovery.abs_object_discoverer import AbsObjectDiscoverer
from instance_segmentation.language_prompted_segmentation.abs_language_prompted_segmenter import AbsLanguagePromptedSegmenter


class LanguageGroundedSegmenter(AbsInstanceSegmenter):
    """
    AbsInstanceSegmenter implementation that composes:
      - AbsObjectDiscoverer
      - AbsLanguagePromptedSegmenter

    Pipeline:
      identify -> segment_from_prompts -> flatten -> SegmentationResults
    """

    def __init__(
        self,
        *,
        object_discoverer: AbsObjectDiscoverer,
        language_segmenter: AbsLanguagePromptedSegmenter,
        min_area_perc: float = 0,          # optional: drop tiny masks
    ):
        super().__init__(name="langGround_p_" + object_discoverer.getName() + "_s_" + language_segmenter.getName())
        self.object_discoverer = object_discoverer
        self.min_area_perc = float(min_area_perc)
        self.language_segmenter = language_segmenter

    @torch.no_grad()
    def segmentInstances(self, image: torch.Tensor) -> SegmentationResults:
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        if image.ndim != 3 or image.shape[0] not in (3, 4):
            raise ValueError("Expected image of shape (C,H,W) with C in {3,4}.")
        if image.dtype != torch.uint8:
            raise ValueError("Expected image dtype=torch.uint8.")

        C, H, W = image.shape

        obj_descriptions = self.object_discoverer.identify(image)
        # print("Object descriptions:", obj_descriptions)

        # Dedupe object descriptions if needed (some discoverers may return duplicates or near-duplicates)
        obj_descriptions = list(set(obj_descriptions))

        seg = self.language_segmenter.segment_from_prompts(image, obj_descriptions)

        # Convert to your SegmentationResults.
        # ids are prompt indices for each mask (can swap to arange if you prefer).
        if seg.masks.shape[0] == 0:
            return SegmentationResults(
                ids=torch.zeros((0,), dtype=torch.int64),
                masks=torch.zeros((0, H, W), dtype=torch.bool),
                boxes=torch.zeros((0, 4), dtype=torch.float32),
                scores=torch.zeros((0,), dtype=torch.float32),
            )

        min_area_pixels = int(self.min_area_perc * H * W)

        # Filter out small masks if min_area_perc > 0
        keep_indices = []
        for i in range(seg.masks.shape[0]):
            m = seg.masks[i]

            area = int(m.sum().item())
            if min_area_pixels > 0 and area < min_area_pixels:
                continue
            keep_indices.append(i)

        if len(keep_indices) == 0:
            return SegmentationResults(
                ids=torch.zeros((0,), dtype=torch.int64),
                masks=torch.zeros((0, H, W), dtype=torch.bool),
                boxes=torch.zeros((0, 4), dtype=torch.float32),
                scores=torch.zeros((0,), dtype=torch.float32),
            )

        filtererd_ids = seg.prompt_ids[keep_indices]
        filtererd_masks = seg.masks[keep_indices]
        filtererd_boxes = seg.boxes[keep_indices]
        filtererd_scores = seg.scores[keep_indices] if seg.scores is not None else None

        return SegmentationResults(
            ids=filtererd_ids.to(torch.int64),
            masks=filtererd_masks.to(torch.bool),
            boxes=filtererd_boxes.to(torch.float32),
            scores=filtererd_scores.to(torch.float32),
        )
