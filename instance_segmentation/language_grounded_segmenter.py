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
    ):
        super().__init__(name="langGround_p_" + object_discoverer.getName() + "_s_" + language_segmenter.getName())
        self.object_discoverer = object_discoverer
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

        return SegmentationResults(
            ids=seg.prompt_ids.to(torch.int64),
            masks=seg.masks.to(torch.bool),
            boxes=seg.boxes.to(torch.float32),
            scores=seg.scores.to(torch.float32),
        )
