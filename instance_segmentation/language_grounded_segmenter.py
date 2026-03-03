from __future__ import annotations

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
        dedupe_iou_threshold: float = 1.01,  # default > 1.0 disables dedupe
        confidence_threshold: float | None = None,
        name_suffix="",
    ):
        if (name_suffix is not ""):
            name_suffix = "_" + name_suffix
        super().__init__(
            name="langGround_p_" + object_discoverer.getName() + "_s_" + language_segmenter.getName() + name_suffix,
            min_area_perc=min_area_perc,
            dedupe_iou_threshold=dedupe_iou_threshold,
            confidence_threshold=confidence_threshold,
        )
        self.object_discoverer = object_discoverer
        self.language_segmenter = language_segmenter

    @torch.no_grad()
    def segmentInstancesRaw(self, image: torch.Tensor) -> SegmentationResults:
        _, H, W = image.shape

        obj_descriptions = self.object_discoverer.identify(image)
        # print("Object descriptions:", obj_descriptions)

        # Dedupe object descriptions if needed (some discoverers may return duplicates or near-duplicates)
        obj_descriptions = list(set(obj_descriptions))

        seg = self.language_segmenter.segment_from_prompts(image, obj_descriptions)

        # Convert to your SegmentationResults.
        # ids are prompt indices for each mask (can swap to arange if you prefer).
        if seg.masks.shape[0] == 0:
            return SegmentationResults.empty(h=H, w=W)

        return SegmentationResults(
            ids=seg.prompt_ids.to(torch.int64),
            masks=seg.masks.to(torch.bool),
            boxes=seg.boxes.to(torch.float32),
            scores=seg.scores.to(torch.float32) if seg.scores is not None else None,
        )
