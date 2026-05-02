from instance_segmentation.abs_instance_segmenter import AbsInstanceSegmenter, SegmentationResults
from dataset_generation.intermediate_results_logger.debug_data import DebugPayload
import torch
from typing import Optional, Tuple

class ConfidenceThresholdSegmentationWrapper(AbsInstanceSegmenter):

    def __init__(
        self,
        base_segmenter: AbsInstanceSegmenter,
        confidence_threshold: Optional[float] = None,
        min_area_perc: float = 0.0,
        dedupe_iou_threshold: float = 1.01,
    ):
        """
        Wraps an existing instance segmenter to filter results by a confidence threshold.

        Args:
            base_segmenter: An instance of AbsInstanceSegmenter to wrap.
            confidence_threshold: Minimum score for a mask to be included in the results.
        """
        super().__init__(
            name=f"{base_segmenter.getName()}_conf_thresh_{confidence_threshold}",
            min_area_perc=min_area_perc,
            dedupe_iou_threshold=dedupe_iou_threshold,
            confidence_threshold=confidence_threshold,
        )
        self.base_segmenter = base_segmenter

    def segmentInstancesRaw(self, image: torch.Tensor) -> Tuple[SegmentationResults, DebugPayload]:
        """
        Delegates segmentation to the wrapped segmenter.

        Args:
            image: torch.Tensor, shape (C, H, W), dtype=uint8, RGB.
        Returns:
            (SegmentationResults, DebugPayload) — the wrapped segmenter's
            payload would already have been flushed by its own
            ``segmentInstances`` call if it produced any debug data, but at
            this nesting level we're inside the outer wrapper's own
            ``segmentInstancesRaw`` (which has no image_id), so we just
            return an empty payload and let the outer ``segmentInstances``
            create its own configured payload.
        """
        # Use a sentinel image_id since this nested call's payload is
        # discarded; the outer wrapper's own segmentInstances() (called by
        # the logger) is what will write debug for the segmentation phase.
        results, _ = self.base_segmenter.segmentInstances(image, image_id="__nested__")
        return results, DebugPayload.empty()