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
            (SegmentationResults, DebugPayload) from the wrapped segmenter before
            this wrapper's own post-processing is applied.
        """
        return self.base_segmenter.segmentInstances(image)