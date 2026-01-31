from instance_segmentation.abs_instance_segmenter import AbsInstanceSegmenter, SegmentationResults

class ConfidenceThresholdSegmentationWrapper(AbsInstanceSegmenter):

    def __init__(self, base_segmenter: AbsInstanceSegmenter, confidence_threshold: float):
        """
        Wraps an existing instance segmenter to filter results by a confidence threshold.

        Args:
            base_segmenter: An instance of AbsInstanceSegmenter to wrap.
            confidence_threshold: Minimum score for a mask to be included in the results.
        """
        super().__init__(name=f"{base_segmenter.getName()}_conf_thresh_{confidence_threshold}")
        self.base_segmenter = base_segmenter
        self.confidence_threshold = confidence_threshold


    def segmentInstances(self, image: torch.Tensor) -> SegmentationResults:
        """
        Segments instances in the image and filters them by confidence threshold.

        Args:
            image: torch.Tensor, shape (C, H, W), dtype=uint8, RGB.
        Returns:
            SegmentationResults with masks filtered by confidence threshold.
        """
        seg_results = self.base_segmenter.segmentInstances(image)

        if seg_results.scores is None:
            return seg_results  # No scores to filter on

        # Filter masks, boxes, and scores based on the confidence threshold
        keep_indices = (seg_results.scores >= self.confidence_threshold).nonzero(as_tuple=True)[0]

        filtered_ids = seg_results.ids[keep_indices]
        filtered_masks = seg_results.masks[keep_indices]
        filtered_boxes = seg_results.boxes[keep_indices]
        filtered_scores = seg_results.scores[keep_indices]

        return SegmentationResults(
            ids=filtered_ids,
            masks=filtered_masks,
            boxes=filtered_boxes,
            scores=filtered_scores
        )