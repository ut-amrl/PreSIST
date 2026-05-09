# Run inside container: python -m unittest discover -s tests -p "test_*.py" -v

import unittest

import torch
from instance_segmentation.abs_instance_segmenter import (
    AbsInstanceSegmenter,
    SegmentationResults,
)


class _DummySegmenter(AbsInstanceSegmenter):
    def __init__(self, raw_results: SegmentationResults, **kwargs):
        super().__init__(name="dummy", **kwargs)
        self._raw_results = raw_results

    def segmentInstancesRaw(self, image: torch.Tensor) -> SegmentationResults:
        return self._raw_results


class _BadReturnSegmenter(AbsInstanceSegmenter):
    def __init__(self):
        super().__init__(name="bad_return")

    def segmentInstancesRaw(self, image: torch.Tensor):  # intentionally wrong return type
        return {"not": "segmentation_results"}


def _make_test_image(h: int = 4, w: int = 4) -> torch.Tensor:
    return torch.zeros((3, h, w), dtype=torch.uint8)


def _full_mask(h: int, w: int) -> torch.Tensor:
    return torch.ones((h, w), dtype=torch.bool)


def _single_pixel_mask(h: int, w: int, y: int, x: int) -> torch.Tensor:
    mask = torch.zeros((h, w), dtype=torch.bool)
    mask[y, x] = True
    return mask


def _rect_mask(h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> torch.Tensor:
    mask = torch.zeros((h, w), dtype=torch.bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _make_results(
    ids: list[int],
    masks: torch.Tensor,
    scores: torch.Tensor | None = None,
    boxes: torch.Tensor | None = None,
) -> SegmentationResults:
    n = masks.shape[0]
    if boxes is None:
        h, w = masks.shape[1], masks.shape[2]
        boxes = torch.tensor([[0, 0, w, h] for _ in range(n)], dtype=torch.float32)
    return SegmentationResults(
        ids=torch.tensor(ids, dtype=torch.int64),
        masks=masks,
        boxes=boxes,
        scores=scores,
    )


class TestAbsInstanceSegmenterFiltering(unittest.TestCase):
    def test_min_area_filter_removes_small_masks(self):
        image = _make_test_image(4, 4)  # total area = 16
        masks = torch.zeros((2, 4, 4), dtype=torch.bool)
        masks[0, 0, 0] = True  # area = 1
        masks[1, :, :] = True  # area = 16
        raw = SegmentationResults(
            ids=torch.tensor([10, 20]),
            masks=masks,
            boxes=torch.tensor([[0, 0, 1, 1], [0, 0, 4, 4]], dtype=torch.float32),
            scores=torch.tensor([0.2, 0.8], dtype=torch.float32),
        )

        segmenter = _DummySegmenter(raw, min_area_perc=0.5)
        out = segmenter.segmentInstances(image)

        self.assertEqual(out.ids.tolist(), [20])
        self.assertEqual(tuple(out.masks.shape), (1, 4, 4))

    def test_iou_dedupe_keeps_higher_scored_duplicate(self):
        image = _make_test_image(4, 4)
        duplicate_mask = torch.ones((4, 4), dtype=torch.bool)
        masks = torch.stack([duplicate_mask, duplicate_mask], dim=0)
        raw = SegmentationResults(
            ids=torch.tensor([100, 200]),
            masks=masks,
            boxes=torch.tensor([[0, 0, 4, 4], [0, 0, 4, 4]], dtype=torch.float32),
            scores=torch.tensor([0.1, 0.9], dtype=torch.float32),
        )

        segmenter = _DummySegmenter(raw, dedupe_iou_threshold=0.9)
        out = segmenter.segmentInstances(image)

        self.assertEqual(out.ids.tolist(), [200])
        self.assertEqual(tuple(out.masks.shape), (1, 4, 4))

    def test_confidence_threshold_filters_low_scores(self):
        image = _make_test_image(4, 4)
        masks = torch.zeros((2, 4, 4), dtype=torch.bool)
        masks[0, :2, :2] = True
        masks[1, 2:, 2:] = True
        raw = SegmentationResults(
            ids=torch.tensor([1, 2]),
            masks=masks,
            boxes=torch.tensor([[0, 0, 2, 2], [2, 2, 4, 4]], dtype=torch.float32),
            scores=torch.tensor([0.4, 0.95], dtype=torch.float32),
        )

        segmenter = _DummySegmenter(raw, confidence_threshold=0.9)
        out = segmenter.segmentInstances(image)

        self.assertEqual(out.ids.tolist(), [2])
        self.assertAlmostEqual(float(out.scores.item()), 0.95, places=6)

    def test_normalization_casts_output_dtypes(self):
        image = _make_test_image(4, 4)
        raw = SegmentationResults(
            ids=torch.tensor([7], dtype=torch.int32),
            masks=torch.ones((1, 4, 4), dtype=torch.uint8),
            boxes=torch.tensor([[0, 0, 4, 4]], dtype=torch.float64),
            scores=torch.tensor([1.0], dtype=torch.float64),
        )

        segmenter = _DummySegmenter(raw)
        out = segmenter.segmentInstances(image)

        self.assertEqual(out.ids.dtype, torch.int64)
        self.assertEqual(out.masks.dtype, torch.bool)
        self.assertEqual(out.boxes.dtype, torch.float32)
        self.assertEqual(out.scores.dtype, torch.float32)

    def test_empty_input_passthrough(self):
        image = _make_test_image(4, 4)
        raw = SegmentationResults.empty(h=4, w=4)

        out = _DummySegmenter(raw).segmentInstances(image)

        self.assertEqual(tuple(out.ids.shape), (0,))
        self.assertEqual(tuple(out.masks.shape), (0, 4, 4))
        self.assertEqual(tuple(out.boxes.shape), (0, 4))
        self.assertEqual(tuple(out.scores.shape), (0,))
        self.assertEqual(out.ids.dtype, torch.int64)
        self.assertEqual(out.masks.dtype, torch.bool)
        self.assertEqual(out.boxes.dtype, torch.float32)
        self.assertEqual(out.scores.dtype, torch.float32)

    def test_area_threshold_boundary_keeps_equal_area_mask(self):
        image = _make_test_image(4, 4)  # 16 pixels total, 0.25 => 4 pixels minimum
        masks = torch.zeros((2, 4, 4), dtype=torch.bool)
        masks[0, :2, :2] = True  # area = 4, should be kept
        masks[1, 0, 0] = True    # area = 1, should be removed
        raw = _make_results(
            ids=[1, 2],
            masks=masks,
            scores=torch.tensor([0.7, 0.8], dtype=torch.float32),
        )

        out = _DummySegmenter(raw, min_area_perc=0.25).segmentInstances(image)
        self.assertEqual(out.ids.tolist(), [1])

    def test_area_filter_can_remove_all_masks(self):
        image = _make_test_image(4, 4)  # 16 pixels total, 0.5 => 8 pixels minimum
        masks = torch.zeros((2, 4, 4), dtype=torch.bool)
        masks[0, :2, :2] = True  # area = 4
        masks[1, 0, 0] = True    # area = 1
        raw = _make_results(
            ids=[10, 20],
            masks=masks,
            scores=torch.tensor([0.9, 0.9], dtype=torch.float32),
        )

        out = _DummySegmenter(raw, min_area_perc=0.5).segmentInstances(image)
        self.assertEqual(tuple(out.masks.shape), (0, 4, 4))
        self.assertEqual(tuple(out.ids.shape), (0,))

    def test_dedupe_without_scores_keeps_lower_index(self):
        image = _make_test_image(4, 4)
        duplicate = _full_mask(4, 4)
        masks = torch.stack([duplicate, duplicate], dim=0)
        raw = _make_results(ids=[5, 6], masks=masks, scores=None)

        out = _DummySegmenter(raw, dedupe_iou_threshold=0.9).segmentInstances(image)
        self.assertEqual(out.ids.tolist(), [5])
        self.assertIsNone(out.scores)

    def test_dedupe_disabled_for_threshold_above_one(self):
        image = _make_test_image(4, 4)
        duplicate = _full_mask(4, 4)
        masks = torch.stack([duplicate, duplicate], dim=0)
        raw = _make_results(
            ids=[100, 200],
            masks=masks,
            scores=torch.tensor([0.1, 0.9], dtype=torch.float32),
        )

        out = _DummySegmenter(raw, dedupe_iou_threshold=1.01).segmentInstances(image)
        self.assertEqual(out.ids.tolist(), [100, 200])

    def test_dedupe_keeps_masks_when_overlap_below_threshold(self):
        image = _make_test_image(4, 4)
        mask_a = _rect_mask(4, 4, 0, 2, 0, 2)  # area 4
        mask_b = _rect_mask(4, 4, 1, 3, 1, 3)  # area 4, overlap 1 => IoU 1/7
        masks = torch.stack([mask_a, mask_b], dim=0)
        raw = _make_results(
            ids=[30, 40],
            masks=masks,
            scores=torch.tensor([0.8, 0.9], dtype=torch.float32),
        )

        out = _DummySegmenter(raw, dedupe_iou_threshold=0.2).segmentInstances(image)
        self.assertEqual(out.ids.tolist(), [30, 40])

    def test_dedupe_equal_iou_threshold_keeps_both(self):
        image = _make_test_image(4, 4)
        # Two 2x3 masks shifted by one column: IoU = 4/8 = 0.5 exactly.
        mask_a = _rect_mask(4, 4, 0, 2, 0, 3)
        mask_b = _rect_mask(4, 4, 0, 2, 1, 4)
        masks = torch.stack([mask_a, mask_b], dim=0)
        raw = _make_results(
            ids=[101, 202],
            masks=masks,
            scores=torch.tensor([0.6, 0.5], dtype=torch.float32),
        )

        out = _DummySegmenter(raw, dedupe_iou_threshold=0.5).segmentInstances(image)
        self.assertEqual(out.ids.tolist(), [101, 202])

    def test_dedupe_threshold_one_keeps_identical_masks(self):
        image = _make_test_image(4, 4)
        duplicate = _full_mask(4, 4)
        masks = torch.stack([duplicate, duplicate], dim=0)
        raw = _make_results(
            ids=[51, 52],
            masks=masks,
            scores=torch.tensor([0.7, 0.9], dtype=torch.float32),
        )

        out = _DummySegmenter(raw, dedupe_iou_threshold=1.0).segmentInstances(image)
        self.assertEqual(out.ids.tolist(), [51, 52])

    def test_dedupe_tie_breaker_uses_lower_index_for_equal_scores(self):
        image = _make_test_image(4, 4)
        duplicate = _full_mask(4, 4)
        masks = torch.stack([duplicate, duplicate], dim=0)
        raw = _make_results(
            ids=[71, 72],
            masks=masks,
            scores=torch.tensor([0.8, 0.8], dtype=torch.float32),
        )

        out = _DummySegmenter(raw, dedupe_iou_threshold=0.9).segmentInstances(image)
        self.assertEqual(out.ids.tolist(), [71])

    def test_dedupe_handles_multiple_duplicate_groups_and_non_overlapping_masks(self):
        image = _make_test_image(6, 6)
        # Group 1 duplicates
        g1_a = _rect_mask(6, 6, 0, 2, 0, 2)
        g1_b = _rect_mask(6, 6, 0, 2, 0, 2)
        # Group 2 duplicates
        g2_a = _rect_mask(6, 6, 3, 6, 3, 6)
        g2_b = _rect_mask(6, 6, 3, 6, 3, 6)
        # Non-overlapping masks that should remain untouched by dedupe
        keep_a = _single_pixel_mask(6, 6, 0, 5)
        keep_b = _single_pixel_mask(6, 6, 5, 0)
        masks = torch.stack([g1_a, g1_b, g2_a, g2_b, keep_a, keep_b], dim=0)
        raw = _make_results(
            ids=[1, 2, 3, 4, 5, 6],
            masks=masks,
            scores=torch.tensor([0.1, 0.9, 0.95, 0.2, 0.6, 0.7], dtype=torch.float32),
        )

        out = _DummySegmenter(raw, dedupe_iou_threshold=0.9).segmentInstances(image)
        self.assertEqual(out.ids.tolist(), [2, 3, 5, 6])

    def test_confidence_threshold_ignored_when_scores_missing(self):
        image = _make_test_image(4, 4)
        masks = torch.stack([_single_pixel_mask(4, 4, 0, 0), _single_pixel_mask(4, 4, 1, 1)], dim=0)
        raw = _make_results(ids=[1, 2], masks=masks, scores=None)

        out = _DummySegmenter(raw, confidence_threshold=0.95).segmentInstances(image)
        self.assertEqual(out.ids.tolist(), [1, 2])
        self.assertIsNone(out.scores)

    def test_confidence_threshold_keeps_equal_boundary(self):
        image = _make_test_image(4, 4)
        masks = torch.stack([_single_pixel_mask(4, 4, 0, 0), _single_pixel_mask(4, 4, 1, 1)], dim=0)
        raw = _make_results(
            ids=[7, 8],
            masks=masks,
            scores=torch.tensor([0.9, 0.8999], dtype=torch.float32),
        )

        out = _DummySegmenter(raw, confidence_threshold=0.9).segmentInstances(image)
        self.assertEqual(out.ids.tolist(), [7])
        self.assertAlmostEqual(float(out.scores.item()), 0.9, places=6)

    def test_confidence_threshold_can_remove_all_masks(self):
        image = _make_test_image(4, 4)
        masks = torch.stack([_single_pixel_mask(4, 4, 0, 0), _single_pixel_mask(4, 4, 1, 1)], dim=0)
        raw = _make_results(
            ids=[11, 12],
            masks=masks,
            scores=torch.tensor([0.1, 0.2], dtype=torch.float32),
        )

        out = _DummySegmenter(raw, confidence_threshold=0.95).segmentInstances(image)
        self.assertEqual(tuple(out.masks.shape), (0, 4, 4))
        self.assertEqual(tuple(out.ids.shape), (0,))
        self.assertEqual(tuple(out.scores.shape), (0,))

    def test_filter_order_area_then_dedupe_then_confidence(self):
        image = _make_test_image(4, 4)  # total area = 16, min area at 0.25 => 4
        small_high_score = torch.zeros((4, 4), dtype=torch.bool)
        small_high_score[0, :3] = True  # area = 3, inside large mask
        large_lower_score = torch.zeros((4, 4), dtype=torch.bool)
        large_lower_score[:3, :3] = True  # area = 9, IoU with small mask = 3/9 = 0.333...
        masks = torch.stack([small_high_score, large_lower_score], dim=0)
        raw = _make_results(
            ids=[1, 2],
            masks=masks,
            scores=torch.tensor([0.9, 0.8], dtype=torch.float32),
        )

        out = _DummySegmenter(
            raw,
            min_area_perc=0.25,
            dedupe_iou_threshold=0.2,
            confidence_threshold=0.75,
        ).segmentInstances(image)
        self.assertEqual(out.ids.tolist(), [2])


class TestAbsInstanceSegmenterValidation(unittest.TestCase):
    def test_constructor_rejects_negative_area(self):
        with self.assertRaises(ValueError):
            _DummySegmenter(SegmentationResults.empty(h=4, w=4), min_area_perc=-1e-6)

    def test_constructor_rejects_negative_iou_threshold(self):
        with self.assertRaises(ValueError):
            _DummySegmenter(SegmentationResults.empty(h=4, w=4), dedupe_iou_threshold=-1e-6)

    def test_segment_instances_rejects_invalid_image_shape(self):
        raw = SegmentationResults.empty(h=4, w=4)
        segmenter = _DummySegmenter(raw)
        with self.assertRaises(ValueError):
            segmenter.segmentInstances(torch.zeros((4, 4), dtype=torch.uint8))

    def test_segment_instances_rejects_invalid_image_channels(self):
        raw = SegmentationResults.empty(h=4, w=4)
        segmenter = _DummySegmenter(raw)
        with self.assertRaises(ValueError):
            segmenter.segmentInstances(torch.zeros((2, 4, 4), dtype=torch.uint8))

    def test_segment_instances_rejects_invalid_image_dtype(self):
        raw = SegmentationResults.empty(h=4, w=4)
        segmenter = _DummySegmenter(raw)
        with self.assertRaises(ValueError):
            segmenter.segmentInstances(torch.zeros((3, 4, 4), dtype=torch.float32))

    def test_segment_instances_rejects_non_segmentation_results_output(self):
        image = _make_test_image(4, 4)
        with self.assertRaises(TypeError):
            _BadReturnSegmenter().segmentInstances(image)

    def test_normalize_rejects_mask_shape_mismatch(self):
        segmenter = _DummySegmenter(SegmentationResults.empty(h=4, w=4))
        bad = SegmentationResults(
            ids=torch.tensor([1]),
            masks=torch.ones((1, 3, 4), dtype=torch.bool),
            boxes=torch.tensor([[0, 0, 4, 4]], dtype=torch.float32),
            scores=torch.tensor([0.5], dtype=torch.float32),
        )
        with self.assertRaises(ValueError):
            segmenter._normalizeSegmentationResults(bad, h=4, w=4)

    def test_normalize_rejects_ids_length_mismatch(self):
        segmenter = _DummySegmenter(SegmentationResults.empty(h=4, w=4))
        bad = SegmentationResults(
            ids=torch.tensor([1, 2], dtype=torch.int64),
            masks=torch.ones((1, 4, 4), dtype=torch.bool),
            boxes=torch.tensor([[0, 0, 4, 4]], dtype=torch.float32),
            scores=torch.tensor([0.5], dtype=torch.float32),
        )
        with self.assertRaises(ValueError):
            segmenter._normalizeSegmentationResults(bad, h=4, w=4)

    def test_normalize_rejects_boxes_shape_mismatch(self):
        segmenter = _DummySegmenter(SegmentationResults.empty(h=4, w=4))
        bad = SegmentationResults(
            ids=torch.tensor([1], dtype=torch.int64),
            masks=torch.ones((1, 4, 4), dtype=torch.bool),
            boxes=torch.tensor([[0, 0, 4]], dtype=torch.float32),
            scores=torch.tensor([0.5], dtype=torch.float32),
        )
        with self.assertRaises(ValueError):
            segmenter._normalizeSegmentationResults(bad, h=4, w=4)

    def test_normalize_rejects_scores_length_mismatch(self):
        segmenter = _DummySegmenter(SegmentationResults.empty(h=4, w=4))
        bad = SegmentationResults(
            ids=torch.tensor([1], dtype=torch.int64),
            masks=torch.ones((1, 4, 4), dtype=torch.bool),
            boxes=torch.tensor([[0, 0, 4, 4]], dtype=torch.float32),
            scores=torch.tensor([0.5, 0.6], dtype=torch.float32),
        )
        with self.assertRaises(ValueError):
            segmenter._normalizeSegmentationResults(bad, h=4, w=4)


if __name__ == "__main__":
    unittest.main()
