import unittest

import torch

from instance_segmentation.abs_instance_segmenter import AbsInstanceSegmenter, SegmentationResults
from instance_segmentation.language_grounded_with_point_reprompt_segmenter import (
    LanguageGroundedWithPointRepromptSegmenter,
)
from instance_segmentation.point_prompted_segmentation.abs_point_prompted_segmenter import (
    AbsPointPromptedSegmenter,
    PointPromptedSegmentationResults,
)


def _make_image(h: int = 8, w: int = 8) -> torch.Tensor:
    return torch.zeros((3, h, w), dtype=torch.uint8)


class _SpyLanguageSegmenter(AbsInstanceSegmenter):
    def __init__(self, raw_results: SegmentationResults):
        super().__init__(name="spy_lang")
        self.raw_results = raw_results
        self.calls = 0

    def segmentInstancesRaw(self, image: torch.Tensor) -> SegmentationResults:
        _ = image
        self.calls += 1
        return self.raw_results


class _SpyPointSegmenter(AbsPointPromptedSegmenter):
    def __init__(self, output: PointPromptedSegmentationResults | None = None):
        super().__init__(name="spy_point")
        self.calls = 0
        self.last_point_coords = None
        self.last_point_labels = None
        self.output = output

    def segment_from_points(self, image: torch.Tensor, point_coords: torch.Tensor, point_labels=None):
        _, h, w = image.shape
        self.calls += 1
        self.last_point_coords = point_coords.clone()
        self.last_point_labels = None if point_labels is None else point_labels.clone()

        if self.output is not None:
            return self.output

        return PointPromptedSegmentationResults(
            point_ids=torch.zeros((0,), dtype=torch.int64),
            masks=torch.zeros((0, h, w), dtype=torch.bool),
            boxes=torch.zeros((0, 4), dtype=torch.float32),
            scores=torch.zeros((0,), dtype=torch.float32),
        )


def _make_seg_results(mask: torch.Tensor, score: float = 0.8) -> SegmentationResults:
    h, w = mask.shape
    return SegmentationResults(
        ids=torch.tensor([0], dtype=torch.int64),
        masks=mask.unsqueeze(0).to(torch.bool),
        boxes=torch.tensor([[0.0, 0.0, float(w), float(h)]], dtype=torch.float32),
        scores=torch.tensor([score], dtype=torch.float32),
    )


class TestLanguageGroundedWithPointRepromptSegmenter(unittest.TestCase):
    def test_no_uncovered_pixels_skips_point_pass(self):
        image = _make_image(8, 8)
        full_mask = torch.ones((8, 8), dtype=torch.bool)
        lang = _SpyLanguageSegmenter(_make_seg_results(full_mask))
        point = _SpyPointSegmenter()
        seg = LanguageGroundedWithPointRepromptSegmenter(
            language_grounded_segmenter=lang,
            point_prompted_segmenter=point,
        )

        out = seg.segmentInstances(image)
        self.assertEqual(lang.calls, 1)
        self.assertEqual(point.calls, 0)
        self.assertEqual(tuple(out.masks.shape), (1, 8, 8))

    def test_small_regions_below_min_area_are_ignored(self):
        image = _make_image(8, 8)
        mask = torch.ones((8, 8), dtype=torch.bool)
        mask[0, 0] = False
        mask[7, 7] = False
        lang = _SpyLanguageSegmenter(_make_seg_results(mask))
        point = _SpyPointSegmenter()
        seg = LanguageGroundedWithPointRepromptSegmenter(
            language_grounded_segmenter=lang,
            point_prompted_segmenter=point,
            min_reprompt_region_area_pixels=2,
        )

        _ = seg.segmentInstances(image)
        self.assertEqual(point.calls, 0)

    def test_region_above_min_area_triggers_point_pass(self):
        image = _make_image(8, 8)
        mask = torch.ones((8, 8), dtype=torch.bool)
        mask[:2, :2] = False
        lang = _SpyLanguageSegmenter(_make_seg_results(mask))
        point = _SpyPointSegmenter()
        seg = LanguageGroundedWithPointRepromptSegmenter(
            language_grounded_segmenter=lang,
            point_prompted_segmenter=point,
            min_reprompt_region_area_pixels=3,
            large_region_area_perc_for_grid=2.0,
            include_region_center_point=True,
        )

        _ = seg.segmentInstances(image)
        self.assertEqual(point.calls, 1)
        self.assertGreaterEqual(int(point.last_point_coords.shape[0]), 1)

    def test_diagonal_pixels_are_connected_with_8_connectivity(self):
        image = _make_image(5, 5)
        mask = torch.ones((5, 5), dtype=torch.bool)
        mask[1, 1] = False
        mask[2, 2] = False
        lang = _SpyLanguageSegmenter(_make_seg_results(mask))
        point = _SpyPointSegmenter()
        seg = LanguageGroundedWithPointRepromptSegmenter(
            language_grounded_segmenter=lang,
            point_prompted_segmenter=point,
            min_reprompt_region_area_pixels=2,
            large_region_area_perc_for_grid=2.0,
            include_region_center_point=True,
        )

        _ = seg.segmentInstances(image)
        self.assertEqual(point.calls, 1)
        self.assertEqual(int(point.last_point_coords.shape[0]), 1)

    def test_grid_stride_changes_sampling_density(self):
        image = _make_image(16, 16)
        empty_first = SegmentationResults.empty(h=16, w=16)
        lang = _SpyLanguageSegmenter(empty_first)

        point_dense = _SpyPointSegmenter()
        dense = LanguageGroundedWithPointRepromptSegmenter(
            language_grounded_segmenter=lang,
            point_prompted_segmenter=point_dense,
            min_reprompt_region_area_pixels=1,
            include_region_center_point=False,
            large_region_area_perc_for_grid=0.0,
            grid_stride_px=2,
            max_points_per_region=1024,
        )
        _ = dense.segmentInstances(image)

        point_sparse = _SpyPointSegmenter()
        sparse = LanguageGroundedWithPointRepromptSegmenter(
            language_grounded_segmenter=lang,
            point_prompted_segmenter=point_sparse,
            min_reprompt_region_area_pixels=1,
            include_region_center_point=False,
            large_region_area_perc_for_grid=0.0,
            grid_stride_px=4,
            max_points_per_region=1024,
        )
        _ = sparse.segmentInstances(image)

        self.assertGreater(
            int(point_dense.last_point_coords.shape[0]),
            int(point_sparse.last_point_coords.shape[0]),
        )

    def test_max_points_per_region_is_enforced(self):
        image = _make_image(16, 16)
        lang = _SpyLanguageSegmenter(SegmentationResults.empty(h=16, w=16))
        point = _SpyPointSegmenter()
        seg = LanguageGroundedWithPointRepromptSegmenter(
            language_grounded_segmenter=lang,
            point_prompted_segmenter=point,
            min_reprompt_region_area_pixels=1,
            include_region_center_point=True,
            large_region_area_perc_for_grid=0.0,
            grid_stride_px=1,
            max_points_per_region=3,
        )

        _ = seg.segmentInstances(image)
        self.assertLessEqual(int(point.last_point_coords.shape[0]), 3)

    def test_configured_point_label_is_passed_through(self):
        image = _make_image(8, 8)
        lang = _SpyLanguageSegmenter(SegmentationResults.empty(h=8, w=8))
        point = _SpyPointSegmenter()
        seg = LanguageGroundedWithPointRepromptSegmenter(
            language_grounded_segmenter=lang,
            point_prompted_segmenter=point,
            min_reprompt_region_area_pixels=1,
            large_region_area_perc_for_grid=2.0,
            point_label=0,
        )

        _ = seg.segmentInstances(image)
        self.assertTrue(torch.all(point.last_point_labels == 0))

    def test_dedupe_applies_after_merging_first_and_second_pass(self):
        image = _make_image(8, 8)
        shared = torch.zeros((8, 8), dtype=torch.bool)
        shared[:4, :4] = True

        first = SegmentationResults(
            ids=torch.tensor([0], dtype=torch.int64),
            masks=shared.unsqueeze(0),
            boxes=torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float32),
            scores=torch.tensor([0.1], dtype=torch.float32),
        )
        second = PointPromptedSegmentationResults(
            point_ids=torch.tensor([0], dtype=torch.int64),
            masks=shared.unsqueeze(0),
            boxes=torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float32),
            scores=torch.tensor([0.9], dtype=torch.float32),
        )
        lang = _SpyLanguageSegmenter(first)
        point = _SpyPointSegmenter(output=second)
        seg = LanguageGroundedWithPointRepromptSegmenter(
            language_grounded_segmenter=lang,
            point_prompted_segmenter=point,
            min_reprompt_region_area_pixels=1,
            dedupe_iou_threshold=0.9,
        )

        out = seg.segmentInstances(image)
        self.assertEqual(tuple(out.masks.shape), (1, 8, 8))
        self.assertAlmostEqual(float(out.scores.item()), 0.9, places=6)


if __name__ == "__main__":
    unittest.main()

