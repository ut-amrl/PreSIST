# Run inside container: python -m unittest discover -s tests -p "test_*.py" -v

import unittest
from unittest.mock import patch

import numpy as np
import torch
from instance_segmentation.abs_instance_segmenter import AbsInstanceSegmenter, SegmentationResults
from instance_segmentation.language_grounded_segmenter import LanguageGroundedSegmenter
from instance_segmentation.language_grounded_with_point_reprompt_segmenter import (
    LanguageGroundedWithPointRepromptSegmenter,
)
from instance_segmentation.language_prompted_segmentation.abs_language_prompted_segmenter import (
    AbsLanguagePromptedSegmenter,
    LanguagePromptedSegmentationResults,
)
from instance_segmentation.object_discovery.abs_object_discoverer import AbsObjectDiscoverer
from instance_segmentation.point_prompted_segmentation.abs_point_prompted_segmenter import (
    AbsPointPromptedSegmenter,
    PointPromptedSegmentationResults,
)


def _make_image(h: int = 6, w: int = 8) -> torch.Tensor:
    return torch.zeros((3, h, w), dtype=torch.uint8)


class _FakeObjectDiscoverer(AbsObjectDiscoverer):
    def __init__(self):
        super().__init__(name="fake_discoverer")

    def identify(self, image: torch.Tensor):
        return ["cat", "dog"]


class _FakeLanguageSegmenter(AbsLanguagePromptedSegmenter):
    def __init__(self):
        super().__init__(name="fake_lang_seg")

    def segment_from_prompts(self, image: torch.Tensor, prompts):
        _, h, w = image.shape
        masks = torch.zeros((2, h, w), dtype=torch.bool)
        masks[0, : h // 2, : w // 2] = True
        masks[1, h // 2 :, w // 2 :] = True
        return LanguagePromptedSegmentationResults(
            prompt_ids=torch.tensor([0, 1], dtype=torch.int64),
            masks=masks,
            boxes=torch.tensor(
                [[0.0, 0.0, float(w // 2), float(h // 2)], [float(w // 2), float(h // 2), float(w), float(h)]],
                dtype=torch.float32,
            ),
            scores=torch.tensor([0.7, 0.8], dtype=torch.float32),
        )


class _FakeMaskGenerator:
    def __init__(self, outputs):
        self._outputs = outputs

    def generate(self, _image):
        return self._outputs


class _FakeSemanticSamModel:
    def to(self, _device):
        return self

    def eval(self):
        return self


class _FakePointPromptedSegmenter(AbsPointPromptedSegmenter):
    def __init__(self):
        super().__init__(name="fake_point_seg")
        self.calls = 0
        self.last_num_points = 0

    def segment_from_points(self, image: torch.Tensor, point_coords: torch.Tensor, point_labels=None):
        self.calls += 1
        self.last_num_points = int(point_coords.shape[0])
        _, h, w = image.shape
        if point_coords.shape[0] == 0:
            return PointPromptedSegmentationResults(
                point_ids=torch.zeros((0,), dtype=torch.int64),
                masks=torch.zeros((0, h, w), dtype=torch.bool),
                boxes=torch.zeros((0, 4), dtype=torch.float32),
                scores=torch.zeros((0,), dtype=torch.float32),
            )

        mask = torch.zeros((h, w), dtype=torch.bool)
        x = int(point_coords[0, 0].item())
        y = int(point_coords[0, 1].item())
        x = min(max(x, 0), w - 1)
        y = min(max(y, 0), h - 1)
        mask[y, x] = True
        return PointPromptedSegmentationResults(
            point_ids=torch.tensor([0], dtype=torch.int64),
            masks=mask.unsqueeze(0),
            boxes=torch.tensor([[float(x), float(y), float(x + 1), float(y + 1)]], dtype=torch.float32),
            scores=torch.tensor([0.6], dtype=torch.float32),
        )


class _FakeLanguageGroundedBase(AbsInstanceSegmenter):
    def __init__(self):
        super().__init__(name="fake_lang_grounded")

    def segmentInstancesRaw(self, image: torch.Tensor) -> SegmentationResults:
        _, h, w = image.shape
        masks = torch.zeros((1, h, w), dtype=torch.bool)
        masks[0, : h // 2, : w // 2] = True
        return SegmentationResults(
            ids=torch.tensor([0], dtype=torch.int64),
            masks=masks,
            boxes=torch.tensor([[0.0, 0.0, float(w // 2), float(h // 2)]], dtype=torch.float32),
            scores=torch.tensor([0.9], dtype=torch.float32),
        )


class _FakeSam3Predictor:
    def set_image(self, _image):
        return None

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        multimask_output=True,
        return_logits=False,
        normalize_coords=False,
    ):
        _ = (point_coords, point_labels, multimask_output, return_logits, normalize_coords)
        masks = np.zeros((2, 6, 8), dtype=np.bool_)
        masks[0, 0:2, 0:2] = True
        masks[1, 3:5, 4:7] = True
        scores = np.array([0.8, 0.55], dtype=np.float32)
        low_res = np.zeros((2, 256, 256), dtype=np.float32)
        return masks, scores, low_res

    def reset_predictor(self):
        return None


class TestConcreteSegmenterShapes(unittest.TestCase):
    def test_language_grounded_segmenter_output_shapes(self):
        image = _make_image(6, 8)
        seg = LanguageGroundedSegmenter(
            object_discoverer=_FakeObjectDiscoverer(),
            language_segmenter=_FakeLanguageSegmenter(),
        )
        out = seg.segmentInstances(image)

        self.assertEqual(tuple(out.ids.shape), (2,))
        self.assertEqual(tuple(out.masks.shape), (2, 6, 8))
        self.assertEqual(tuple(out.boxes.shape), (2, 4))
        self.assertEqual(tuple(out.scores.shape), (2,))

    def test_sam2_segmenter_output_shapes_with_mocked_generator(self):
        try:
            import instance_segmentation.sam2_auto_segmenter as sam2_mod
        except ModuleNotFoundError as exc:
            self.skipTest(f"SAM2 dependency not available: {exc}")

        image = _make_image(6, 8)
        raw = [
            {
                "segmentation": np.ones((6, 8), dtype=np.bool_),
                "bbox": [0.0, 0.0, 8.0, 6.0],
                "predicted_iou": 0.9,
            }
        ]

        with patch.object(sam2_mod, "customBuildSam2", return_value=object()), patch.object(
            sam2_mod, "SAM2AutomaticMaskGenerator", return_value=_FakeMaskGenerator(raw)
        ):
            seg = sam2_mod.SAM2InstanceSegmenter(
                checkpoint="unused.pt",
                model_cfg="unused.yaml",
                name="sam2_test",
            )
            out = seg.segmentInstances(image)

        self.assertEqual(tuple(out.ids.shape), (1,))
        self.assertEqual(tuple(out.masks.shape), (1, 6, 8))
        self.assertEqual(tuple(out.boxes.shape), (1, 4))
        self.assertEqual(tuple(out.scores.shape), (1,))

    def test_sam3_segmenter_output_shapes_with_mocked_generator(self):
        try:
            import instance_segmentation.sam3_auto_segmenter as sam3_mod
        except ModuleNotFoundError as exc:
            self.skipTest(f"SAM3 dependency not available: {exc}")

        image = _make_image(6, 8)
        raw = [
            {
                "segmentation": np.ones((6, 8), dtype=np.bool_),
                "bbox": [1.0, 1.0, 4.0, 3.0],
                "predicted_iou": 0.75,
            }
        ]

        with patch.object(
            sam3_mod.SAM3AutomaticMaskGenerator,
            "from_pretrained",
            return_value=_FakeMaskGenerator(raw),
        ):
            seg = sam3_mod.SAM3InstanceSegmenter(name="sam3_test")
            out = seg.segmentInstances(image)

        self.assertEqual(tuple(out.ids.shape), (1,))
        self.assertEqual(tuple(out.masks.shape), (1, 6, 8))
        self.assertEqual(tuple(out.boxes.shape), (1, 4))
        self.assertEqual(tuple(out.scores.shape), (1,))

    def test_semantic_sam_segmenter_output_shapes_with_mocked_generator(self):
        try:
            import instance_segmentation.semantic_sam_auto_segmenter as ss_mod
        except ModuleNotFoundError as exc:
            self.skipTest(f"SemanticSAM dependency not available: {exc}")

        image = _make_image(6, 8)
        raw = [
            {
                "segmentation": np.ones((6, 8), dtype=np.bool_),
                "bbox": [0.0, 0.0, 8.0, 6.0],
                "predicted_iou": 0.88,
            }
        ]

        with patch.object(ss_mod, "customBuildSemanticSam", return_value=_FakeSemanticSamModel()), patch.object(
            ss_mod, "SemanticSamAutomaticMaskGenerator", return_value=_FakeMaskGenerator(raw)
        ):
            seg = ss_mod.SemanticSAMInstanceSegmenter(
                checkpoint="unused.pt",
                model_type="T",
                name="sem_sam_test",
                resize_short_edge=6,
            )
            out = seg.segmentInstances(image)

        self.assertEqual(tuple(out.ids.shape), (1,))
        self.assertEqual(tuple(out.masks.shape), (1, 6, 8))
        self.assertEqual(tuple(out.boxes.shape), (1, 4))
        self.assertEqual(tuple(out.scores.shape), (1,))

    def test_sam3_point_prompted_segmenter_output_shapes_with_mocked_predictor(self):
        try:
            from instance_segmentation.point_prompted_segmentation.sam3_point_prompted_segmenter import (
                SAM3PointPromptedSegmenter,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"SAM3 dependency not available: {exc}")

        image = _make_image(6, 8)
        seg = SAM3PointPromptedSegmenter(
            name="sam3_point_test",
            predictor=_FakeSam3Predictor(),
            score_thresh=0.6,
        )
        out = seg.segment_from_points(
            image=image,
            point_coords=torch.tensor([[2.0, 1.0]], dtype=torch.float32),
            point_labels=torch.tensor([1], dtype=torch.int64),
        )

        self.assertEqual(tuple(out.point_ids.shape), (1,))
        self.assertEqual(tuple(out.masks.shape), (1, 6, 8))
        self.assertEqual(tuple(out.boxes.shape), (1, 4))
        self.assertEqual(tuple(out.scores.shape), (1,))

    def test_language_grounded_with_point_reprompt_merges_outputs(self):
        image = _make_image(6, 8)
        point_seg = _FakePointPromptedSegmenter()
        seg = LanguageGroundedWithPointRepromptSegmenter(
            language_grounded_segmenter=_FakeLanguageGroundedBase(),
            point_prompted_segmenter=point_seg,
            min_reprompt_region_area_pixels=1,
            large_region_area_perc_for_grid=2.0,
            max_points_per_region=1,
            dedupe_iou_threshold=1.01,
        )
        out = seg.segmentInstances(image)

        self.assertEqual(point_seg.calls, 1)
        self.assertEqual(tuple(out.ids.shape), (2,))
        self.assertEqual(tuple(out.masks.shape), (2, 6, 8))
        self.assertEqual(tuple(out.boxes.shape), (2, 4))
        self.assertEqual(tuple(out.scores.shape), (2,))

    def test_language_grounded_with_point_reprompt_skips_small_regions(self):
        image = _make_image(6, 8)
        point_seg = _FakePointPromptedSegmenter()
        seg = LanguageGroundedWithPointRepromptSegmenter(
            language_grounded_segmenter=_FakeLanguageGroundedBase(),
            point_prompted_segmenter=point_seg,
            min_reprompt_region_area_pixels=10_000,  # bigger than any uncovered component
            large_region_area_perc_for_grid=2.0,
            max_points_per_region=1,
            dedupe_iou_threshold=1.01,
        )
        out = seg.segmentInstances(image)

        self.assertEqual(point_seg.calls, 0)
        self.assertEqual(tuple(out.ids.shape), (1,))
        self.assertEqual(tuple(out.masks.shape), (1, 6, 8))


if __name__ == "__main__":
    unittest.main()
