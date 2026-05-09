import unittest

import numpy as np
import torch


def _make_image(h: int = 6, w: int = 8) -> torch.Tensor:
    return torch.zeros((3, h, w), dtype=torch.uint8)


class _FakePredictor:
    def __init__(self):
        self.set_calls = 0
        self.reset_calls = 0
        self.predict_calls = 0
        self.last_labels = []

    def set_image(self, _image):
        self.set_calls += 1

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        multimask_output=True,
        return_logits=False,
        normalize_coords=False,
    ):
        _ = (multimask_output, return_logits, normalize_coords)
        self.predict_calls += 1
        self.last_labels.append(np.asarray(point_labels).copy())

        x = int(point_coords[0, 0])
        y = int(point_coords[0, 1])
        h, w = 6, 8

        m1 = np.zeros((h, w), dtype=np.bool_)
        m2 = np.zeros((h, w), dtype=np.bool_)
        m1[max(0, min(y, h - 1)), max(0, min(x, w - 1))] = True
        m2[max(0, min(y + 1, h - 1)), max(0, min(x + 1, w - 1))] = True

        masks = np.stack([m1, m2], axis=0)
        scores = np.array([0.9, 0.4], dtype=np.float32)
        low_res = np.zeros((2, 256, 256), dtype=np.float32)
        return masks, scores, low_res

    def reset_predictor(self):
        self.reset_calls += 1


class TestSAM3PointPromptedSegmenter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from instance_segmentation.point_prompted_segmentation.sam3_point_prompted_segmenter import (  # noqa: E501
                SAM3PointPromptedSegmenter,
            )
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"SAM3 dependency not available: {exc}")
        cls.SAM3PointPromptedSegmenter = SAM3PointPromptedSegmenter

    def test_empty_point_coords_returns_empty_results(self):
        predictor = _FakePredictor()
        seg = self.SAM3PointPromptedSegmenter(name="sam3_point", predictor=predictor)
        image = _make_image(6, 8)

        out = seg.segment_from_points(
            image=image,
            point_coords=torch.zeros((0, 2), dtype=torch.float32),
            point_labels=torch.zeros((0,), dtype=torch.int64),
        )

        self.assertEqual(tuple(out.point_ids.shape), (0,))
        self.assertEqual(tuple(out.masks.shape), (0, 6, 8))
        self.assertEqual(tuple(out.boxes.shape), (0, 4))
        self.assertEqual(tuple(out.scores.shape), (0,))
        self.assertEqual(predictor.set_calls, 0)
        self.assertEqual(predictor.predict_calls, 0)
        self.assertEqual(predictor.reset_calls, 0)

    def test_invalid_point_coords_shape_raises(self):
        predictor = _FakePredictor()
        seg = self.SAM3PointPromptedSegmenter(name="sam3_point", predictor=predictor)
        image = _make_image(6, 8)

        with self.assertRaises(ValueError):
            seg.segment_from_points(image=image, point_coords=torch.zeros((4,), dtype=torch.float32))

    def test_invalid_point_labels_shape_raises(self):
        predictor = _FakePredictor()
        seg = self.SAM3PointPromptedSegmenter(name="sam3_point", predictor=predictor)
        image = _make_image(6, 8)

        with self.assertRaises(ValueError):
            seg.segment_from_points(
                image=image,
                point_coords=torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
                point_labels=torch.tensor([1], dtype=torch.int64),
            )

    def test_default_point_labels_are_foreground(self):
        predictor = _FakePredictor()
        seg = self.SAM3PointPromptedSegmenter(name="sam3_point", predictor=predictor)
        image = _make_image(6, 8)
        _ = seg.segment_from_points(
            image=image,
            point_coords=torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
            point_labels=None,
        )

        self.assertEqual(predictor.predict_calls, 2)
        self.assertTrue(all(int(x.item()) == 1 for labels in predictor.last_labels for x in labels))

    def test_predictor_called_once_per_point_and_resets(self):
        predictor = _FakePredictor()
        seg = self.SAM3PointPromptedSegmenter(name="sam3_point", predictor=predictor)
        image = _make_image(6, 8)
        _ = seg.segment_from_points(
            image=image,
            point_coords=torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 1.0]], dtype=torch.float32),
            point_labels=torch.tensor([1, 1, 1], dtype=torch.int64),
        )

        self.assertEqual(predictor.set_calls, 1)
        self.assertEqual(predictor.predict_calls, 3)
        self.assertEqual(predictor.reset_calls, 1)

    def test_multimask_outputs_are_flattened_with_point_ids(self):
        predictor = _FakePredictor()
        seg = self.SAM3PointPromptedSegmenter(name="sam3_point", predictor=predictor)
        image = _make_image(6, 8)
        out = seg.segment_from_points(
            image=image,
            point_coords=torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
            point_labels=torch.tensor([1, 1], dtype=torch.int64),
        )

        self.assertEqual(tuple(out.masks.shape), (4, 6, 8))
        self.assertEqual(out.point_ids.tolist(), [0, 0, 1, 1])

    def test_score_threshold_filters_low_scoring_masks(self):
        predictor = _FakePredictor()
        seg = self.SAM3PointPromptedSegmenter(
            name="sam3_point",
            predictor=predictor,
            score_thresh=0.8,
        )
        image = _make_image(6, 8)
        out = seg.segment_from_points(
            image=image,
            point_coords=torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
            point_labels=torch.tensor([1, 1], dtype=torch.int64),
        )

        self.assertEqual(tuple(out.masks.shape), (2, 6, 8))
        self.assertEqual(out.point_ids.tolist(), [0, 1])
        self.assertTrue(torch.all(out.scores >= 0.8))

    def test_boxes_are_derived_from_mask_extents(self):
        predictor = _FakePredictor()
        seg = self.SAM3PointPromptedSegmenter(name="sam3_point", predictor=predictor, score_thresh=0.8)
        image = _make_image(6, 8)
        out = seg.segment_from_points(
            image=image,
            point_coords=torch.tensor([[3.0, 4.0]], dtype=torch.float32),
            point_labels=torch.tensor([1], dtype=torch.int64),
        )

        self.assertEqual(tuple(out.boxes.shape), (1, 4))
        self.assertEqual(out.boxes[0].tolist(), [3.0, 4.0, 4.0, 5.0])


if __name__ == "__main__":
    unittest.main()

