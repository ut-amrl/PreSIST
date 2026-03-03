from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from instance_segmentation.point_prompted_segmentation.abs_point_prompted_segmenter import (
    AbsPointPromptedSegmenter,
    PointPromptedSegmentationResults,
)
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor


def _mask_to_box_xyxy(mask: torch.Tensor) -> torch.Tensor:
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return torch.zeros((4,), dtype=torch.float32)
    x0 = xs.min().item()
    y0 = ys.min().item()
    x1 = xs.max().item() + 1
    y1 = ys.max().item() + 1
    return torch.tensor([x0, y0, x1, y1], dtype=torch.float32)


def _build_predictor_from_pretrained(
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
) -> SAM3InteractiveImagePredictor:
    from sam3.model_builder import (
        _create_vision_backbone,
        build_tracker,
        download_ckpt_from_hf,
    )
    from sam3.model.vl_combiner import SAM3VLBackbone

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    vision_backbone = _create_vision_backbone(enable_inst_interactivity=True)
    backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)

    tracker = build_tracker(
        apply_temporal_disambiguation=False,
        with_backbone=False,
    )
    tracker.backbone = backbone

    if checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]

    tracker_ckpt = {
        k.replace("tracker.", ""): v for k, v in ckpt.items() if "tracker" in k
    }
    tracker.load_state_dict(tracker_ckpt, strict=False)

    backbone_ckpt = {}
    for k, v in ckpt.items():
        if "detector.backbone.vision_backbone" in k or "detector.backbone.visual" in k:
            new_key = k.replace("detector.backbone.", "")
            backbone_ckpt[new_key] = v
    if backbone_ckpt:
        backbone.load_state_dict(backbone_ckpt, strict=False)

    tracker.to(device)
    tracker.eval()

    return SAM3InteractiveImagePredictor(tracker)


class SAM3PointPromptedSegmenter(AbsPointPromptedSegmenter):
    """
    SAM3 point-prompted segmenter that returns flattened masks across points.
    """

    def __init__(
        self,
        *,
        name: str,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        predictor: Optional[SAM3InteractiveImagePredictor] = None,
        score_thresh: float = -1.0,
        multimask_output: bool = True,
    ):
        super().__init__(name=name)
        self.score_thresh = float(score_thresh)
        self.multimask_output = bool(multimask_output)

        if predictor is not None and checkpoint_path is not None:
            raise ValueError("Provide either `predictor` or `checkpoint_path`, not both.")

        if predictor is not None:
            self.predictor = predictor
        else:
            self.predictor = _build_predictor_from_pretrained(
                checkpoint_path=checkpoint_path,
                device=device,
            )

    @staticmethod
    def _empty_results(h: int, w: int) -> PointPromptedSegmentationResults:
        return PointPromptedSegmentationResults(
            point_ids=torch.zeros((0,), dtype=torch.int64),
            masks=torch.zeros((0, h, w), dtype=torch.bool),
            boxes=torch.zeros((0, 4), dtype=torch.float32),
            scores=torch.zeros((0,), dtype=torch.float32),
        )

    @torch.no_grad()
    def segment_from_points(
        self,
        image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: Optional[torch.Tensor] = None,
    ) -> PointPromptedSegmentationResults:
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        if image.ndim != 3 or image.shape[0] not in (3, 4):
            raise ValueError("Expected image (C,H,W) with C in {3,4}.")
        if image.dtype != torch.uint8:
            raise ValueError("Expected image dtype=torch.uint8.")

        _, h, w = image.shape

        if not isinstance(point_coords, torch.Tensor):
            raise TypeError("point_coords must be a torch.Tensor of shape (P,2).")
        if point_coords.ndim != 2 or point_coords.shape[1] != 2:
            raise ValueError("Expected point_coords shape (P,2).")
        if point_coords.shape[0] == 0:
            return self._empty_results(h=h, w=w)

        if point_labels is None:
            point_labels = torch.ones((point_coords.shape[0],), dtype=torch.int64)
        if point_labels.ndim != 1 or point_labels.shape[0] != point_coords.shape[0]:
            raise ValueError("Expected point_labels shape (P,).")

        img_np = image[:3].permute(1, 2, 0).contiguous().cpu().numpy()
        self.predictor.set_image(img_np)

        all_point_ids: List[int] = []
        all_masks: List[torch.Tensor] = []
        all_boxes: List[torch.Tensor] = []
        all_scores: List[float] = []

        coords_cpu = point_coords.detach().to("cpu", dtype=torch.float32)
        labels_cpu = point_labels.detach().to("cpu", dtype=torch.int64)

        for pid in range(coords_cpu.shape[0]):
            xy = coords_cpu[pid].numpy().reshape(1, 2)
            plabel = np.asarray([int(labels_cpu[pid].item())], dtype=np.int64)

            masks_np, iou_np, _ = self.predictor.predict(
                point_coords=xy,
                point_labels=plabel,
                multimask_output=self.multimask_output,
                return_logits=False,
                normalize_coords=False,
            )

            if masks_np is None or len(masks_np) == 0:
                continue

            masks_t = torch.as_tensor(masks_np, dtype=torch.bool, device="cpu")
            iou_t = torch.as_tensor(iou_np, dtype=torch.float32, device="cpu")

            for i in range(masks_t.shape[0]):
                score = float(iou_t[i].item()) if i < iou_t.shape[0] else 0.0
                if self.score_thresh >= 0 and score < self.score_thresh:
                    continue

                mask = masks_t[i]
                if mask.shape != (h, w):
                    mask = mask.reshape(h, w)
                all_point_ids.append(pid)
                all_masks.append(mask)
                all_boxes.append(_mask_to_box_xyxy(mask))
                all_scores.append(score)

        self.predictor.reset_predictor()

        if len(all_masks) == 0:
            return self._empty_results(h=h, w=w)

        return PointPromptedSegmentationResults(
            point_ids=torch.tensor(all_point_ids, dtype=torch.int64),
            masks=torch.stack(all_masks, dim=0).to(torch.bool),
            boxes=torch.stack(all_boxes, dim=0).to(torch.float32),
            scores=torch.tensor(all_scores, dtype=torch.float32),
        )

