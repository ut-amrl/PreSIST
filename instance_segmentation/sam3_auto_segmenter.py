from typing import Optional, Dict, Any

import torch
import numpy as np

from instance_segmentation.abs_instance_segmenter import (
    AbsInstanceSegmenter,
    SegmentationResults,
)

from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from instance_segmentation.raw_sam3_automatic_mask_generator import SAM3AutomaticMaskGenerator



class SAM3InstanceSegmenter(AbsInstanceSegmenter):
    """
    SAM3 wrapper that uses SAM3AutomaticMaskGenerator for segment-everything.

    Weight loading options:

    (A) Use an existing predictor:
        SAM3InstanceSegmenter(name="sam3", predictor=my_predictor)

    (B) Use a local checkpoint path:
        SAM3InstanceSegmenter(name="sam3", checkpoint_path="sam3.pt")

    (C) Use Hugging Face weights (default):
        SAM3InstanceSegmenter(name="sam3")  # downloads via HF
    """

    def __init__(
        self,
        *,
        name: str,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        predictor: Optional[SAM3InteractiveImagePredictor] = None,
        min_area_perc: float = 0.0,
        dedupe_iou_threshold: float = 1.01,
        confidence_threshold: Optional[float] = None,
        # mask_generator_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if name is None:
            raise ValueError("A name must be provided for the SAM3InstanceSegmenter.")
        super().__init__(
            name=name,
            min_area_perc=min_area_perc,
            dedupe_iou_threshold=dedupe_iou_threshold,
            confidence_threshold=confidence_threshold,
        )

        if predictor is not None and checkpoint_path is not None:
            raise ValueError(
                "Provide either `predictor` OR `checkpoint_path`, not both."
            )

        self._device = device
        # self._mask_generator_kwargs = mask_generator_kwargs or {}

        # self._mask_generator_kwargs = dict(
        #     points_per_side=16,             # ↓ total prompts 4x
        #     points_per_batch=16,            # smaller forward batches
        #     pred_iou_thresh=0.86,           # a bit stricter
        #     stability_score_thresh=0.96,    # a bit stricter
        #     stability_score_offset=1.0,
        #     mask_threshold=0.0,
        #     box_nms_thresh=0.7,
        #     crop_n_layers=0,                # start with no crops
        #     crop_overlap_ratio=512 / 1500,
        #     crop_n_points_downscale_factor=2,
        #     min_mask_region_area=0,
        #     output_mode="binary_mask",
        #     use_m2m=False,
        #     multimask_output=False,         # *** huge memory win ***
        # )

        self._mask_generator_kwargs = dict(
            points_per_side=16,
            points_per_batch=16,
            pred_iou_thresh=0.0,          # <─ disable IoU filtering
            stability_score_thresh=0.0,   # <─ disable stability filtering
            stability_score_offset=1.0,
            mask_threshold=0.0,
            box_nms_thresh=0.7,
            crop_n_layers=0,
            crop_overlap_ratio=512 / 1500,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=0,
            output_mode="binary_mask",
            use_m2m=False,
            multimask_output=False,
        )

        if predictor is not None:
            # (A) You built the predictor yourself
            self._mask_generator = SAM3AutomaticMaskGenerator(
                predictor=predictor,
                **self._mask_generator_kwargs,
            )
        else:
            # (B) or (C) – let SAM3AutomaticMaskGenerator.from_pretrained handle it.
            # If checkpoint_path is None, it will internally call download_ckpt_from_hf()
            # and fetch the weights from Hugging Face.
            self._mask_generator = SAM3AutomaticMaskGenerator.from_pretrained(
                checkpoint_path=checkpoint_path,
                device=device,
                **self._mask_generator_kwargs,
            )

    @staticmethod
    def _xywh_to_xyxy(xywh):
        x, y, w, h = map(float, xywh)
        return [x, y, x + w, y + h]

    def segmentInstancesRaw(self, image: torch.Tensor) -> SegmentationResults:
        # (C, H, W) -> (H, W, C) numpy uint8
        img_np = image[:3].permute(1, 2, 0).contiguous().cpu().numpy()
        H, W = img_np.shape[:2]

        raw = self._mask_generator.generate(img_np)

        if len(raw) == 0:
            return SegmentationResults.empty(h=H, w=W)

        ids = torch.arange(len(raw), dtype=torch.int64)

        masks = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(r["segmentation"], dtype=bool)
                )
                for r in raw
            ],
            dim=0,
        )  # (N, H, W) bool

        boxes = torch.tensor(
            [self._xywh_to_xyxy(r.get("bbox", [0.0, 0.0, 0.0, 0.0])) for r in raw],
            dtype=torch.float32,
        )

        scores = torch.tensor(
            [float(r.get("predicted_iou", 0.0)) for r in raw],
            dtype=torch.float32,
        )

        return SegmentationResults(
            ids=ids,
            masks=masks,
            boxes=boxes,
            scores=scores,
        )
