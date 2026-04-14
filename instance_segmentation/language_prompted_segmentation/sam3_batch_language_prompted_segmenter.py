"""
sam3_language_prompted_segmenter_batched.py

Alternate implementation of AbsLanguagePromptedSegmenter for SAM3 that supports
TRUE multi-prompt batching (one forward pass for many text prompts) using SAM3's
Datapoint + collate + PostProcessImage pipeline (as in sam3_image_batched_inference.ipynb).

This file is self-contained except for:
- your project's AbsLanguagePromptedSegmenter + LanguagePromptedSegmentationResults
- SAM3 library imports via `from sam3.... import ...`
- standard deps: torch, numpy, PIL

Notes:
- This implementation uses per-instance query ids (no global mutable counter).
- It chunks prompts (prompts_per_forward) to control VRAM.
- It attempts to map each detection back to the originating prompt using keys
  exposed by the SAM3 postprocessor output. If it cannot, it assigns prompt_id=-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
from PIL import Image

from tqdm import tqdm

from instance_segmentation.language_prompted_segmentation.abs_language_prompted_segmenter import (
    AbsLanguagePromptedSegmenter,
    LanguagePromptedSegmentationResults,
)

# --- SAM3 imports (all within sam3.* namespace) ---
from sam3 import build_sam3_image_model

from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device

from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint,
)

from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)

from sam3.eval.postprocessors import PostProcessImage


def _mask_to_box_xyxy(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute tight XYXY box around a boolean mask. Returns float32 [x0,y0,x1,y1].
    """
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return torch.zeros((4,), dtype=torch.float32)
    x0 = xs.min().item()
    y0 = ys.min().item()
    x1 = xs.max().item() + 1
    y1 = ys.max().item() + 1
    return torch.tensor([x0, y0, x1, y1], dtype=torch.float32)


def _to_cpu_bool_masks(masks: Any, *, H: int, W: int) -> torch.Tensor:
    """
    Normalize masks to torch.bool on CPU with shape (K,H,W).
    """
    if masks is None:
        return torch.zeros((0, H, W), dtype=torch.bool)

    if isinstance(masks, torch.Tensor):
        m = masks.detach().to("cpu")
    else:
        m = torch.as_tensor(masks, device="cpu")

    # Common shapes: (K,H,W) already, or something reshapeable.
    if m.ndim == 2:
        m = m.unsqueeze(0)
    if m.ndim != 3:
        m = m.reshape(-1, H, W)

    # ensure correct spatial size
    if m.shape[-2:] != (H, W):
        m = m.reshape(m.shape[0], H, W)

    return m.to(torch.bool)


def _to_cpu_f32(x: Any) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").to(torch.float32)
    return torch.as_tensor(x, device="cpu", dtype=torch.float32)


def _to_cpu_i64_list(x: Any) -> Optional[List[int]]:
    """
    Convert a tensor/array/list to a python List[int] on CPU.
    """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").to(torch.int64).tolist()
    return torch.as_tensor(x, device="cpu", dtype=torch.int64).tolist()


class SAM3BatchedLanguagePromptedSegmenter(AbsLanguagePromptedSegmenter):
    """
    SAM3 language-prompted segmenter using the SAM3 "batched inference" API:
    - Create one Datapoint per image
    - Add N text prompts (FindQueryLoaded) to the same datapoint
    - Transform + collate once
    - Forward once
    - Postprocess once

    Returns 0..K detections per prompt (flattened) and provides prompt_ids mapping
    detections to the originating prompt (by index in the input `prompts` list).

    Key params:
    - prompts_per_forward: prompt chunk size to control VRAM
    - score_thresh: optional additional threshold at this wrapper level
    """

    def __init__(
        self,
        *,
        name: str,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        # batching
        prompts_per_forward: int = 32,
        # thresholds
        # score_thresh: float = -1.0,
        score_thresh: float = 0.5,
        # behavior if we can't map detections back to prompts
        # transform config (match notebook defaults)
        resize_size: int = 1008,
        resize_max_size: int = 1008,
    ):
        super().__init__(name=name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.prompts_per_forward = int(prompts_per_forward)
        if self.prompts_per_forward <= 0:
            raise ValueError("prompts_per_forward must be positive")

        self.score_thresh = float(score_thresh)

        # Per-instance query id counter (avoid global mutable state)
        self._query_uid = 1

        # Build model
        build_kwargs: Dict[str, Any] = {}
        if checkpoint_path is not None:
            build_kwargs["checkpoint_path"] = checkpoint_path

        try:
            self.model = build_sam3_image_model(**build_kwargs)
        except TypeError:
            # Fallback if the installed SAM3 signature differs.
            # This mirrors the defensive pattern in your original file.
            self.model = build_sam3_image_model()

        self.model = self.model.to(self.device)
        self.model.eval()

        # Transform pipeline (as in notebook)
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=resize_size,
                    max_size=resize_max_size,
                    square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Postprocessor (as in notebook)
        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=self.score_thresh,
            to_cpu=False,
        )

    # -------------------------
    # Datapoint helper methods
    # -------------------------

    def _create_empty_datapoint(self) -> Datapoint:
        """
        A datapoint is a single image on which we can apply several queries at once.
        """
        return Datapoint(find_queries=[], images=[])

    def _set_image(self, datapoint: Datapoint, pil_image: Image.Image) -> None:
        """
        Add the image to be processed to the datapoint.
        """
        w, h = pil_image.size
        datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]

    # def _add_text_prompt(self, datapoint: Datapoint, text_query: str) -> int:
    #     """
    #     Add a text query to the datapoint. Returns a unique query uid.
    #     """
    #     assert len(datapoint.images) == 1, "please set the image first"
    #     w, h = datapoint.images[0].size

    def _add_text_prompt(self, datapoint: Datapoint, text_query: str) -> int:
        assert len(datapoint.images) == 1, "please set the image first"
        w, h = datapoint.images[0].size

        uid = int(self._query_uid)
        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[],  # unused for inference
                is_exhaustive=True,  # unused for inference
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=uid,
                    original_image_id=uid,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                ),
            )
        )
        self._query_uid += 1
        return uid

    # -------------------------
    # Output mapping utilities
    # -------------------------

    @staticmethod
    def _extract_detection_query_ids(img_res: Dict[str, Any]) -> Optional[List[int]]:
        """
        Try to extract a per-detection query id list from postprocessor output.

        SAM3 versions can differ; we try common keys.
        """
        # Most desirable: explicit query id per detection
        for key in ("query_ids", "query_id", "find_query_ids", "find_query_id"):
            if key in img_res:
                return _to_cpu_i64_list(img_res.get(key))

        # Sometimes metadata ids are surfaced; depending on implementation this may be
        # aligned with detections or may be per-query; we still attempt it.
        for key in ("coco_image_id", "original_image_id", "coco_ids", "image_ids"):
            if key in img_res:
                return _to_cpu_i64_list(img_res.get(key))

        return None

    # -------------------------
    # Main API implementation
    # -------------------------

    @torch.no_grad()
    def segment_from_prompts(
        self,
        image: torch.Tensor,
        prompts: List[str],
    ) -> LanguagePromptedSegmentationResults:
        """
        True multi-prompt batching via SAM3 datapoint API.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        if image.ndim != 3 or image.shape[0] not in (3, 4):
            raise ValueError("Expected image (C,H,W) with C in {3,4}.")
        if image.dtype != torch.uint8:
            raise ValueError("Expected image dtype=torch.uint8.")

        C, H, W = image.shape

        if len(prompts) == 0:
            return LanguagePromptedSegmentationResults(
                prompt_ids=torch.zeros((0,), dtype=torch.int64),
                masks=torch.zeros((0, H, W), dtype=torch.bool),
                boxes=torch.zeros((0, 4), dtype=torch.float32),
                scores=torch.zeros((0,), dtype=torch.float32),
            )

        # Convert to PIL once
        img_np = image[:3].permute(1, 2, 0).contiguous().cpu().numpy()
        pil = Image.fromarray(img_np, mode="RGB")

        all_prompt_ids: List[int] = []
        all_masks: List[torch.Tensor] = []
        all_boxes: List[torch.Tensor] = []
        all_scores: List[float] = []

        device = torch.device(self.device)

        # Chunk prompts to control VRAM
        for chunk_start in tqdm(range(0, len(prompts), self.prompts_per_forward), leave=False, desc="Processing prompt chunks"):
            chunk = prompts[chunk_start : chunk_start + self.prompts_per_forward]

            dp = self._create_empty_datapoint()
            self._set_image(dp, pil)

            # Map query_uid -> input prompt index
            uid_to_pid: Dict[int, int] = {}
            for i, prompt in enumerate(chunk):
                uid = self._add_text_prompt(dp, prompt)
                uid_to_pid[int(uid)] = chunk_start + i

            # Transform once
            dp = self.transform(dp)

            raw_batch = collate([dp], dict_key="dummy")
            # print("collate returned:", type(raw_batch))
            # if isinstance(raw_batch, dict):
                # print("dict keys:", raw_batch.keys())

            batch = raw_batch["dummy"]
            # print("batch type before device copy:", type(batch))

            batch = copy_data_to_device(batch, device, non_blocking=True)
            # print("batch type after device copy:", type(batch))

            # outputs = self.model(batch)

            outputs = self.model(batch)

            # SAM3 expects a LIST of BatchedInferenceMetadata objects here
            find_metadatas = batch.find_metadatas
            if not isinstance(find_metadatas, (list, tuple)):
                find_metadatas = [find_metadatas]

            processed = self.postprocessor.process_results(outputs, find_metadatas)

            if not isinstance(processed, dict):
                raise RuntimeError(f"Unexpected process_results output type: {type(processed)}")

            # processed is keyed by query/image id (the uid you stored in coco_image_id)
            for qid, pid in uid_to_pid.items():
                qres = processed.get(int(qid), None)
                if qres is None:
                    continue

                masks_raw = qres.get("masks", None)
                if masks_raw is None:
                    continue

                # SAM3 commonly returns [K,1,H,W]; squeeze channel if present
                if isinstance(masks_raw, torch.Tensor):
                    mt = masks_raw.detach().to("cpu")
                else:
                    mt = torch.as_tensor(masks_raw, device="cpu")

                if mt.ndim == 4 and mt.shape[1] == 1:
                    mt = mt.squeeze(1)
                elif mt.ndim == 2:
                    mt = mt.unsqueeze(0)
                elif mt.ndim != 3:
                    raise RuntimeError(f"Unexpected mask shape for qid={qid}: {tuple(mt.shape)}")

                masks_t = mt.to(torch.bool)
                K = int(masks_t.shape[0])
                if K == 0:
                    continue

                scores_t = _to_cpu_f32(qres.get("scores", None))
                boxes_t = _to_cpu_f32(qres.get("boxes", None))

                for k in range(K):
                    m = masks_t[k]

                    # Defensive resize/reshape handling
                    if m.shape == (W, H):
                        m = m.T
                    if m.shape != (H, W):
                        raise RuntimeError(
                            f"Mask shape mismatch for qid={qid}, det={k}: got {tuple(m.shape)}, expected {(H, W)}"
                        )

                    s = float(scores_t[k].item()) if scores_t is not None and scores_t.numel() > k else 0.0
                    if self.score_thresh >= 0 and scores_t is not None and s < self.score_thresh:
                        continue

                    if boxes_t is not None and boxes_t.ndim == 2 and boxes_t.shape[0] > k and boxes_t.shape[1] == 4:
                        b = boxes_t[k]
                    else:
                        b = _mask_to_box_xyxy(m)

                    all_prompt_ids.append(pid)
                    all_masks.append(m)
                    all_boxes.append(b.to(torch.float32))
                    all_scores.append(s)

            # outputs = self.model(batch)

            # print("model outputs type:", type(outputs))

            # if isinstance(outputs, dict):
            #     print("model output keys:", outputs.keys())
            # elif isinstance(outputs, (list, tuple)):
            #     print("model output len:", len(outputs))
            #     for i, x in enumerate(outputs[:5]):
            #         print(f"  outputs[{i}] type:", type(x))
            #         if isinstance(x, dict):
            #             print(f"  outputs[{i}] keys:", x.keys())

            # # target_sizes = torch.tensor([[H, W]], device=device, dtype=torch.int64)
            # # processed = self.postprocessor(
            # #     outputs,
            # #     target_sizes_boxes=target_sizes,
            # #     target_sizes_masks=target_sizes,
            # #     consistent=True,
            # # )

            # # Forward

            # # Postprocess using SAM3's metadata-aware helper
            # processed = self.postprocessor.process_results(outputs, batch.find_metadatas)

            # # Typically list[dict] per image
            # if isinstance(processed, (list, tuple)):
            #     if len(processed) == 0:
            #         continue
            #     img_res = processed[0]
            # elif isinstance(processed, dict):
            #     img_res = processed
            # else:
            #     raise RuntimeError(f"Unexpected postprocessor output type: {type(processed)}")

            # # Extract detections
            # masks_raw = img_res.get("masks", None)
            # if masks_raw is None:
            #     # No masks returned
            #     continue

            # masks_t = _to_cpu_bool_masks(masks_raw, H=H, W=W)
            # if masks_t.numel() == 0:
            #     continue

            # boxes_t = _to_cpu_f32(img_res.get("boxes", None))
            # scores_t = _to_cpu_f32(img_res.get("scores", None))

            # K = int(masks_t.shape[0])

            # det_query_ids = self._extract_detection_query_ids(img_res)

            # # Validate query id alignment if present
            # if det_query_ids is not None and len(det_query_ids) != K:
            #     # Some builds store ids per query rather than per detection.
            #     # Without correct alignment, we cannot map reliably.
            #     det_query_ids = None

            # for k in range(K):
            #     m = masks_t[k]
            #     if m.shape != (H, W):
            #         m = m.reshape(H, W)

            #     s = float(scores_t[k].item()) if scores_t is not None and scores_t.numel() > k else 0.0
            #     if self.score_thresh >= 0 and scores_t is not None and s < self.score_thresh:
            #         continue

            #     if boxes_t is not None and boxes_t.ndim == 2 and boxes_t.shape[1] == 4 and boxes_t.shape[0] > k:
            #         b = boxes_t[k]
            #     else:
            #         b = _mask_to_box_xyxy(m)

            #     if det_query_ids is not None:
            #         qid = int(det_query_ids[k])
            #         pid = uid_to_pid.get(qid, None)
            #         if pid is None:
            #             # Unknown query id; skip to avoid corrupt mapping
            #             continue
            #     else:
            #         pid = -1  # unattributed

            #     all_prompt_ids.append(pid)
            #     all_masks.append(m)
            #     all_boxes.append(b.to(torch.float32))
            #     all_scores.append(s)

        if len(all_masks) == 0:
            return LanguagePromptedSegmentationResults(
                prompt_ids=torch.zeros((0,), dtype=torch.int64),
                masks=torch.zeros((0, H, W), dtype=torch.bool),
                boxes=torch.zeros((0, 4), dtype=torch.float32),
                scores=torch.zeros((0,), dtype=torch.float32),
            )

        masks_out = torch.stack(all_masks, dim=0)
        boxes_out = torch.stack(all_boxes, dim=0).to(torch.float32)
        scores_out = torch.tensor(all_scores, dtype=torch.float32)
        prompt_ids_out = torch.tensor(all_prompt_ids, dtype=torch.int64)

        return LanguagePromptedSegmentationResults(
            prompt_ids=prompt_ids_out,
            masks=masks_out,
            boxes=boxes_out,
            scores=scores_out,
        )