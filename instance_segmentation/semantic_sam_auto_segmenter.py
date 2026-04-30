from __future__ import annotations

import os
from typing import Optional, Sequence, Union, List, Tuple
import numpy as np
import torch
from PIL import Image

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from instance_segmentation.abs_instance_segmenter import AbsInstanceSegmenter, SegmentationResults
from dataset_generation.intermediate_results_logger.debug_data import DebugPayload
from semantic_sam import build_semantic_sam, SemanticSamAutomaticMaskGenerator  # type: ignore
import semantic_sam

from utils.threading_utils import ForwardSerializeLock



# Can be 1-6, float in [0..1], string key, or list of levels
# 1 is least detailed, 6 is most detailed
# if key in ("all", "max", "fine+"):
#     return [1, 2, 3, 4, 5, 6]
# if key in ("fine", "detailed", "parts"):
#     return [2, 3, 4, 5, 6]
# if key in ("medium", "default"):
#     return [2, 3, 4]
# if key in ("coarse", "semantic", "big"):


GranularitySpec = Union[int, float, str, Sequence[int]]


import yaml
import json
import argparse
import logging

logger = logging.getLogger(__name__)


def load_config_dict_to_opt(opt, config_dict):
    """
    Load the key, value pairs from config_dict to opt, overriding existing values in opt
    if there is any.
    """
    if not isinstance(config_dict, dict):
        raise TypeError("Config must be a Python dictionary")
    for k, v in config_dict.items():
        k_parts = k.split('.')
        pointer = opt
        for k_part in k_parts[:-1]:
            if k_part not in pointer:
                pointer[k_part] = {}
            pointer = pointer[k_part]
            assert isinstance(pointer, dict), "Overriding key needs to be inside a Python dict."
        ori_value = pointer.get(k_parts[-1])
        pointer[k_parts[-1]] = v
        if ori_value:
            logger.warning(f"Overrided {k} from {ori_value} to {pointer[k_parts[-1]]}")

def load_opt_from_config_file(conf_file):
    """
    Load opt from the config files, settings in later files can override those in previous files.

    Args:
        conf_files: config file path

    Returns:
        dict: a dictionary of opt settings
    """
    opt = {}
    with open(conf_file, encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    load_config_dict_to_opt(opt, config_dict)

    return opt


def _levels_from_granularity(g: GranularitySpec) -> List[int]:
    if isinstance(g, int):
        if not (1 <= g <= 6):
            raise ValueError("If granularity is an int, it must be in [1..6].")
        return [g]
    if isinstance(g, (list, tuple)):
        lv = [int(x) for x in g]
        if any(x < 1 or x > 6 for x in lv):
            raise ValueError("All levels must be in [1..6].")
        out: List[int] = []
        for x in lv:
            if x not in out:
                out.append(x)
        return out
    if isinstance(g, float):
        if not (0.0 <= g <= 1.0):
            raise ValueError("If granularity is a float, it must be in [0..1].")
        if g < 1 / 3:
            return [2]
        if g < 2 / 3:
            return [2, 3, 4]
        return [1, 2, 3, 4, 5, 6]
    if isinstance(g, str):
        key = g.strip().lower()
        if key in ("all", "max", "fine+"):
            return [1, 2, 3, 4, 5, 6]
        if key in ("fine", "detailed", "parts"):
            return [2, 3, 4, 5, 6]
        if key in ("medium", "default"):
            return [2, 3, 4]
        if key in ("coarse", "semantic", "big"):
            return [2]
        raise ValueError(f"Unknown granularity string: {g!r}")
    raise TypeError(f"Unsupported granularity type: {type(g)}")


def _mask_to_xyxy(mask: np.ndarray) -> List[float]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max() + 1)
    y1 = float(ys.max() + 1)
    return [x0, y0, x1, y1]


def _xywh_to_xyxy(xywh: Sequence[float]) -> List[float]:
    x, y, w, h = map(float, xywh)
    return [x, y, x + w, y + h]


def _scale_xyxy_box(box_xyxy: Sequence[float], sx: float, sy: float) -> List[float]:
    x0, y0, x1, y1 = map(float, box_xyxy)
    return [x0 * sx, y0 * sy, x1 * sx, y1 * sy]


def customBuildSemanticSam(model_type: str, checkpoint: str, device: Optional[str] = None):
    """
    Lightweight replacement for sam2.build_sam.build_sam2 that does NOT use Hydra's
    config search path (so it plays nicely inside your own Hydra app).

    Args:
        model_type: " T" or "L"
        checkpoint:
            Path to the SemanticSAM checkpoint (.pt).
        device:
            e.g. "cuda", "cuda:0", "cpu", or None to leave on CPU.

    Returns:
        A torch.nn.Module SemanticSAM model, moved to the requested device and set to eval().
    """

    # ----------------------------
    # 1. Resolve the config path
    # ----------------------------
    cfgs={'T':"configs/semantic_sam_only_sa-1b_swinT.yaml",
          'L':"configs/semantic_sam_only_sa-1b_swinL.yaml"}

    sam_cfg=cfgs[model_type]

    # If it's not absolute and doesn't exist where we are, try relative to the semantic_sam package dir
    if not os.path.isabs(sam_cfg) and not os.path.exists(sam_cfg):
        semantic_sam_dir = os.path.dirname(os.path.abspath(semantic_sam.__file__))
        # Go up another level to the package root
        semantic_sam_dir = os.path.dirname(semantic_sam_dir)
        candidate = os.path.join(semantic_sam_dir, sam_cfg)
        if os.path.exists(candidate):
            cfg_path = candidate
        else:
            raise FileNotFoundError(
                f"Could not find SemanticSAM config file '{sam_cfg}'. "
                f"Tried '{sam_cfg}' and '{candidate}'."
            )

    opt = load_opt_from_config_file(cfg_path)
    builtModel = semantic_sam.build_model(opt)
    model_semantic_sam = semantic_sam.BaseModel.BaseModel(opt, builtModel).from_pretrained(checkpoint).eval().cuda()
    return model_semantic_sam


def _prepare_image_from_tensor_like_repo(
    image_chw_u8: torch.Tensor,
    device: torch.device,
    resize_short_edge: int = 640,
):
    """
    In-memory mimic of the repo's `prepare_image(image_pth)`:

        image = Image.open(...).convert('RGB')
        Resize(640, BICUBIC)  # short edge -> 640, keep aspect ratio
        image_ori = np.asarray(image_ori)
        images = torch.from_numpy(...).permute(2,0,1).cuda()

    Returns:
        resized_np: np.ndarray (Hr, Wr, 3) uint8
        resized_chw_u8: torch.Tensor (3, Hr, Wr) uint8 on `device`
    """
    if image_chw_u8.ndim != 3 or image_chw_u8.shape[0] not in (3, 4):
        raise ValueError("Expected input image tensor shape (C,H,W) with C in {3,4}.")
    if image_chw_u8.dtype != torch.uint8:
        raise ValueError("Expected image dtype=torch.uint8.")

    # CHW -> HWC uint8 RGB
    img_np = image_chw_u8[:3].permute(1, 2, 0).contiguous().cpu().numpy()
    pil = Image.fromarray(img_np).convert("RGB")

    # Resize: int size => match shorter edge to size, keep aspect (torchvision semantics)
    pil_resized = TF.resize(
        pil,
        size=resize_short_edge,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )

    resized_np = np.asarray(pil_resized)  # (Hr, Wr, 3) uint8
    resized_chw_u8 = torch.from_numpy(resized_np.copy()).permute(2, 0, 1).to(device=device)
    return resized_np, resized_chw_u8


def _resize_masks_to_original(
    masks_bool: torch.Tensor,  # (N, Hr, Wr) bool
    orig_hw: Sequence[int],    # (H0, W0)
) -> torch.Tensor:
    """
    Upsample boolean masks from resized (Hr,Wr) back to original (H0,W0).
    Use nearest interpolation to preserve mask discreteness.
    """
    if masks_bool.ndim != 3:
        raise ValueError("Expected masks_bool with shape (N, Hr, Wr).")
    H0, W0 = int(orig_hw[0]), int(orig_hw[1])

    # bool -> float -> interpolate -> bool
    m = masks_bool.unsqueeze(1).to(dtype=torch.float32)  # (N,1,Hr,Wr)
    m_up = torch.nn.functional.interpolate(m, size=(H0, W0), mode="nearest")
    return (m_up[:, 0] > 0.5).to(dtype=torch.bool)       # (N,H0,W0)

class Test(AbsInstanceSegmenter):
    def __init__(
        self,
        *,
        min_area_perc: float = 0.0,
        dedupe_iou_threshold: float = 1.01,
        confidence_threshold: Optional[float] = None,
    ):
        super().__init__(
            name="TestSegmenter",
            min_area_perc=min_area_perc,
            dedupe_iou_threshold=dedupe_iou_threshold,
            confidence_threshold=confidence_threshold,
        )


class SemanticSAMInstanceSegmenter(AbsInstanceSegmenter):
    """
    Semantic-SAM-backed segmenter that:
      1) preprocesses like the repo (short-edge resize to 640),
      2) runs auto mask generation,
      3) maps masks + boxes back to the *original* input resolution.
    """

    def __init__(
        self,
        *,
        checkpoint: str,
        model_type: str,  # typically "T" or "L"
        device: Optional[str] = None,
        granularity: GranularitySpec = "all",
        resize_short_edge: int = 640,
        min_area_perc: float = 0.0,
        dedupe_iou_threshold: float = 1.01,
        confidence_threshold: Optional[float] = None,
        name: Optional[str] = None,
        generator_kwargs: Optional[dict] = None,
    ):
        if name is None:
            raise ValueError("A name must be provided for the SemanticSAMInstanceSegmenter.")
        super().__init__(
            name=name,
            min_area_perc=min_area_perc,
            dedupe_iou_threshold=dedupe_iou_threshold,
            confidence_threshold=confidence_threshold,
        )

        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._levels = _levels_from_granularity(granularity)
        self._resize_short_edge = int(resize_short_edge)
        self._generator_kwargs = generator_kwargs or {}

        # self._model = build_semantic_sam(model_type=model_type, ckpt=checkpoint)
        self._model = customBuildSemanticSam(model_type=model_type, checkpoint=checkpoint, device=self._device)
        if hasattr(self._model, "to"):
            self._model = self._model.to(self._device)
        if hasattr(self._model, "eval"):
            self._model.eval()

        self._mask_generator = SemanticSamAutomaticMaskGenerator(
            self._model,
            level=self._levels,
            **self._generator_kwargs,
        )
        # Defense-in-depth: PyTorch model + non-atomic mutable state
        # (setGranularity rebuilds the mask generator). Single-threaded today;
        # the lock keeps it correct under future cross-scene threading.
        self._forward_lock = ForwardSerializeLock(
            label=f"{type(self).__name__}.segmentInstancesRaw",
            advice="segmentation cannot be parallelized across host threads on "
                   "one model; run from a single-threaded driver",
        )

    def setGranularity(self, granularity: GranularitySpec) -> None:
        self._levels = _levels_from_granularity(granularity)
        self._mask_generator = SemanticSamAutomaticMaskGenerator(
            self._model,
            level=self._levels,
            **self._generator_kwargs,
        )

    def segmentInstancesRaw(self, image: torch.Tensor) -> Tuple[SegmentationResults, DebugPayload]:
        H0, W0 = int(image.shape[1]), int(image.shape[2])

        # 1) Preprocess like repo (resized space)
        resized_np, input_image = _prepare_image_from_tensor_like_repo(
            image, self._device, resize_short_edge=self._resize_short_edge
        )
        Hr, Wr = int(resized_np.shape[0]), int(resized_np.shape[1])

        # 2) Run generator in resized space
        with self._forward_lock.acquire():
            outputs = self._mask_generator.generate(input_image)
        if outputs is None or len(outputs) == 0:
            print("No masks generated by SemanticSAM.")
            return SegmentationResults.empty(h=H0, w=W0), DebugPayload.empty()

        print("Generated", len(outputs), "masks from SemanticSAM.")

        # Scale factors resized -> original
        sx = W0 / float(Wr)  # x uses width
        sy = H0 / float(Hr)  # y uses height

        masks_resized: List[torch.Tensor] = []
        boxes_orig: List[List[float]] = []
        scores_list: List[float] = []

        for o in outputs:
            if not isinstance(o, dict) or "segmentation" not in o:
                continue


            seg_np = np.asarray(o["segmentation"])

            if seg_np.dtype != np.bool_:
                seg_np = seg_np.astype(np.bool_)

            if seg_np.shape != (Hr, Wr):
                raise ValueError(f"Mask shape {seg_np.shape} != expected {(Hr, Wr)} in resized space.")

            seg_area = seg_np.sum()
            if seg_area == 0:
                continue

            masks_resized.append(torch.from_numpy(seg_np))

            # Box in resized coordinates (prefer provided bbox; else compute from mask)
            if "bbox" in o and o["bbox"] is not None:
                box_xyxy_r = _xywh_to_xyxy(o["bbox"])
            else:
                box_xyxy_r = _mask_to_xyxy(seg_np)

            # Map box back to original coordinates
            boxes_orig.append(_scale_xyxy_box(box_xyxy_r, sx=sx, sy=sy))

            s = o.get("predicted_iou", None) or o.get("iou", None) or o.get("score", None)
            scores_list.append(float(s) if s is not None else 0.0)

        if len(masks_resized) == 0:
            return SegmentationResults.empty(h=H0, w=W0), DebugPayload.empty()

        # 3) Map masks back to original resolution
        masks_resized_t = torch.stack(masks_resized, dim=0).to(dtype=torch.bool)   # (N,Hr,Wr)
        masks_orig_t = _resize_masks_to_original(masks_resized_t, (H0, W0))        # (N,H0,W0)

        boxes_t = torch.tensor(boxes_orig, dtype=torch.float32)                    # (N,4) in original coords
        scores_t = torch.tensor(scores_list, dtype=torch.float32)
        ids_t = torch.arange(masks_orig_t.shape[0], dtype=torch.int64)

        return SegmentationResults(ids=ids_t, masks=masks_orig_t, boxes=boxes_t, scores=scores_t), DebugPayload.empty()
