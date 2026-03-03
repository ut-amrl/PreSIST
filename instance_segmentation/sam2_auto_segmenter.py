from instance_segmentation.abs_instance_segmenter import AbsInstanceSegmenter, SegmentationResults
from typing import Optional
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import sam2
import os
from omegaconf import OmegaConf
from hydra.utils import instantiate

# =======================================
# SAM 2-backed instance segmenter (auto)
# =======================================


def customBuildSam2(model_cfg: str, checkpoint: str, device: Optional[str] = None):
    """
    Lightweight replacement for sam2.build_sam.build_sam2 that does NOT use Hydra's
    config search path (so it plays nicely inside your own Hydra app).

    Args:
        model_cfg:
            - Either an absolute path to the YAML config, e.g.
              "/workspace/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
            - Or a path *relative to the installed `sam2` package*, e.g.
              "configs/sam2.1/sam2.1_hiera_b+.yaml"
        checkpoint:
            Path to the SAM2 checkpoint (.pt).
        device:
            e.g. "cuda", "cuda:0", "cpu", or None to leave on CPU.

    Returns:
        A torch.nn.Module SAM2 model, moved to the requested device and set to eval().
    """

    # ----------------------------
    # 1. Resolve the config path
    # ----------------------------
    cfg_path = model_cfg

    # If it's not absolute and doesn't exist where we are, try relative to the sam2 package dir
    if not os.path.isabs(cfg_path) and not os.path.exists(cfg_path):
        sam2_dir = os.path.dirname(os.path.abspath(sam2.__file__))  # .../site-packages/sam2 or /workspace/sam2/sam2
        candidate = os.path.join(sam2_dir, model_cfg)
        if os.path.exists(candidate):
            cfg_path = candidate
        else:
            raise FileNotFoundError(
                f"Could not find SAM2 config file '{model_cfg}'. "
                f"Tried '{model_cfg}' and '{candidate}'."
            )

    # ----------------------------
    # 2. Load YAML into OmegaConf
    # ----------------------------
    cfg = OmegaConf.load(cfg_path)

    if "model" not in cfg:
        raise KeyError(
            f"SAM2 config at {cfg_path!r} does not contain a top-level 'model' key; "
            f"got top-level keys: {list(cfg.keys())}"
        )

    # ----------------------------
    # 3. Instantiate the model
    # ----------------------------
    # The YAML you showed is already Hydra-style with `_target_` keys
    # so `instantiate(cfg.model, _recursive_=True)` will build the full SAM2 model.
    model = instantiate(cfg.model, _recursive_=True)

    # ----------------------------
    # 4. Load checkpoint weights
    # ----------------------------
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")

        # Some SAM2 checkpoints are wrapped like {"model": state_dict}
        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)

        if missing_keys:
            print("SAM2 checkpoint missing keys: %s", missing_keys)
        if unexpected_keys:
            print("SAM2 checkpoint had unexpected keys: %s", unexpected_keys)

    # ----------------------------
    # 5. Move to device and eval
    # ----------------------------
    if device is not None:
        model = model.to(device)

    model.eval()
    return model

class SAM2InstanceSegmenter(AbsInstanceSegmenter):
    """
    Minimal SAM 2 wrapper for *automatic* (segment-everything) instance masks.

    Usage:
        seg = SAM2InstanceSegmenter(
            checkpoint=".../sam2_hiera_large.pt",
            model_cfg="sam2_hiera_l.yaml",
            device="cuda"
        )
        img = ...  # torch.uint8, (3, H, W), RGB
        res = seg.segment_instances(img)
    """

    def __init__(
        self,
        *,
        checkpoint: str,
        model_cfg: str,
        device: Optional[str] = None,
        mask_generator_kwargs: Optional[dict] = None,
        name: Optional[str] = None,
        min_area_perc: float = 0.0,
        dedupe_iou_threshold: float = 1.01,
        confidence_threshold: Optional[float] = None,
    ):
        # self._sam = build_sam2(
        #     model_cfg,
        #     checkpoint,
        #     device=device,
        # )

        if (name is None):
            # Throw an error if no name is provided
            raise ValueError("A name must be provided for the SAM2InstanceSegmenter.")
        super().__init__(
            name=name,
            min_area_perc=min_area_perc,
            dedupe_iou_threshold=dedupe_iou_threshold,
            confidence_threshold=confidence_threshold,
        )

        self._sam = customBuildSam2(
            model_cfg,
            checkpoint,
            device=device,
        )
        self._mask_generator_kwargs = mask_generator_kwargs or {}
        self._mask_generator = None  # created on first use

    def _ensure_mask_generator(self):
        if self._mask_generator is None:
            self._mask_generator = SAM2AutomaticMaskGenerator(
                self._sam, **self._mask_generator_kwargs
            )

    @staticmethod
    def _xywh_to_xyxy(xywh):
        x, y, w, h = map(float, xywh)
        return [x, y, x + w, y + h]

    def segmentInstancesRaw(self, image: torch.Tensor) -> SegmentationResults:
        """
        Args:
            image: torch.uint8 tensor, shape (3, H, W), RGB.

        Returns:
            SegmentationResults with:
                ids: (N,) int64
                masks:  (N, H, W) bool
                boxes:  (N, 4) XYXY float32
                scores: (N,) float32 (predicted IoU from SAM2)
        """
        # SAM2 expects HxWxC uint8 RGB numpy array
        img_np = image[:3].permute(1, 2, 0).contiguous().cpu().numpy()

        self._ensure_mask_generator()
        raw = self._mask_generator.generate(img_np)  # list[dict], each has 'segmentation', 'bbox' (xywh), 'predicted_iou', ...

        if len(raw) == 0:
            H, W = image.shape[-2:]
            return SegmentationResults.empty(h=H, w=W)

        ids = torch.arange(len(raw), dtype=torch.int64)

        # Convert to minimal tensors
        masks = torch.stack(
            [torch.from_numpy(r["segmentation"].astype(np.bool_)) for r in raw],
            dim=0,
        )  # (N, H, W) bool
        boxes = torch.tensor(
            [self._xywh_to_xyxy(r.get("bbox", [0, 0, 0, 0])) for r in raw],
            dtype=torch.float32,
        )  # (N, 4)
        scores = torch.tensor(
            [float(r.get("predicted_iou", 0.0)) for r in raw],
            dtype=torch.float32,
        )  # (N,)

        return SegmentationResults(ids=ids, masks=masks, boxes=boxes, scores=scores)


