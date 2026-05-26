"""Inferencer for VisionAnchorMidpoint-trained checkpoints.

The trainer (persistence_eval.methods.training.vision_anchor_midpoint) supervises
a (image, mask) -> scalar in log-seconds, where the target is
``log(t_end - anchor_time)`` and ``t_end`` is the per-track midpoint between
last-observed and first-missing. We treat the network's prediction as a t50
estimate and emit ``params={"t50": t50_seconds}`` so an exponential survival
generator (lam = t50 / ln(2)) can build the survival function downstream.

Per (image, mask) forward pass — no batching across the K masks in a frame, to
mirror how the trainer dataset feeds samples one at a time.
"""

from __future__ import annotations

from typing import Optional

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from persistence_eval.core.domain import ImageMaskGroup, PersistenceParameterEstimate
from persistence_eval.methods.inferencers.base import AbsPersistenceParameterInferencer


class VisionAnchorMidpointInferencer(AbsPersistenceParameterInferencer):
    def __init__(
        self,
        model_cfg,
        checkpoint_path: str,
        device: str = "cuda",
        image_size: int = 518,
        name: str | None = None,
    ) -> None:
        self._device = torch.device(device)
        self._image_size = int(image_size)
        self.name = name

        # Hydra recursively instantiates by default, so model_cfg may already be
        # the constructed nn.Module rather than the DictConfig.
        if isinstance(model_cfg, torch.nn.Module):
            self._model = model_cfg
        else:
            import hydra.utils
            self._model = hydra.utils.instantiate(model_cfg)
        self._model = self._model.to(self._device)
        self._model.eval()
        self._load_checkpoint(checkpoint_path)

        # Match AnchorMidpointDataset's image preprocessing exactly so eval-time
        # inputs are distributionally identical to training-time inputs.
        self._image_tx = T.Compose([
            T.Resize((self._image_size, self._image_size), interpolation=InterpolationMode.BILINEAR),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"VisionAnchorMidpointInferencer checkpoint not found: {path}")
        # weights_only=False because the trainer pickles full RNG state (numpy
        # arrays) alongside the model state_dict — incompatible with the
        # torch>=2.6 weights_only=True default. Trusted source (our own trainer).
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        self._model.load_state_dict(state)

    def _prep_image(self, image: torch.Tensor) -> torch.Tensor:
        # ImageMaskGroup.image is C×H×W uint8.
        if image.dtype != torch.uint8:
            image = image.to(torch.uint8)
        return self._image_tx(image).unsqueeze(0).to(self._device)  # 1×3×S×S

    def _prep_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask is H×W bool. Match dataset: float32, (1,1,H,W), nearest-resize to image_size.
        m = mask.to(torch.float32).unsqueeze(0).unsqueeze(0)
        m = F.interpolate(m, size=(self._image_size, self._image_size), mode="nearest")
        return m.to(self._device)  # 1×1×S×S

    def infer(
        self, group: ImageMaskGroup, *, one_off: bool = False,
    ) -> tuple[list[PersistenceParameterEstimate], Optional[float]]:
        img_tensor = self._prep_image(group.image)

        results: list[PersistenceParameterEstimate] = []
        with torch.no_grad():
            for mask in group.masks:
                mask_tensor = self._prep_mask(mask)
                pred = self._model(img_tensor, mask_tensor)
                # Model returns shape (B,) in log-seconds.
                log_t = float(pred.squeeze().detach().cpu().item())
                # Clamp to avoid overflow in exp() for pathological outputs.
                log_t = max(min(log_t, 30.0), -30.0)
                t50 = math.exp(log_t)
                results.append(
                    PersistenceParameterEstimate(
                        params={"t50": t50},
                        metadata={"pred_log_seconds": log_t},
                    )
                )
        return results, None
