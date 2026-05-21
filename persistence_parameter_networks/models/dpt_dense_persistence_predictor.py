"""DPT-style dense persistence predictor.

Decoder follows the recipe from Ranftl et al. "Vision Transformers for
Dense Prediction" (2021):

  1. Tap 4 intermediate ViT layers (provided by
     :class:`DenseRichOutputSelaEncoder`).
  2. **Reassemble** each tap to a per-stage spatial resolution by 1x1
     projection + transposed/strided convolution. Earliest layers are
     upsampled, latest are downsampled, producing a multi-scale pyramid.
  3. **Fusion**: starting from the coarsest stage, apply Residual Conv
     Units, ×2 bilinear upsample, and add the next-finer reassembled
     stage. Repeat until we are back at the highest pyramid resolution.
  4. **Head**: a small conv stack produces 3 output channels, then
     bilinear upsample to the input image resolution.

Channel layout follows the per-mask model: ``[log p25, log(p75/p25 - 1),
conf]``. The conf channel is passed through ``torch.sigmoid`` only when
``confidence_from_logits`` is False, matching
:class:`PersistenceOutputPredictor`.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from persistence_parameter_networks.models.dense_persistence_predictor import (
    DensePersistencePredictor,
)
from persistence_parameter_networks.models.img_encoders.selavpr.dense.dense_rich_img_encoder import (
    DenseRichOutputSelaEncoder,
)
from persistence_parameter_networks.models.img_encoders.selavpr.rich_img_encoder import (
    RichOutputSelaEncoderArgs,
)


class _ResidualConvUnit(nn.Module):
    """Two 3x3 convs with a residual connection. DPT's basic building block."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.activation(x))
        out = self.conv2(self.activation(out))
        return out + x


class _FusionBlock(nn.Module):
    """Combine a previous-stage feature with the current skip, then ×2 upsample.

    Following DPT's FeatureFusionBlock_custom: each input goes through a
    Residual Conv Unit, the two are summed, a final RCU refines the
    result, then we upsample by 2.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.rcu_skip = _ResidualConvUnit(dim)
        self.rcu_out = _ResidualConvUnit(dim)
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(
        self,
        prev: torch.Tensor | None,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        skip = self.rcu_skip(skip)
        if prev is not None:
            if prev.shape[-2:] != skip.shape[-2:]:
                prev = F.interpolate(
                    prev, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = prev + skip
        else:
            x = skip
        x = self.rcu_out(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.out_conv(x)
        return x


class _Reassemble(nn.Module):
    """Project ViT token map to ``out_dim`` channels and rescale spatially.

    ``scale > 1`` upsamples by ``scale`` via transposed convolution;
    ``scale < 1`` downsamples by ``1/scale`` via strided convolution;
    ``scale == 1`` keeps spatial size.
    """

    def __init__(self, in_dim: int, out_dim: int, scale: float):
        super().__init__()
        # 1x1 channel projection (no spatial change) — done before any resampling.
        self.project = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=True)

        if scale > 1.0:
            stride = int(round(scale))
            self.resample: nn.Module = nn.ConvTranspose2d(
                out_dim, out_dim, kernel_size=stride, stride=stride, padding=0
            )
        elif scale < 1.0:
            inv = int(round(1.0 / scale))
            # Strided convs of kernel=3, stride=inv (matches DPT-Base for
            # 1/2 and 1/4 downsamples).
            self.resample = nn.Conv2d(
                out_dim, out_dim, kernel_size=3, stride=inv, padding=1
            )
        else:
            self.resample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        x = self.resample(x)
        return x


class DPTDecoder(nn.Module):
    """DPT decoder over 4 intermediate token maps.

    Stages are ordered shallow -> deep: ``stages[0]`` is the earliest tap
    (and will be upsampled the most), ``stages[-1]`` is the latest tap
    (most downsampled). Fusion proceeds from deep -> shallow.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        decoder_dim: int = 256,
        scales: Sequence[float] = (2.0, 1.0, 0.5, 0.25),
    ):
        super().__init__()
        assert len(scales) == 4, "DPT expects exactly 4 taps"
        self.reassembles = nn.ModuleList(
            [_Reassemble(in_dim, decoder_dim, s) for s in scales]
        )
        self.fusions = nn.ModuleList([_FusionBlock(decoder_dim) for _ in scales])

    def forward(self, token_maps: List[torch.Tensor]) -> torch.Tensor:
        assert len(token_maps) == len(self.reassembles), (
            f"expected {len(self.reassembles)} taps, got {len(token_maps)}"
        )
        reassembled = [r(t) for r, t in zip(self.reassembles, token_maps)]
        # Iterate from deepest (coarsest) back to shallowest.
        path: torch.Tensor | None = None
        for skip, fuse in zip(reversed(reassembled), reversed(self.fusions)):
            path = fuse(path, skip)
        # path is at 2x the resolution of the shallowest reassembled map
        # (because each FusionBlock ends with ×2 upsample). For our
        # default scales (2, 1, 0.5, 0.25), the shallowest is 2x patch
        # grid ≈ 74; path is therefore ≈ 148.
        return path


class DPTDensePersistencePredictor(DensePersistencePredictor):
    """DPT-style decoder on top of DINOv2 + SELA adapters.

    Args:
        backbone_cfg: forwarded to :class:`DenseRichOutputSelaEncoder`.
        decoder_dim: channel width inside the DPT decoder.
        scales: relative spatial scales for the 4 reassembled stages
            (shallow -> deep). Defaults to ``(2, 1, 0.5, 0.25)``.
        confidence_from_logits: when False, applies sigmoid to the conf
            channel of the dense output.
        name: model_name used for output directories.
    """

    def __init__(
        self,
        backbone_cfg: RichOutputSelaEncoderArgs,
        decoder_dim: int = 256,
        scales: Sequence[float] = (2.0, 1.0, 0.5, 0.25),
        confidence_from_logits: bool = False,
        name: str = "dpt_dense_predictor",
        **_unused_kwargs,
    ):
        # The base Hydra ``config.yaml`` unconditionally injects a
        # ``model/cross_attn_pooler`` default which surfaces as
        # ``pooler_cfg`` on whichever model we instantiate. The dense
        # predictor has no use for it; swallow any such legacy kwargs
        # rather than forcing every dense top-level config to override
        # the base defaults list.
        if _unused_kwargs:
            print(
                f"[DPTDensePersistencePredictor] ignoring unused kwargs: "
                f"{sorted(_unused_kwargs.keys())}"
            )
        super().__init__(
            name=name, confidence_from_logits=confidence_from_logits
        )
        self.backbone = DenseRichOutputSelaEncoder(backbone_cfg)
        self.decoder = DPTDecoder(
            in_dim=int(self.backbone.features_dim),
            decoder_dim=decoder_dim,
            scales=scales,
        )
        self.head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim // 2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim // 2, 3, kernel_size=1, bias=True),
        )

    def predict_dense(self, image: torch.Tensor) -> torch.Tensor:
        encoder_out = self.backbone(image)
        token_maps: List[torch.Tensor] = encoder_out["intermediate_token_maps"]
        decoded = self.decoder(token_maps)
        out = self.head(decoded)
        # Bilinear upsample to input resolution.
        out = F.interpolate(
            out, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        # Apply sigmoid to the confidence channel (index 2) when the loss
        # expects post-sigmoid values. Matches PersistenceOutputPredictor.
        if not self.confidence_from_logits:
            y50 = out[:, 0:1]
            yr = out[:, 1:2]
            yc = torch.sigmoid(out[:, 2:3])
            out = torch.cat([y50, yr, yc], dim=1)
        return out
