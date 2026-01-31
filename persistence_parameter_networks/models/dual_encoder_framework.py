from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from abc import ABC, abstractmethod

from persistence_parameter_networks.img_encoders.abs_img_encoder import AbsImgEncoder
from utils.image_utils import apply_mask, crop_and_resize

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Combiner interface + options
# ----------------------------


class FeatureCombiner(nn.Module, ABC):
    @abstractmethod
    def forward(self, context_feat: torch.Tensor, instance_feat: torch.Tensor) -> torch.Tensor:
        """
        context_feat:  (B, nc)
        instance_feat: (B, ni) or (B, T, ni) depending on implementation
        returns downstream_feat: (B, d)
        """
        ...

class ConcatMLPCombiner(FeatureCombiner):
    """
    Concat [context; instance] then MLP -> (B, d)

    Supports:
      context_feat:  (B, nc)
      instance_feat: (B, ni)

    If nc/ni differ from the init values, this module expects you to pass the correct nc/ni
    at construction time.
    """
    def __init__(
        self,
        nc: int,
        ni: int,
        d: int,
        hidden: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.nc = nc
        self.ni = ni

        in_dim = nc + ni
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d),
        )

    def forward(self, context_feat: torch.Tensor, instance_feat: torch.Tensor) -> torch.Tensor:
        if context_feat.dim() != 2:
            raise ValueError(f"context_feat must be (B,nc), got {tuple(context_feat.shape)}")
        if instance_feat.dim() != 2:
            raise ValueError(f"instance_feat must be (B,ni), got {tuple(instance_feat.shape)}")
        if context_feat.shape[-1] != self.nc:
            raise ValueError(f"Expected context last dim {self.nc}, got {context_feat.shape[-1]}")
        if instance_feat.shape[-1] != self.ni:
            raise ValueError(f"Expected instance last dim {self.ni}, got {instance_feat.shape[-1]}")

        x = torch.cat([context_feat, instance_feat], dim=-1)
        return self.net(x)


class CrossAttnCombiner(FeatureCombiner):
    """
    Cross-attention combiner with optional projection for mismatched dims.

    context_feat:  (B, nc)                -- query token
    instance_feat: (B, ni) or (B, T, ni)  -- keys/values tokens
    Returns:       (B, d)

    Internally attends in a shared attention space of size nattn.

    Simplified case:
      - if nc == nattn, context projection is Identity
      - if ni == nattn, instance projection is Identity
    """
    def __init__(
        self,
        nc: int,
        ni: int,
        d: int,
        nattn: int | None = None,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        ff_hidden: int = 512,
        ff_dropout: float = 0.0,
        use_residual: bool = True,
        pre_norm: bool = True,
    ):
        super().__init__()

        self.nc = nc
        self.ni = ni
        self.use_residual = use_residual
        self.pre_norm = pre_norm

        # Attention embedding dimension (the space q/k/v live in)
        self.nattn = nattn if nattn is not None else max(nc, ni)
        if self.nattn % num_heads != 0:
            raise ValueError(f"nattn={self.nattn} must be divisible by num_heads={num_heads}")

        # Only project when needed
        self.q_proj = nn.Identity() if nc == self.nattn else nn.Linear(nc, self.nattn)
        self.kv_proj = nn.Identity() if ni == self.nattn else nn.Linear(ni, self.nattn)

        self.q_norm = nn.LayerNorm(self.nattn)
        self.kv_norm = nn.LayerNorm(self.nattn)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.nattn,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.attn_out_norm = nn.LayerNorm(self.nattn)

        self.ff = nn.Sequential(
            nn.Linear(self.nattn, ff_hidden),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(ff_hidden, self.nattn),
        )

        self.proj_out = nn.Linear(self.nattn, d)

    def forward(
        self,
        context_feat: torch.Tensor,   # (B, nc)
        instance_feat: torch.Tensor,  # (B, ni) or (B, T, ni)
    ) -> torch.Tensor:
        if context_feat.dim() != 2:
            raise ValueError(f"context_feat must be (B,nc), got {tuple(context_feat.shape)}")
        if context_feat.shape[-1] != self.nc:
            raise ValueError(f"Expected context last dim {self.nc}, got {context_feat.shape[-1]}")

        if instance_feat.dim() == 2:
            if instance_feat.shape[-1] != self.ni:
                raise ValueError(f"Expected instance last dim {self.ni}, got {instance_feat.shape[-1]}")
            instance_feat = instance_feat.unsqueeze(1)  # (B, 1, ni)
        elif instance_feat.dim() == 3:
            if instance_feat.shape[-1] != self.ni:
                raise ValueError(f"Expected instance last dim {self.ni}, got {instance_feat.shape[-1]}")
        else:
            raise ValueError(f"instance_feat must be (B,ni) or (B,T,ni), got {tuple(instance_feat.shape)}")

        # Project into shared attention space (identity when already matching)
        q = self.q_proj(context_feat).unsqueeze(1)   # (B, 1, nattn)
        kv = self.kv_proj(instance_feat)            # (B, T, nattn)

        # Attention (no mask)
        if self.pre_norm:
            qn = self.q_norm(q)
            kvn = self.kv_norm(kv)
            attn_out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        else:
            attn_out, _ = self.attn(q, kv, kv, need_weights=False)

        # Residual after attention
        x = (q + attn_out) if self.use_residual else attn_out
        x = self.attn_out_norm(x)

        # Feed-forward (+ optional residual)
        ff_out = self.ff(x)
        x = (x + ff_out) if self.use_residual else ff_out

        x = x.squeeze(1)          # (B, nattn)
        return self.proj_out(x)   # (B, d)


# ----------------------------
# Main model
# ----------------------------

@dataclass
class InstancePreprocessConfig:
    mode: str = "mask"  # "mask" or "crop" or "crop_mask"
    crop_size: Tuple[int, int] = (224, 224)  # not used if mode="crop_mask"
    mask_fill: float = 0.0  # background fill for mode="mask" or "crop_mask"


class DualEncoderModel(nn.Module):
    """
    Dual-encoder model with vector-only encoder outputs.

    Steps:
      1) context_feat  = context_encoder(image)        -> (B, nc)
      2) instance_img  = preprocess(image, mask)
         instance_feat = instance_encoder(instance_img) -> (B, ni)
      3) downstream_feat = combiner(context_feat, instance_feat) -> (B, D)
      4) output = head(downstream_feat) -> (B, M)

    Notes:
      - Encoders must implement getOutputFeatureLength() -> int
      - head can be provided; if not, a default MLP head is used.
    """
    def __init__(
        self,
        context_encoder: AbsImgEncoder,
        instance_encoder: AbsImgEncoder,
        outputDim: int,
        combiner: FeatureCombiner | None = None,
        downstream_dim: int = 512,
        preprocess_cfg: Optional[InstancePreprocessConfig] = None,
        head: Optional[nn.Module] = None,
        head_hidden: int = 256,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.instance_encoder = instance_encoder
        self.preprocess_cfg = preprocess_cfg if preprocess_cfg is not None else InstancePreprocessConfig()

        # Query encoder output dimensions
        if not hasattr(context_encoder, "getOutputFeatureLength"):
            raise AttributeError("context_encoder must implement getOutputFeatureLength()")
        if not hasattr(instance_encoder, "getOutputFeatureLength"):
            raise AttributeError("instance_encoder must implement getOutputFeatureLength()")

        self.nc = int(context_encoder.getOutputDim())
        self.ni = int(instance_encoder.getOutputDim())

        # Default combiner (handles nc != ni) if not provided
        if combiner is None:
            self.combiner = ConcatMLPCombiner(
                nc=self.nc,
                ni=self.ni,
                d=downstream_dim,
                hidden=max(downstream_dim, 2 * downstream_dim),
                dropout=head_dropout,
            )
        else:
            self.combiner = combiner

        # Head: user-specified or default
        if head is None:
            self.head = nn.Sequential(
                nn.LayerNorm(downstream_dim),
                nn.Linear(downstream_dim, head_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, outputDim),
            )
        else:
            self.head = head

    def make_instance_image(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        cfg = self.preprocess_cfg
        if cfg.mode == "mask":
            return apply_mask(image, mask, fill=cfg.mask_fill)
        elif cfg.mode == "crop":
            return crop_and_resize(image, mask, out_hw=cfg.crop_size)
        elif cfg.mode == "crop_mask":
            cropped = crop_and_resize(image, mask, out_hw=cfg.crop_size)
            cropped_mask = crop_and_resize(mask.unsqueeze(1), mask, out_hw=cfg.crop_size).squeeze(1)
            return apply_mask(cropped, cropped_mask, fill=cfg.mask_fill)
        else:
            raise ValueError(f"Unknown preprocess mode: {cfg.mode}")

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 1) context encoding
        context_feat = self.context_encoder(image)  # (B, nc)
        if context_feat.dim() != 2:
            raise ValueError(f"context_encoder must return (B,nc), got {tuple(context_feat.shape)}")

        # 2) instance encoding
        instance_img = self.make_instance_image(image, mask)
        instance_feat = self.instance_encoder(instance_img)  # (B, ni)
        if instance_feat.dim() != 2:
            raise ValueError(f"instance_encoder must return (B,ni), got {tuple(instance_feat.shape)}")

        # 3) combine
        downstream_feat = self.combiner(context_feat, instance_feat)  # (B, D)

        # 4) head
        out = self.head(downstream_feat)

        # Optional: helpful check that custom heads output (B,m)
        if out.dim() != 2 or out.shape[0] != image.shape[0]:
            raise ValueError(
                f"head must return a 2D tensor (B, m). Got {tuple(out.shape)} for batch {image.shape[0]}"
            )
        return out
