"""Vision-only single-scalar predictor (image + mask -> scalar log-time).

Composes the same backbone + pooler + cross-attention head as
`PersistenceObjConditionedCrossAttn`, but with `outputLength=1` and a
forward that returns a 1-D tensor of shape (B,) — appropriate for direct
log-MSE supervision, not the (y50, yr, yc) distillation contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from persistence_parameter_networks.models.attention_based_predictor import (
    MultiMaskObjectConditionedCrossAttnPredictor,
    PoolerConfig,
    build_pooler,
)
from persistence_parameter_networks.models.img_encoders.selavpr.rich_img_encoder import (
    RichOutputSelaEncoder,
    RichOutputSelaEncoderArgs,
)


class VisionScalarPredictor(nn.Module):
    def __init__(
        self,
        backbone_cfg: RichOutputSelaEncoderArgs,
        pooler_cfg: PoolerConfig,
        attn_dim: int | None = None,
        num_heads: int = 8,
        ff_hidden: int = 512,
        name: str = "vision_scalar_predictor",
    ) -> None:
        super().__init__()
        backbone = RichOutputSelaEncoder(backbone_cfg)
        pooler = build_pooler(pooler_cfg)
        self.predictor = MultiMaskObjectConditionedCrossAttnPredictor(
            backbone=backbone,
            outputLength=1,
            pooler=pooler,
            attn_dim=attn_dim,
            num_heads=num_heads,
            ff_hidden=ff_hidden,
        )
        self.name = f"{name}__{pooler_cfg.kind}"

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Inner predictor returns (B, 1) for single-mask input.
        out = self.predictor(image, mask)
        return out.squeeze(-1)
