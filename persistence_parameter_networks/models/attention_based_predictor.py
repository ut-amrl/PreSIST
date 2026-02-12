import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC
from typing import Optional

from persistence_parameter_networks.models.abs_raw_predictor import AbsRawPredictor, ensure_bk1hw

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Utility: resize masks to ViT patch grid
# ------------------------------------------------------------
def resize_mask_to_hw(mask: torch.Tensor, H: int, W: int):
    """
    mask: (B, 1, H_img, W_img)
    returns: (B, H, W)
    """
    # Use nearest-neighbor to preserve mask structure
    m = F.interpolate(mask.float(), size=(H, W), mode="nearest")
    return m.squeeze(1)  # (B, H, W)


# ------------------------------------------------------------
# Utility: masked average pooling over patch tokens
# ------------------------------------------------------------
def masked_token_pooling(tokens_hw: torch.Tensor, mask_hw: torch.Tensor):
    """
    tokens_hw: (B, H, W, C)
    mask_hw:   (B, H, W)  (0/1 or soft)
    returns:   (B, C)
    """
    B, H, W, C = tokens_hw.shape

    w = mask_hw.reshape(B, H * W)
    t = tokens_hw.reshape(B, H * W, C)

    w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-6)
    pooled = (t * w.unsqueeze(-1)).sum(dim=1) / w_sum
    return pooled  # (B, C)


class BaseObjectPooler(nn.Module):
    """
    forward(out, masks) -> q: (B, K, Cq)
    where masks: (B, K, 1, H_img, W_img)
    """
    def forward(self, out: dict, masks_bk1hw: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LocalMapPooler(BaseObjectPooler):
    def __init__(self, mask_mode: Literal["nearest", "bilinear"] = "nearest", eps: float = 1e-6):
        super().__init__()
        self.mask_mode = mask_mode
        self.eps = eps

    def forward(self, out: dict, mask: torch.Tensor) -> torch.Tensor:
        if "local_map" not in out:
            raise KeyError("Backbone output missing 'local_map'. Construct GeoLocalizationNet with use_local=True.")

        local_map = out["local_map"]                 # (B, 128, Hl, Wl)
        Hl, Wl = out["local_hw"]

        # Resize mask to local grid
        m_local = F.interpolate(
            mask.float(),
            size=(Hl, Wl),
            mode=self.mask_mode,
            align_corners=False if self.mask_mode in ("bilinear", "bicubic") else None
        )                                             # (B, 1, Hl, Wl)

        q = masked_avg_pool_map(local_map, m_local, eps=self.eps)  # (B, 128)
        return q


class DilatedMaskTokenPooler(BaseObjectPooler):
    def __init__(self, mask_mode: Literal["nearest", "bilinear"] = "nearest", dilate_iters: int = 1, eps: float = 1e-6):
        super().__init__()
        self.mask_mode = mask_mode
        self.dilate_iters = dilate_iters
        self.eps = eps

    @staticmethod
    def _dilate_binary_hw(mask_hw: torch.Tensor, iters: int = 1) -> torch.Tensor:
        """
        mask_hw: (B, H, W) float/binary. We'll treat >0 as 1 for dilation.
        Returns float mask (0/1).
        """
        m = (mask_hw > 0).float().unsqueeze(1)  # (B, 1, H, W)
        for _ in range(iters):
            m = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
        return m.squeeze(1)  # (B, H, W)

    def forward(self, out: dict, mask: torch.Tensor) -> torch.Tensor:
        tokens_hw = out["patch_tokens_hw"]
        H, W = out["grid_hw"]

        weights_hw = resize_mask(mask, (H, W), mode=self.mask_mode)  # (B, H, W)

        # If mask collapses (sum==0), dilation won't help; but if it is tiny (sum small), dilation helps.
        weights_hw = self._dilate_binary_hw(weights_hw, iters=self.dilate_iters)

        q = masked_avg_pool_tokens_hw(tokens_hw, weights_hw, eps=self.eps)
        return q

# class ObjectConditionedCrossAttnPredictor(AbsRawPredictor):
#     def __init__(
#         self,
#         backbone: nn.Module,
#         outputLength: int,
#         pooler: Optional[nn.Module] = None,
#         attn_dim: Optional[int] = None,
#         num_heads: int = 8,
#         ff_hidden: int = 512
#     ):
#         super().__init__(outputLength)

#         self.backbone = backbone
#         self.pooler = pooler or MaskDownsampleTokenPooler(mask_mode="nearest")  # default = your current behavior

#         # Infer token dim for K/V from backbone
#         test_out_dim = None
#         with torch.no_grad():
#             try:
#                 dummy = torch.zeros(1, 3, backbone.img_hw[0], backbone.img_hw[1])
#                 out = backbone(dummy)
#                 test_out_dim = out["patch_tokens"].shape[-1]
#             except Exception:
#                 test_out_dim = getattr(backbone, "features_dim", None)
#         if test_out_dim is None:
#             raise ValueError("Cannot infer DINO token dimension from backbone output.")

#         token_dim = test_out_dim
#         attn_dim = attn_dim or token_dim

#         # We allow pooler output dimension to differ from token_dim (e.g., local_map is 128)
#         # So we project q from q_dim -> attn_dim, and project kv from token_dim -> attn_dim.
#         self.q_proj = nn.Linear(self._infer_pooler_dim(backbone), attn_dim)
#         self.kv_proj = nn.Linear(token_dim, attn_dim)

#         self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True)

#         self.ff = nn.Sequential(
#             nn.LayerNorm(attn_dim),
#             nn.Linear(attn_dim, ff_hidden),
#             nn.GELU(),
#             nn.Linear(ff_hidden, attn_dim)
#         )

#         self.head = nn.Sequential(
#             nn.LayerNorm(attn_dim),
#             nn.Linear(attn_dim, outputLength)
#         )

#     def _infer_pooler_dim(self, backbone: nn.Module) -> int:
#         # Run a tiny forward to get q dim from pooler, robustly.
#         with torch.no_grad():
#             dummy = torch.zeros(1, 3, backbone.img_hw[0], backbone.img_hw[1])
#             out = backbone(dummy)
#             dummy_mask = torch.zeros(1, 1, backbone.img_hw[0], backbone.img_hw[1])
#             q = self.pooler(out, dummy_mask)
#             return q.shape[-1]

#     def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         out = self.backbone(image)

#         patch_tokens = out["patch_tokens"]           # (B, N, C_token)

#         # 1) Build object query q using the selected pooling strategy
#         q = self.pooler(out, mask)                   # (B, C_q)

#         # 2) Project and cross-attend
#         q = self.q_proj(q).unsqueeze(1)              # (B, 1, attn_dim)
#         kv = self.kv_proj(patch_tokens)              # (B, N, attn_dim)

#         a, _ = self.attn(q, kv, kv)                  # (B, 1, attn_dim)

#         x = a.squeeze(1)
#         x = self.ff(x)

#         return self.head(x)





def resize_mask(mask_b1hw: torch.Tensor, size_hw: tuple[int, int], mode: str = "nearest") -> torch.Tensor:
    m = F.interpolate(
        mask_b1hw.float(),
        size=size_hw,
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None
    )
    return m.squeeze(1)  # (B, H, W)

def masked_avg_pool_tokens_hw(tokens_hw: torch.Tensor, weights_hw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, H, W, C = tokens_hw.shape
    t = tokens_hw.reshape(B, H * W, C)
    w = weights_hw.reshape(B, H * W)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=eps)
    return (t * w.unsqueeze(-1)).sum(dim=1) / w_sum  # (B, C)

def masked_avg_pool_map(map_bchw: torch.Tensor, mask_b1hw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = mask_b1hw.float()
    w_sum = m.sum(dim=(2, 3)).clamp(min=eps)          # (B, 1)
    return (map_bchw * m).sum(dim=(2, 3)) / w_sum     # (B, C)



class MaskDownsampleTokenPooler(BaseObjectPooler):
    """
    Downsample each mask to token grid and pool patch_tokens_hw.
    """
    def __init__(self, mask_mode: Literal["nearest", "bilinear"] = "nearest", eps: float = 1e-6):
        super().__init__()
        self.mask_mode = mask_mode
        self.eps = eps

    def forward(self, out: dict, masks_bk1hw: torch.Tensor) -> torch.Tensor:
        tokens_hw = out["patch_tokens_hw"]      # (B, Ht, Wt, C)
        Ht, Wt = out["grid_hw"]

        B, K, _, H_img, W_img = masks_bk1hw.shape
        qs = []
        for k in range(K):
            mask_b1hw = masks_bk1hw[:, k]       # (B, 1, H_img, W_img)
            w_hw = resize_mask(mask_b1hw, (Ht, Wt), mode=self.mask_mode)  # (B, Ht, Wt)
            qk = masked_avg_pool_tokens_hw(tokens_hw, w_hw, eps=self.eps) # (B, C)
            qs.append(qk)

        return torch.stack(qs, dim=1)           # (B, K, C)


class FeatureUpsampleImagePooler(BaseObjectPooler):
    """
    Upsample patch_token_map to image resolution and pool with original (high-res) mask.
    """
    def __init__(self, up_mode: Literal["bilinear", "bicubic"] = "bilinear", eps: float = 1e-6):
        super().__init__()
        self.up_mode = up_mode
        self.eps = eps

    def forward(self, out: dict, masks_bk1hw: torch.Tensor) -> torch.Tensor:
        patch_token_map = out["patch_token_map"]      # (B, C, Ht, Wt)
        B, K, _, H_img, W_img = masks_bk1hw.shape

        feat_img = F.interpolate(
            patch_token_map,
            size=(H_img, W_img),
            mode=self.up_mode,
            align_corners=False
        )                                             # (B, C, H_img, W_img)

        qs = []
        for k in range(K):
            qk = masked_avg_pool_map(feat_img, masks_bk1hw[:, k], eps=self.eps)  # (B, C)
            qs.append(qk)

        return torch.stack(qs, dim=1)                 # (B, K, C)


class MultiMaskObjectConditionedCrossAttnPredictor(nn.Module):
    """
    Updated version of your cross-attn predictor:
      - Accepts mask: (B,1,H,W) or (B,K,1,H,W)
      - Outputs: (B,outputLength) or (B,K,outputLength)
    """
    def __init__(
        self,
        backbone: nn.Module,
        outputLength: int,
        pooler: nn.Module,
        attn_dim: int | None = None,
        num_heads: int = 8,
        ff_hidden: int = 512,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooler = pooler

        # Infer token dim
        token_dim = getattr(backbone, "features_dim", None)
        if token_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, backbone.img_hw[0], backbone.img_hw[1])
                out = backbone(dummy)
                token_dim = out["patch_tokens"].shape[-1]
        self.token_dim = token_dim

        # Infer pooler dim (Cq)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, backbone.img_hw[0], backbone.img_hw[1])
            out = backbone(dummy)
            dummy_mask = torch.zeros(1, 1, 1, backbone.img_hw[0], backbone.img_hw[1])
            q = pooler(out, dummy_mask)               # (1, K=1, Cq)
            q_dim = q.shape[-1]

        self.attn_dim = attn_dim or token_dim

        self.q_proj = nn.Linear(q_dim, self.attn_dim)
        self.kv_proj = nn.Linear(token_dim, self.attn_dim)

        self.attn = nn.MultiheadAttention(self.attn_dim, num_heads=num_heads, batch_first=True)

        self.ff = nn.Sequential(
            nn.LayerNorm(self.attn_dim),
            nn.Linear(self.attn_dim, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, self.attn_dim),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(self.attn_dim),
            nn.Linear(self.attn_dim, outputLength),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masks_bk1hw, multi = ensure_bk1hw(mask)     # (B, K, 1, H, W)
        B, K, _, _, _ = masks_bk1hw.shape

        out = self.backbone(image)
        patch_tokens = out["patch_tokens"]          # (B, N, C_token)

        # Build K object queries
        q = self.pooler(out, masks_bk1hw)           # (B, K, Cq)

        # Project
        q = self.q_proj(q)                          # (B, K, attn_dim)
        kv = self.kv_proj(patch_tokens)             # (B, N, attn_dim)

        # Flatten queries for attention: treat each object as its own batch element
        q_flat = q.reshape(B * K, 1, self.attn_dim)               # (B*K, 1, D)
        kv_flat = kv.unsqueeze(1).expand(B, K, *kv.shape[1:])     # (B, K, N, D)
        kv_flat = kv_flat.reshape(B * K, kv.shape[1], self.attn_dim)  # (B*K, N, D)

        a, _ = self.attn(q_flat, kv_flat, kv_flat)  # (B*K, 1, D)
        x = a.squeeze(1)                            # (B*K, D)
        x = self.ff(x)
        pred = self.head(x)                         # (B*K, outputLength)

        pred = pred.reshape(B, K, -1)               # (B, K, outputLength)
        return pred[:, 0] if not multi else pred



# # Variant A1: original behavior
# pooler = MaskDownsampleTokenPooler(mask_mode="nearest")

# # Variant A2: soft mask downsample (often a strong baseline improvement)
# pooler = MaskDownsampleTokenPooler(mask_mode="bilinear")

# # Variant B: upsample features to image then pool with original mask
# pooler = FeatureUpsampleImagePooler(up_mode="bilinear")

# # Variant C: pool on learned local_map (requires GeoLocalizationNet(use_local=True))
# pooler = LocalMapPooler(mask_mode="nearest")        # or "bilinear"

# # Variant D: token pooling + dilation rescue for tiny masks
# pooler = DilatedMaskTokenPooler(mask_mode="nearest", dilate_iters=1)
