import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from persistence_parameter_networks.models.abs_raw_predictor import AbsRawPredictor, ensure_bk1hw


# ----------------------------
# Small helper: pooling on token grid
# ----------------------------
def resize_mask_to_hw(mask_b1hw: torch.Tensor, H: int, W: int, mode: str = "bilinear") -> torch.Tensor:
    """
    mask_b1hw: (B, 1, H_img, W_img) float/binary
    returns:   (B, H, W) float weights
    """
    m = F.interpolate(
        mask_b1hw.float(),
        size=(H, W),
        mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic") else None,
    )
    return m.squeeze(1)  # (B, H, W)


def masked_token_pooling(tokens_hw: torch.Tensor, mask_hw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    tokens_hw: (B, H, W, C)
    mask_hw:   (B, H, W) float weights
    returns:   (B, C)
    """
    B, H, W, C = tokens_hw.shape
    t = tokens_hw.reshape(B, H * W, C)
    w = mask_hw.reshape(B, H * W)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=eps)
    return (t * w.unsqueeze(-1)).sum(dim=1) / w_sum


class DETRDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float, interact_queries: bool):
        super().__init__()
        self.interact_queries = interact_queries

        if interact_queries:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(d_model)
        else:
            self.self_attn = None
            self.norm1 = None

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # q: (B, K, D), memory: (B, N, D)
        if self.interact_queries:
            q2, _ = self.self_attn(q, q, q)
            q = self.norm1(q + self.dropout(q2))

        q2, _ = self.cross_attn(q, memory, memory)
        q = self.norm2(q + self.dropout(q2))

        q2 = self.ffn(q)
        q = self.norm3(q + self.dropout(q2))
        return q


class DETRLikeMaskConditionedPersistencePredictor(AbsRawPredictor):
    """
    - mask: (B,1,H,W) or (B,K,1,H,W)
    - output: (B,outputLength) or (B,K,outputLength)
    """
    def __init__(
        self,
        backbone: nn.Module,
        outputLength: int,
        d_model: int | None = None,
        nhead: int = 8,
        num_layers: int = 3,
        dim_ff: int = 1024,
        dropout: float = 0.0,
        interact_queries: bool = True,       # <-- constructor argument you asked for
        mask_resize_mode: str = "bilinear",
        init_from_mask_pool: bool = True,
        learned_query_fallback: bool = True,
        max_queries: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.backbone = backbone
        self.outputLength = outputLength
        self.mask_resize_mode = mask_resize_mode
        self.init_from_mask_pool = init_from_mask_pool
        self.learned_query_fallback = learned_query_fallback
        self.eps = eps

        # Infer token dim
        token_dim = getattr(backbone, "features_dim", None)
        if token_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, backbone.img_hw[0], backbone.img_hw[1])
                out = backbone(dummy)
                token_dim = out["patch_tokens"].shape[-1]

        self.token_dim = token_dim
        self.d_model = d_model or token_dim

        self.mem_proj = nn.Linear(self.token_dim, self.d_model)
        self.q_init_proj = nn.Linear(self.token_dim, self.d_model) if init_from_mask_pool else None

        if learned_query_fallback:
            self.learned_queries = nn.Parameter(torch.randn(1, max_queries, self.d_model) * 0.02)
        else:
            self.learned_queries = None

        self.layers = nn.ModuleList([
            DETRDecoderLayer(
                d_model=self.d_model,
                nhead=nhead,
                dim_ff=dim_ff,
                dropout=dropout,
                interact_queries=interact_queries
            )
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, outputLength)
        )

    def _init_queries_from_masks(self, out: dict, masks_bk1hw: torch.Tensor) -> torch.Tensor:
        tokens_hw = out["patch_tokens_hw"]     # (B, Ht, Wt, C)
        Ht, Wt = out["grid_hw"]
        B, K, _, H_img, W_img = masks_bk1hw.shape

        qs = []
        for k in range(K):
            m = masks_bk1hw[:, k]  # (B,1,H_img,W_img)
            m_hw = F.interpolate(
                m.float(),
                size=(Ht, Wt),
                mode=self.mask_resize_mode,
                align_corners=False if self.mask_resize_mode in ("bilinear", "bicubic") else None
            ).squeeze(1)  # (B, Ht, Wt)

            # masked pooling on token grid
            B2, H2, W2, C = tokens_hw.shape
            t = tokens_hw.reshape(B2, H2 * W2, C)
            w = m_hw.reshape(B2, H2 * W2)
            w_sum = w.sum(dim=1, keepdim=True).clamp(min=self.eps)
            qk = (t * w.unsqueeze(-1)).sum(dim=1) / w_sum   # (B, C)
            qs.append(qk)

        q0 = torch.stack(qs, dim=1)            # (B, K, C_token)
        q0 = self.q_init_proj(q0)              # (B, K, d_model)
        return q0

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masks_bk1hw, multi = ensure_bk1hw(mask)      # (B, K, 1, H, W)
        B, K, _, _, _ = masks_bk1hw.shape

        out = self.backbone(image)
        patch_tokens = out["patch_tokens"]           # (B, N, C_token)
        memory = self.mem_proj(patch_tokens)         # (B, N, d_model)

        q = None
        if self.init_from_mask_pool:
            q = self._init_queries_from_masks(out, masks_bk1hw)  # (B, K, d_model)

        if self.learned_query_fallback:
            if self.learned_queries is None:
                raise RuntimeError("learned_query_fallback=True but learned_queries is None")
            if K > self.learned_queries.shape[1]:
                raise ValueError(f"K={K} exceeds max_queries={self.learned_queries.shape[1]}")
            learned = self.learned_queries[:, :K].expand(B, K, -1)  # (B, K, d_model)
            q = learned if q is None else (q + learned)

        if q is None:
            raise ValueError("No query initialization selected.")

        for layer in self.layers:
            q = layer(q, memory)                         # (B, K, d_model)

        pred = self.head(q)                              # (B, K, outputLength)
        return pred[:, 0] if not multi else pred
