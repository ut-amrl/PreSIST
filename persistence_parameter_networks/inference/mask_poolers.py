"""Pool a dense per-pixel prediction map into per-instance vectors.

Used by :class:`DenseAdapter` to bridge a dense ``predict_dense(image)``
output to the unified ``predict_persistence(image, masks, batch_index)``
contract, and by the dense task's validation facade to provide a
per-mask forward to the existing validator.

All poolers accept the same flat/concat representation defined by the
project (see plan ``§ Mask representation``):

    dense:           (B, C, H, W)
    instance_masks:  (sum_N_i, H, W)   non-negative; ``> 0`` = inside instance
    batch_index:     (sum_N_i,)        int — which image each row belongs to

Returns:

    pooled:          (sum_N_i, C)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class MaskPooler(ABC):
    @abstractmethod
    def pool(
        self,
        dense: torch.Tensor,            # (B, C, H, W)
        instance_masks: torch.Tensor,   # (Σ N_i, H, W)
        batch_index: torch.Tensor,      # (Σ N_i,)
    ) -> torch.Tensor:                  # (Σ N_i, C)
        raise NotImplementedError


def _per_image_pool(
    dense: torch.Tensor,
    instance_masks: torch.Tensor,
    batch_index: torch.Tensor,
    *,
    reducer,
) -> torch.Tensor:
    """Iterate images and call ``reducer(dense_b, masks_b)`` on each.

    The B-length Python loop avoids the ``dense[batch_index]`` gather,
    which would materialize a ``(Σ N_i, C, H, W)`` tensor and re-spend
    the memory we saved by going flat in the first place. B is the
    training batch size (typically ≤ 32), so this loop is cheap.
    """
    B, C, _, _ = dense.shape
    N = int(instance_masks.shape[0])
    out = dense.new_zeros((N, C))
    if N == 0:
        return out
    for b in range(B):
        sel = (batch_index == b).nonzero(as_tuple=False).view(-1)
        if sel.numel() == 0:
            continue
        masks_b = instance_masks[sel]              # (n_b, H, W)
        dense_b = dense[b]                         # (C, H, W)
        out[sel] = reducer(dense_b, masks_b)
    return out


class MaskedMeanPooler(MaskPooler):
    """Per-channel mean of dense values inside each instance mask.

    The default pooler; matches what an averaging instance-level loss
    would expect.
    """

    def pool(self, dense, instance_masks, batch_index):
        def reducer(dense_b, masks_b):
            w = (masks_b > 0).to(dense_b.dtype)                 # (n_b, H, W)
            area = w.sum(dim=(-1, -2)).clamp(min=1.0)           # (n_b,)
            # (n_b, C) = Σ_{h,w} dense_b[c,h,w] * w[n,h,w] / area
            # Implement with einsum to keep memory bounded.
            num = torch.einsum("chw,nhw->nc", dense_b, w)
            return num / area.unsqueeze(-1)
        return _per_image_pool(
            dense, instance_masks, batch_index, reducer=reducer
        )


class CenterPixelPooler(MaskPooler):
    """Read the dense prediction at the mask centroid pixel.

    Useful as a baseline / sanity-check pooler. Falls back to the
    masked-mean if the centroid pixel happens to be off the mask
    (geometrically the centroid of an L-shape can lie outside it).
    """

    def pool(self, dense, instance_masks, batch_index):
        _, C, H, W = dense.shape

        def reducer(dense_b, masks_b):
            w = (masks_b > 0).to(torch.float32)              # (n_b, H, W)
            area = w.sum(dim=(-1, -2)).clamp(min=1.0)         # (n_b,)
            ys = torch.arange(H, device=dense_b.device, dtype=torch.float32)
            xs = torch.arange(W, device=dense_b.device, dtype=torch.float32)
            cy = (w * ys.view(1, H, 1)).sum(dim=(-1, -2)) / area
            cx = (w * xs.view(1, 1, W)).sum(dim=(-1, -2)) / area
            cy_idx = cy.round().clamp(0, H - 1).long()        # (n_b,)
            cx_idx = cx.round().clamp(0, W - 1).long()
            # Read at centroid; if centroid is outside mask, fall back to mean.
            on_mask = w[torch.arange(w.shape[0], device=w.device), cy_idx, cx_idx] > 0
            centroid_vals = dense_b[:, cy_idx, cx_idx].permute(1, 0)  # (n_b, C)
            mean_vals = torch.einsum("chw,nhw->nc", dense_b, w) / area.unsqueeze(-1)
            return torch.where(on_mask.unsqueeze(-1), centroid_vals, mean_vals)
        return _per_image_pool(
            dense, instance_masks, batch_index, reducer=reducer
        )
