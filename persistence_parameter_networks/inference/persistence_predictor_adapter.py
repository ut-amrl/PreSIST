"""External adapter that provides a uniform per-mask inference contract.

Background: per-mask models (e.g. ``PersistenceObjConditionedCrossAttn``)
take ``(image, mask)`` and return ``(y50_hat, yr_hat, yc_hat)`` per
instance. Dense models take only ``image`` and produce a ``(B, 3, H, W)``
map. To compare across architectures and to drive downstream inference
uniformly, we wrap any model in an adapter that exposes:

    predict_persistence(image, instance_masks, batch_index) -> (Σ N_i, 3)

The adapter is the only place per-mask vs dense distinction lives. Models
themselves are untouched — for the existing per-mask predictor we don't
even subclass, we just iterate over flat mask rows and call its native
forward.

The training path does NOT go through this adapter. The dense task's
loss runs on the dense map directly; the per-mask task runs the per-mask
forward directly. Adapters are for validation and inference only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import torch
import torch.nn as nn

from persistence_parameter_networks.inference.mask_poolers import (
    MaskedMeanPooler,
    MaskPooler,
)


class PersistencePredictorAdapter(nn.Module, ABC):
    """Abstract: image + flat instance masks -> per-instance 3-vectors."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def predict_persistence(
        self,
        image: torch.Tensor,            # (B, 3, H, W)
        instance_masks: torch.Tensor,   # (Σ N_i, H, W)
        batch_index: torch.Tensor,      # (Σ N_i,)
    ) -> torch.Tensor:                  # (Σ N_i, 3)
        raise NotImplementedError


class PerMaskAdapter(PersistencePredictorAdapter):
    """Wraps a per-mask model whose ``forward(image, mask) -> (y50, yr, yc)``
    handles a single instance per image. Calls the model once per
    instance, picking the right image via ``batch_index``."""

    def predict_persistence(self, image, instance_masks, batch_index):
        # The legacy per-mask path expects (B', 3, H, W) images and
        # (B', 1, H, W) masks; for each instance we gather the right
        # image and the matching mask, stack, and forward.
        N = int(instance_masks.shape[0])
        if N == 0:
            return image.new_zeros((0, 3))
        # Group by image to amortize backbone forward across all
        # instances belonging to one image. This is what the existing
        # per-mask training also relies on.
        out = image.new_zeros((N, 3))
        for b in torch.unique(batch_index).tolist():
            sel = (batch_index == b).nonzero(as_tuple=False).view(-1)
            n_b = int(sel.numel())
            img_b = image[b : b + 1].expand(n_b, -1, -1, -1)   # (n_b, 3, H, W)
            masks_b = instance_masks[sel].unsqueeze(1).to(img_b.dtype)  # (n_b, 1, H, W)
            y50, yr, yc = self.model(img_b, masks_b)
            out[sel, 0] = y50.view(-1)
            out[sel, 1] = yr.view(-1)
            out[sel, 2] = yc.view(-1)
        return out


class DenseAdapter(PersistencePredictorAdapter):
    """Wraps a :class:`DensePersistencePredictor`. Runs ``predict_dense``
    once over the batch of images and pools the dense map within each
    instance mask via the configured :class:`MaskPooler`."""

    def __init__(self, model: nn.Module, pooler: MaskPooler | None = None):
        super().__init__(model)
        self.pooler = pooler or MaskedMeanPooler()

    def predict_persistence(self, image, instance_masks, batch_index):
        dense = self.model.predict_dense(image)             # (B, 3, H, W)
        return self.pooler.pool(dense, instance_masks, batch_index)


def make_adapter(
    model: nn.Module,
    *,
    pooler: MaskPooler | None = None,
) -> PersistencePredictorAdapter:
    """Dispatch on model type. Adds new dense subclasses without code
    changes here (we check via isinstance on the dense ABC)."""
    # Lazy import to avoid a top-level circular dep with the model
    # module that imports from `inference/` later.
    from persistence_parameter_networks.models.dense_persistence_predictor import (
        DensePersistencePredictor,
    )

    if isinstance(model, DensePersistencePredictor):
        return DenseAdapter(model, pooler=pooler)
    return PerMaskAdapter(model)
