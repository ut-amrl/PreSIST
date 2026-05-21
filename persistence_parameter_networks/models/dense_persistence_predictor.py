"""Abstract base class for dense (per-pixel) persistence predictors.

The contract is intentionally narrow: ``predict_dense(image) -> (B, 3, H, W)``
where the 3 channels are ``[log p25, log(p75/p25 - 1), conf]``. Whether the
``conf`` channel is logits or post-sigmoid is governed by
``confidenceFromLogits``, mirroring the per-mask :class:`PersistenceOutputPredictor`.

Concrete implementations (e.g. DPT, SETR-PUP, U-Net) provide the decoder.
Cross-model contracts like per-instance predictions live in an external
adapter layer (added in Stage 5), not on this class — the dense predictor
itself only knows how to produce dense maps.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class DensePersistencePredictor(nn.Module, ABC):
    """Image -> (B, 3, H, W) per-pixel persistence prediction."""

    def __init__(
        self,
        name: str,
        confidence_from_logits: bool = False,
    ):
        super().__init__()
        self.model_name = name
        self.confidence_from_logits = confidence_from_logits

    @abstractmethod
    def predict_dense(self, image: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 3, H, W)`` with sigmoid already applied to the conf
        channel when ``confidence_from_logits`` is False."""
        raise NotImplementedError

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Provide a default forward so the predictor can be used like any
        # other nn.Module. Subclasses can override if they need to.
        return self.predict_dense(image)
