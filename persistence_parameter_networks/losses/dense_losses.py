"""Dense (per-pixel) loss family for the dense persistence predictor.

All dense losses share a uniform signature so the trainer / task layer
doesn't branch on which specific variant is in use::

    loss(
        pred_map,        # (B, 3, H, W)
        target_map,      # (B, 3, H, W)
        loss_mask,       # (B, 1, H, W) bool / uint8
        pixel_weights,   # (B, 1, H, W) float
        instance_masks,  # (Σ N_i, H, W) or None
        batch_index,     # (Σ N_i,)    or None
        instance_labels, # (Σ N_i, 3) or None
    ) -> (scalar_loss, metrics_dict)

The per-instance tensors are only consumed by losses that opt in via
``requires_per_instance_tensors()``. The default loss
(:class:`DenseWeightedMultiLoss`) ignores them entirely so the training
hot path is decoupled from per-image instance-count variance.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from persistence_parameter_networks.inference.mask_poolers import (
    MaskedMeanPooler,
    MaskPooler,
)
from persistence_parameter_networks.losses.weighted_multi_loss import (
    WeightedMultiLoss,
)


class DenseLoss(nn.Module, ABC):
    """Abstract base: produce ``(scalar_loss, metrics_dict)`` from the dense
    prediction and rasterized supervision."""

    def requires_per_instance_tensors(self) -> bool:
        return False

    @abstractmethod
    def forward(
        self,
        pred_map: torch.Tensor,
        target_map: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_weights: torch.Tensor,
        instance_masks: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.Tensor] = None,
        instance_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError


def _weighted_reduce(
    per_pixel: torch.Tensor,
    loss_mask: torch.Tensor,
    pixel_weights: torch.Tensor,
) -> torch.Tensor:
    """``sum(mask * weight * per_pixel) / sum(mask * weight).clamp(min=1)``.

    ``per_pixel`` and ``loss_mask``/``pixel_weights`` must broadcast
    against each other; loss_mask is treated as float.
    """
    m = loss_mask.to(per_pixel.dtype)
    w = pixel_weights.to(per_pixel.dtype)
    num = (per_pixel * m * w).sum()
    den = (m * w).sum().clamp(min=1e-12)
    return num / den


class DenseWeightedMultiLoss(DenseLoss):
    """Per-pixel analogue of :class:`WeightedMultiLoss`.

    Components and weights mirror the per-mask version: MSE (or Huber)
    on y50 and yr, MSE / BCE / Huber on the (post-sigmoid by default)
    conf channel. The pixel-wise loss is averaged with ``loss_mask`` and
    ``pixel_weights`` weighting per :func:`_weighted_reduce`.

    Defaults match :class:`WeightedMultiLoss` so swapping in this loss
    shouldn't change loss-magnitude calibration vs. the per-mask runs.
    """

    def __init__(
        self,
        w50: float = 1.0,
        wr: float = 1.0,
        wc: float = 0.5,
        useHuberForLogVals: bool = False,
        delta_50: float = 1.0,
        delta_r: float = 1.0,
        confidence_loss: str = "mse",
        confidence_from_logits: bool = False,
        delta_c: float = 0.25,
        clamp_confidence_probs: bool = True,
        eps: float = 1e-6,
        bg_conf_weight: float = 0.0,
        bg_conf_target: float = 0.0,
    ):
        super().__init__()
        self.w50 = w50
        self.wr = wr
        self.wc = wc
        self.useHuberForLogVals = useHuberForLogVals
        self.delta_50 = delta_50
        self.delta_r = delta_r
        self.confidence_loss = confidence_loss
        self.confidence_from_logits = confidence_from_logits
        self.delta_c = delta_c
        self.clamp_confidence_probs = clamp_confidence_probs
        self.eps = eps
        # Background-only confidence supervision. When >0, every pixel
        # outside loss_mask is supervised against bg_conf_target on the
        # confidence channel only (y50/yr untouched on background — they
        # have no meaningful target there). The bg term is contracted by
        # bg_conf_weight so a small value like 0.01 nudges the model to
        # say "low confidence" on unlabeled regions without overwhelming
        # the primary per-pixel signal.
        self.bg_conf_weight = float(bg_conf_weight)
        self.bg_conf_target = float(bg_conf_target)

        if self.confidence_from_logits and self.confidence_loss != "bce":
            raise ValueError(
                "confidence_from_logits=True is only valid when confidence_loss='bce'."
            )

    def _log_per_pixel(self, pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
        if self.useHuberForLogVals:
            return F.smooth_l1_loss(pred, target, beta=delta, reduction="none")
        return (pred - target).pow(2)

    def _conf_per_pixel(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=pred.dtype)
        if self.confidence_loss == "bce":
            if self.confidence_from_logits:
                return F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            if self.clamp_confidence_probs:
                pred = pred.clamp(self.eps, 1.0 - self.eps)
            return F.binary_cross_entropy(pred, target, reduction="none")
        if self.confidence_loss == "mse":
            return (pred - target).pow(2)
        if self.confidence_loss == "huber":
            return F.smooth_l1_loss(pred, target, beta=self.delta_c, reduction="none")
        raise ValueError(f"Unknown confidence_loss: {self.confidence_loss!r}")

    def forward(
        self,
        pred_map: torch.Tensor,
        target_map: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_weights: torch.Tensor,
        instance_masks: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.Tensor] = None,
        instance_labels: Optional[torch.Tensor] = None,
    ):
        pp_50 = self._log_per_pixel(pred_map[:, 0:1], target_map[:, 0:1], self.delta_50)
        pp_r = self._log_per_pixel(pred_map[:, 1:2], target_map[:, 1:2], self.delta_r)
        pp_c = self._conf_per_pixel(pred_map[:, 2:3], target_map[:, 2:3])

        loss_50 = _weighted_reduce(pp_50, loss_mask, pixel_weights)
        loss_r = _weighted_reduce(pp_r, loss_mask, pixel_weights)
        loss_c = _weighted_reduce(pp_c, loss_mask, pixel_weights)

        full = self.w50 * loss_50 + self.wr * loss_r + self.wc * loss_c
        metrics: Dict[str, torch.Tensor] = {
            "loss_50": loss_50.detach(),
            "loss_r": loss_r.detach(),
            "loss_c": loss_c.detach(),
        }

        if self.bg_conf_weight > 0.0:
            # Background = pixels outside loss_mask. Supervise conf only,
            # target = bg_conf_target (typically 0). Use _conf_per_pixel
            # so the same confidence_loss kind (mse/bce/huber) applies on
            # both sides — bg_loss_c is comparable to loss_c in scale.
            bg_mask = (~loss_mask.to(torch.bool)).to(pred_map.dtype)
            bg_target = torch.full_like(pred_map[:, 2:3], self.bg_conf_target)
            bg_pp_c = self._conf_per_pixel(pred_map[:, 2:3], bg_target)
            # Uniform weight on background — area normalization is
            # handled by averaging; no need to mirror pixel_weights here.
            bg_weight = torch.ones_like(bg_mask)
            bg_loss_c = _weighted_reduce(bg_pp_c, bg_mask, bg_weight)
            full = full + self.bg_conf_weight * bg_loss_c
            metrics["bg_loss_c"] = bg_loss_c.detach()

        metrics["full_loss"] = full.detach()
        return full, metrics


class DenseWeightedMultiLossWithInstanceAux(DenseLoss):
    """Composes :class:`DenseWeightedMultiLoss` with a per-instance pooled
    auxiliary term.

    The auxiliary term pools the dense prediction within each instance
    mask via the supplied :class:`MaskPooler` and applies the original
    :class:`WeightedMultiLoss` to the pooled vector against the
    instance's label. Acts as a calibration anchor that ties the dense
    output's per-instance mean to what the per-mask model would produce.
    """

    def __init__(
        self,
        dense_loss: Optional[DenseWeightedMultiLoss] = None,
        per_instance_loss: Optional[WeightedMultiLoss] = None,
        aux_weight: float = 0.1,
        pooler: Optional[MaskPooler] = None,
    ):
        super().__init__()
        self.dense_loss = dense_loss or DenseWeightedMultiLoss()
        self.per_instance_loss = per_instance_loss or WeightedMultiLoss()
        self.aux_weight = float(aux_weight)
        self.pooler = pooler or MaskedMeanPooler()

    def requires_per_instance_tensors(self) -> bool:
        return True

    def forward(
        self,
        pred_map,
        target_map,
        loss_mask,
        pixel_weights,
        instance_masks=None,
        batch_index=None,
        instance_labels=None,
    ):
        primary, metrics = self.dense_loss(
            pred_map, target_map, loss_mask, pixel_weights
        )
        # If there are no instances in this batch (rare but possible
        # under heavy filtering), skip the aux term.
        if instance_masks is None or instance_masks.numel() == 0:
            metrics["aux_loss"] = pred_map.new_tensor(0.0)
            return primary, metrics

        pooled = self.pooler.pool(pred_map, instance_masks, batch_index)
        # pooled: (N, 3), instance_labels: (N, 3)
        aux_loss, aux_metrics, _ = self.per_instance_loss(
            pooled[:, 0], instance_labels[:, 0],
            pooled[:, 1], instance_labels[:, 1],
            pooled[:, 2], instance_labels[:, 2],
        )
        full = primary + self.aux_weight * aux_loss
        metrics["aux_loss"] = aux_loss.detach()
        for k, v in aux_metrics.items():
            metrics[f"aux_{k}"] = v
        metrics["full_loss"] = full.detach()
        return full, metrics
