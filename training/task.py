"""Training-task abstraction.

A ``TrainingTask`` owns a (model, loss_fn) pair and exposes a single
``compute_loss(batch)`` entry point. The trainer becomes task-agnostic: it
moves batches to the device and asks the task for the loss, leaving the
forward shape and loss-call wiring inside the task.

Two concrete implementations:

- :class:`PerMaskPersistenceTask` wraps the existing per-instance models
  (e.g. ``PersistenceObjConditionedCrossAttn``) and the existing
  :class:`WeightedMultiLoss` family.
- :class:`DensePersistenceTask` wraps a
  :class:`DensePersistencePredictor` and a :class:`DenseLoss`.

Both expose ``make_validation_model() -> nn.Module`` which returns
something that exposes ``model(image, mask) -> (y50_hat, yr_hat, yc_hat)``
so the existing :class:`SurvivalValidationRunner` can validate any task
without branching. For the dense task the returned module is a thin
facade that runs ``predict_dense`` and pools inside the mask via a
:class:`MaskPooler`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from persistence_parameter_networks.inference.mask_poolers import (
    MaskedMeanPooler,
    MaskPooler,
)


class TrainingTask(ABC):
    """Abstract base: owns ``(model, loss_fn)`` and produces a scalar loss
    + metrics dict from a single batch."""

    def __init__(self, model: nn.Module, loss_fn: nn.Module):
        self.model = model
        self.loss_fn = loss_fn

    @abstractmethod
    def compute_loss(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Run the model on the batch and apply the loss.

        Args:
            batch: dict with task-specific keys, already on the correct device.

        Returns:
            (scalar_loss_tensor, metrics_dict)
        """
        raise NotImplementedError

    def make_validation_model(self) -> nn.Module:
        """Return an ``nn.Module`` whose ``forward(image, mask)`` produces
        the per-mask ``(y50_hat, yr_hat, yc_hat)`` tuple the validator
        expects. The default returns ``self.model`` directly (per-mask
        tasks). The dense task overrides this with a pooling facade."""
        return self.model


class PerMaskPersistenceTask(TrainingTask):
    """Wraps a per-instance persistence model + the matching weighted-multi loss.

    Reproduces exactly the call shape that ``training/trainer.py`` previously
    inlined::

        y50_hat, yr_hat, yc_hat = model(image, mask)
        loss, metrics, _ = loss_fn(y50_hat, y50, yr_hat, yr, yc_hat, yc)

    Batch keys: ``image``, ``mask``, ``y50``, ``yr``, ``yc``.
    """

    def compute_loss(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        image = batch["image"]
        mask = batch["mask"]
        y50 = batch["y50"]
        yr = batch["yr"]
        yc = batch["yc"]

        y50_hat, yr_hat, yc_hat = self.model(image, mask)

        out = self.loss_fn(y50_hat, y50, yr_hat, yr, yc_hat, yc)
        if not (isinstance(out, (tuple, list)) and len(out) == 3):
            raise ValueError(
                "loss_fn must return (lossVal, lossMetricsDict, debugInfo)."
            )
        loss_val, metrics, _ = out
        if not torch.is_tensor(loss_val):
            raise ValueError("lossVal must be a scalar torch.Tensor.")
        return loss_val, (metrics if isinstance(metrics, dict) else {})


class _DensePerMaskFacade(nn.Module):
    """Adapt a :class:`DensePersistencePredictor` to the per-mask
    ``forward(image, mask) -> (y50_hat, yr_hat, yc_hat)`` interface the
    existing validator expects. Pools the dense prediction within each
    instance mask using the supplied :class:`MaskPooler`.

    The wrapped dense model retains its own ``predict_dense`` API, so
    the dense training path remains unaffected; this facade is only
    used to drive the existing per-instance validator without
    refactoring it.
    """

    def __init__(self, dense_model: nn.Module, pooler: MaskPooler):
        super().__init__()
        self.dense_model = dense_model
        self.pooler = pooler

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        # The per-mask validator yields one (image, mask) pair per row,
        # with mask of shape (B, 1, H, W). The MaskPooler expects
        # (Σ N_i, H, W) flat masks plus a batch_index; here Σ N_i == B
        # and batch_index = [0, 1, ..., B-1].
        if mask.dim() == 4 and mask.shape[1] == 1:
            instance_masks = mask.squeeze(1)
        elif mask.dim() == 3:
            instance_masks = mask
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(mask.shape)}; expected "
                "(B, 1, H, W) or (B, H, W)."
            )
        B = image.shape[0]
        batch_index = torch.arange(B, device=image.device, dtype=torch.long)
        dense = self.dense_model.predict_dense(image)        # (B, 3, H, W)
        pooled = self.pooler.pool(dense, instance_masks, batch_index)  # (B, 3)
        return pooled[:, 0], pooled[:, 1], pooled[:, 2]


class DensePersistenceTask(TrainingTask):
    """Wraps a :class:`DensePersistencePredictor` + a :class:`DenseLoss`.

    Training path: ``predict_dense`` + dense loss on the rasterized
    supervision. No per-instance pooling on the training hot path
    unless the loss explicitly opts in via
    ``requires_per_instance_tensors()``.

    Validation path: :meth:`make_validation_model` returns a facade that
    pools the dense prediction inside each per-instance mask, so the
    existing validator can score the dense model on its per-instance
    val loader.

    Batch keys (training): ``image``, ``target_map``, ``loss_mask``,
    ``pixel_weights``; optionally ``instance_masks``, ``batch_index``,
    ``instance_labels`` (only when the configured loss needs them).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        validation_pooler: Optional[MaskPooler] = None,
    ):
        super().__init__(model=model, loss_fn=loss_fn)
        self.validation_pooler = validation_pooler or MaskedMeanPooler()

    def compute_loss(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        image = batch["image"]
        target_map = batch["target_map"]
        loss_mask = batch["loss_mask"]
        pixel_weights = batch["pixel_weights"]
        pred_map = self.model.predict_dense(image)

        # The loss-side per-instance tensors only travel in the batch
        # when the configured loss asks for them.
        loss_val, metrics = self.loss_fn(
            pred_map=pred_map,
            target_map=target_map,
            loss_mask=loss_mask,
            pixel_weights=pixel_weights,
            instance_masks=batch.get("instance_masks"),
            batch_index=batch.get("batch_index"),
            instance_labels=batch.get("instance_labels"),
        )
        if not torch.is_tensor(loss_val):
            raise ValueError("DenseLoss must return a scalar torch.Tensor.")
        return loss_val, (metrics if isinstance(metrics, dict) else {})

    def make_validation_model(self) -> nn.Module:
        return _DensePerMaskFacade(self.model, self.validation_pooler)
