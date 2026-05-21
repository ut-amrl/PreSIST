"""Training-task abstraction.

A ``TrainingTask`` owns a (model, loss_fn) pair and exposes a single
``compute_loss(batch)`` entry point. The trainer becomes task-agnostic: it
moves batches to the device and asks the task for the loss, leaving the
forward shape and loss-call wiring inside the task.

Two concrete implementations are anticipated:

- :class:`PerMaskPersistenceTask` (this module) — wraps the existing
  per-instance models (e.g. ``PersistenceObjConditionedCrossAttn``) and
  the existing ``WeightedMultiLoss`` family.
- ``DensePersistenceTask`` (added later) — wraps the dense per-pixel
  predictor and a ``DenseLoss``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


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
