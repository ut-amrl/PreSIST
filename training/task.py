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

Both expose ``make_validation_model() -> MaskedValidationModel`` which
returns a module supporting two call shapes:

- ``forward(image, mask) -> (y50_hat, yr_hat, yc_hat)`` — used by
  :class:`SurvivalValidationRunner` (one mask per image per call).
- ``forward_multi(image, masks) -> (K, 3)`` — used by the inferencer
  (one image with K masks; the dense facade calls ``predict_dense``
  once and pools K times).

For the dense task the returned model pools inside each mask via a
:class:`MaskPooler`. The per-mask-vs-dense dispatch lives in the free
function :func:`build_validation_model`, which the inferencer also
calls (it has no loss with which to construct a full task).
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

    @abstractmethod
    def make_validation_model(self) -> "MaskedValidationModel":
        """Return a :class:`MaskedValidationModel` exposing both the
        single-mask ``forward(image, mask) -> (y50, yr, yc)`` contract the
        per-instance validator uses today and the multi-mask
        ``forward_multi(image, masks) -> (K, 3)`` contract the inferencer
        uses (one forward per image, K mask outputs).
        """
        raise NotImplementedError


class MaskedValidationModel(nn.Module, ABC):
    """Architecture-agnostic per-mask prediction model.

    Wraps either a per-mask model (identity-like) or a dense model
    (predict_dense + pool). Exposes two call shapes:

    - ``forward(image, mask)`` — one image with one (broadcast) mask,
      same contract :class:`SurvivalValidationRunner` already calls.
    - ``forward_multi(image, masks)`` — one image with K masks, used by
      the inferencer to amortize the backbone over masks. ``image`` may
      be ``(C, H, W)`` or ``(1, C, H, W)``; ``masks`` is ``(K, H, W)``.
      Returns ``(K, 3)`` with columns ``[y50, yr, yc]``.
    """

    @abstractmethod
    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def forward_multi(self, image: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PerMaskValidationModel(MaskedValidationModel):
    """Wraps a per-mask model (``forward(image, mask) -> 3-tuple``) in
    the unified :class:`MaskedValidationModel` interface. ``forward``
    delegates straight through; ``forward_multi`` loops K times and
    stacks the outputs.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        return self.model(image, mask)

    def forward_multi(self, image: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        outs = []
        for k in range(masks.shape[0]):
            m = masks[k].unsqueeze(0).unsqueeze(0).to(image.dtype)  # (1, 1, H, W)
            y50, yr, yc = self.model(image, m)
            outs.append(torch.stack([y50.reshape(()), yr.reshape(()), yc.reshape(())]))
        return torch.stack(outs, dim=0)


class DensePerMaskValidationModel(MaskedValidationModel):
    """Adapt a :class:`DensePersistencePredictor` to the per-mask
    interface. Pools the dense prediction within each instance mask
    using the supplied :class:`MaskPooler`.

    The wrapped dense model retains its own ``predict_dense`` API, so
    the dense training path remains unaffected; this facade is used both
    to drive the existing per-instance validator and to serve the
    inferencer's multi-mask call.
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

    def forward_multi(self, image: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        dense = self.dense_model.predict_dense(image)        # (1, 3, H, W)
        batch_index = torch.zeros(masks.shape[0], dtype=torch.long, device=image.device)
        return self.pooler.pool(dense, masks, batch_index)   # (K, 3)


class DenormalizingValidationModel(MaskedValidationModel):
    """Wraps a :class:`MaskedValidationModel` and un-applies the z-score
    normalization that ``DistillationDatasetCfg.y50_mean/std`` /
    ``yr_mean/std`` applied to the training targets, so downstream
    validation metrics and inferencer outputs land in the original
    (log-)space — directly comparable across `_norm` and non-`_norm` runs.

    ``yc`` passes through untouched (it was never normalized).

    Side effect: when this wrapper is in play, ``val_loss`` is computed
    on log-space predictions, so it no longer matches ``train_loss``
    (which sees the normalized targets). This is an explicit tradeoff:
    metric consistency across runs is more important than train/val
    loss-scale parity within a single run.
    """

    def __init__(
        self,
        inner: MaskedValidationModel,
        *,
        y50_mean: float,
        y50_std: float,
        yr_mean: float,
        yr_std: float,
    ):
        super().__init__()
        self.inner = inner
        self.register_buffer("y50_mean", torch.tensor(float(y50_mean)))
        self.register_buffer("y50_std", torch.tensor(float(y50_std)))
        self.register_buffer("yr_mean", torch.tensor(float(yr_mean)))
        self.register_buffer("yr_std", torch.tensor(float(yr_std)))

    def forward(self, image: torch.Tensor, mask: torch.Tensor):
        y50, yr, yc = self.inner(image, mask)
        return (
            y50 * self.y50_std + self.y50_mean,
            yr * self.yr_std + self.yr_mean,
            yc,
        )

    def forward_multi(self, image: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        out = self.inner.forward_multi(image, masks)  # (K, 3) [y50, yr, yc]
        one = torch.ones_like(self.y50_std)
        zero = torch.zeros_like(self.y50_mean)
        scale = torch.stack([self.y50_std, self.yr_std, one])
        shift = torch.stack([self.y50_mean, self.yr_mean, zero])
        return out * scale + shift


def build_validation_model(
    model: nn.Module,
    *,
    pooler: Optional[MaskPooler] = None,
    y50_mean: Optional[float] = None,
    y50_std: Optional[float] = None,
    yr_mean: Optional[float] = None,
    yr_std: Optional[float] = None,
) -> MaskedValidationModel:
    """Per-mask-vs-dense dispatch factory. Used by both the task classes
    and the inferencer (which has no loss to construct a full task with).

    ``pooler`` is only consulted for dense models; defaults to
    :class:`MaskedMeanPooler` when not provided.

    When all four ``(y50_mean, y50_std, yr_mean, yr_std)`` are provided
    (i.e. this checkpoint was trained with normalized targets), the
    returned model is additionally wrapped in a
    :class:`DenormalizingValidationModel` so its outputs are in log-space.
    If any of the four is ``None``, no wrapping is applied — keeping
    non-`_norm` configs unaffected.
    """
    from persistence_parameter_networks.models.dense_persistence_predictor import (
        DensePersistencePredictor,
    )

    if isinstance(model, DensePersistencePredictor):
        inner: MaskedValidationModel = DensePerMaskValidationModel(
            model, pooler or MaskedMeanPooler()
        )
    else:
        inner = PerMaskValidationModel(model)

    stats = (y50_mean, y50_std, yr_mean, yr_std)
    if all(s is not None for s in stats):
        return DenormalizingValidationModel(
            inner,
            y50_mean=y50_mean,
            y50_std=y50_std,
            yr_mean=yr_mean,
            yr_std=yr_std,
        )
    if any(s is not None for s in stats):
        raise ValueError(
            "Target-normalization stats must be provided as a complete set "
            "(y50_mean, y50_std, yr_mean, yr_std) or not at all; got "
            f"{stats}."
        )
    return inner


class PerMaskPersistenceTask(TrainingTask):
    """Wraps a per-instance persistence model + the matching weighted-multi loss.

    Reproduces exactly the call shape that ``training/trainer.py`` previously
    inlined::

        y50_hat, yr_hat, yc_hat = model(image, mask)
        loss, metrics, _ = loss_fn(y50_hat, y50, yr_hat, yr, yc_hat, yc)

    Batch keys: ``image``, ``mask``, ``y50``, ``yr``, ``yc``.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        *,
        y50_mean: Optional[float] = None,
        y50_std: Optional[float] = None,
        yr_mean: Optional[float] = None,
        yr_std: Optional[float] = None,
    ):
        super().__init__(model=model, loss_fn=loss_fn)
        self._y50_mean = y50_mean
        self._y50_std = y50_std
        self._yr_mean = yr_mean
        self._yr_std = yr_std

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

    def make_validation_model(self) -> MaskedValidationModel:
        return build_validation_model(
            self.model,
            y50_mean=self._y50_mean,
            y50_std=self._y50_std,
            yr_mean=self._yr_mean,
            yr_std=self._yr_std,
        )


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
        *,
        y50_mean: Optional[float] = None,
        y50_std: Optional[float] = None,
        yr_mean: Optional[float] = None,
        yr_std: Optional[float] = None,
    ):
        super().__init__(model=model, loss_fn=loss_fn)
        # None → MaskedMeanPooler at validation-model build time. Stored
        # as-is so Hydra-configurable poolers flow through cleanly.
        self.validation_pooler = validation_pooler
        self._y50_mean = y50_mean
        self._y50_std = y50_std
        self._yr_mean = yr_mean
        self._yr_std = yr_std

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

    def make_validation_model(self) -> MaskedValidationModel:
        return build_validation_model(
            self.model,
            pooler=self.validation_pooler,
            y50_mean=self._y50_mean,
            y50_std=self._y50_std,
            yr_mean=self._yr_mean,
            yr_std=self._yr_std,
        )
