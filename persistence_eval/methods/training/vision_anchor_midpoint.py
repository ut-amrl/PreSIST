"""Vision anchor-midpoint trainer.

Implements the `AbsTrainer` protocol so it slots into `TrainingOrchestrator`
alongside the distribution-fit trainers. Trains a vision model
(image + binary mask -> scalar log-time) directly from per-anchor
observations on the annotation-tool manifest, with log-MSE supervision and
held-out-track validation + early stopping.

The orchestrator's serialized output schema (per-bucket params dict) is
still produced — the entry references the saved checkpoint rather than a
closed-form parameter set. Eval-side wiring is handled separately.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from persistence_eval.methods.training.base import (
    ParamEntry,
    TrainingDataset,
)
from persistence_eval.methods.training.vision_anchor_dataset import AnchorMidpointDataset


_GLOBAL_KEY = "__global__"
_MODEL_TARGET = "persistence_parameter_networks.models.vision_scalar_predictor.VisionScalarPredictor"


class VisionAnchorMidpointTrainer:
    grouping: str
    name: str

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        lr_scheduler: Callable[..., torch.optim.lr_scheduler._LRScheduler] | None = None,
        epochs: int = 50,
        batch_size: int = 16,
        num_workers: int = 4,
        early_stopping_patience: int = 10,
        image_size: int = 518,
        grad_clip_norm: float | None = 1.0,
        device: str | None = None,
        log_every_n_batches: int = 20,
        grouping: str = "class",
        name: str = "vision_anchor_midpoint",
        use_wandb: bool = False,
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
    ) -> None:
        self.grouping = grouping
        self.name = name

        self._model = model
        self._optimizer_factory = optimizer
        self._lr_scheduler_factory = lr_scheduler
        self._epochs = int(epochs)
        self._batch_size = int(batch_size)
        self._num_workers = int(num_workers)
        self._early_stopping_patience = int(early_stopping_patience)
        self._image_size = int(image_size)
        self._grad_clip_norm = grad_clip_norm
        self._device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._log_every = int(log_every_n_batches)
        self._use_wandb = bool(use_wandb)
        self._wandb_entity = wandb_entity
        self._wandb_project = wandb_project
        self._wandb_run_name = wandb_run_name

        # Set by the orchestrator after construction.
        self.output_dir: Path | None = None
        # Optional config dict for wandb (set by the orchestrator).
        self.wandb_config: dict | None = None

    def train(self, dataset: TrainingDataset) -> dict[str, ParamEntry]:
        if self.output_dir is None:
            raise RuntimeError(
                "VisionAnchorMidpointTrainer.output_dir must be set by the "
                "orchestrator before train()."
            )
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_tracks = list(dataset.iter_tracks())
        val_tracks = list(dataset.iter_val_tracks())

        train_set = AnchorMidpointDataset(
            train_tracks,
            manifest_dir=dataset.manifest_dir,
            image_size=self._image_size,
        )
        val_set = AnchorMidpointDataset(
            val_tracks,
            manifest_dir=dataset.manifest_dir,
            image_size=self._image_size,
        )

        if len(train_set) == 0:
            raise RuntimeError(
                "No training anchors produced — every track lacks first_missing "
                "or has no positive observations with image+mask."
            )

        print(
            f"[VisionAnchorMidpoint] train tracks={len(train_tracks)} "
            f"anchors={len(train_set)} | val tracks={len(val_tracks)} "
            f"anchors={len(val_set)}"
        )

        train_loader = DataLoader(
            train_set,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=(self._device.type == "cuda"),
            drop_last=False,
        )
        val_loader = (
            DataLoader(
                val_set,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                pin_memory=(self._device.type == "cuda"),
                drop_last=False,
            )
            if len(val_set) > 0
            else None
        )

        model = self._model.to(self._device)
        optimizer = self._optimizer_factory(params=model.parameters())
        scheduler = (
            self._lr_scheduler_factory(optimizer=optimizer)
            if self._lr_scheduler_factory is not None
            else None
        )

        wb = None
        if self._use_wandb:
            try:
                import wandb as _wandb
                _wandb.init(
                    entity=self._wandb_entity,
                    project=self._wandb_project,
                    name=self._wandb_run_name,
                    config=self.wandb_config or {},
                )
                wb = _wandb
            except Exception as e:
                print(f"[VisionAnchorMidpoint] wandb disabled (init failed: {e})")
                wb = None

        best_val = math.inf
        best_epoch = -1
        epochs_since_improve = 0
        best_ckpt_path = out_dir / "best_model.pt"
        last_ckpt_path = out_dir / "last_model.pt"

        history: list[dict[str, float]] = []

        for epoch in range(self._epochs):
            t0 = time.time()
            train_loss = self._run_epoch(model, train_loader, optimizer, train=True, epoch=epoch)
            val_loss = (
                self._run_epoch(model, val_loader, None, train=False, epoch=epoch)
                if val_loader is not None
                else float("nan")
            )
            if scheduler is not None:
                scheduler.step()

            elapsed = time.time() - t0
            print(
                f"[VisionAnchorMidpoint] epoch {epoch} train_log_mse={train_loss:.4f} "
                f"val_log_mse={val_loss:.4f} ({elapsed:.1f}s)"
            )
            history.append({
                "epoch": epoch,
                "train_log_mse": train_loss,
                "val_log_mse": val_loss,
            })
            if wb is not None:
                lr = optimizer.param_groups[0].get("lr", float("nan"))
                wb.log(
                    {
                        "train/log_mse": train_loss,
                        "val/log_mse": val_loss,
                        "train/lr": lr,
                        "epoch_seconds": elapsed,
                    },
                    step=epoch,
                )

            torch.save({"model": model.state_dict(), "epoch": epoch}, last_ckpt_path)

            score_for_best = val_loss if val_loader is not None else train_loss
            if score_for_best < best_val:
                best_val = score_for_best
                best_epoch = epoch
                epochs_since_improve = 0
                torch.save({"model": model.state_dict(), "epoch": epoch}, best_ckpt_path)
                if wb is not None:
                    wb.log(
                        {"best/val_log_mse": best_val, "best/epoch": best_epoch},
                        step=epoch,
                    )
            else:
                epochs_since_improve += 1
                if (
                    val_loader is not None
                    and epochs_since_improve >= self._early_stopping_patience
                ):
                    print(
                        f"[VisionAnchorMidpoint] early-stopping at epoch {epoch} "
                        f"(no val improvement for {self._early_stopping_patience} epochs)"
                    )
                    break

        if not best_ckpt_path.exists():
            torch.save({"model": model.state_dict(), "epoch": best_epoch}, best_ckpt_path)

        with (out_dir / "training_history.json").open("w") as f:
            json.dump(history, f, indent=2)

        if wb is not None:
            try:
                wb.finish()
            except Exception:
                pass

        entry: ParamEntry = {
            "params": {
                "checkpoint_path": str(best_ckpt_path.name),
                "model_target": _MODEL_TARGET,
                "image_size": self._image_size,
            },
            "confidence": None,
            "metadata": {
                "trainer": self.name,
                "n_train_anchors": len(train_set),
                "n_val_anchors": len(val_set),
                "n_train_tracks": len(train_tracks),
                "n_val_tracks": len(val_tracks),
                "best_val_log_mse": (None if math.isinf(best_val) else best_val),
                "best_epoch": best_epoch,
                "epochs_run": len(history),
            },
        }
        # Provide both `__global__` and a class-bucketed entry so the
        # orchestrator's bucket-keyed serialization stays consistent.
        return {_GLOBAL_KEY: entry, "0": entry}

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer,
        *,
        train: bool,
        epoch: int,
    ) -> float:
        if train:
            model.train()
        else:
            model.eval()

        loss_fn = nn.MSELoss()
        total_loss = 0.0
        total_count = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch_idx, batch in enumerate(loader):
                image = batch["image"].to(self._device, non_blocking=True)
                mask = batch["mask"].to(self._device, non_blocking=True)
                target = batch["target"].to(self._device, non_blocking=True)

                pred = model(image, mask)
                if pred.ndim > 1:
                    pred = pred.squeeze(-1)
                loss = loss_fn(pred, target)

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self._grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self._grad_clip_norm
                        )
                    optimizer.step()

                bs = image.size(0)
                total_loss += float(loss.detach().cpu()) * bs
                total_count += bs

                if train and self._log_every > 0 and batch_idx % self._log_every == 0:
                    print(
                        f"  [epoch {epoch}] batch {batch_idx}/{len(loader)} "
                        f"loss={float(loss.detach().cpu()):.4f}"
                    )

        return total_loss / max(total_count, 1)
