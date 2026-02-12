from __future__ import annotations

import json
import os
import time
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from utils.metrics import AverageMeter
from training.validator import SurvivalValidationRunner

from training.checkpointing import CheckpointMeta, make_checkpoint_dict, save_checkpoint, load_training_state_from_checkpoint

# ----------------------------
# JSONL Logger (fallback when W&B isn't used)
# ----------------------------
class JsonlLogger:
    """Append-only JSONL logger."""
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def log(self, payload: Dict[str, Any], step: Optional[int] = None):
        record = dict(payload)
        if step is not None:
            record["step"] = int(step)
        record["_ts"] = time.time()
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


# ----------------------------
# Utilities
# ----------------------------
def _maybe_import_wandb():
    try:
        import wandb  # type: ignore
        return wandb
    except Exception:
        return None


def _to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


def _unpack_batch(batch: Union[Dict[str, Any], Tuple[Any, ...]]) -> Dict[str, Any]:
    """
    Supports:
      - dict with keys: image, mask, y50, yr, yc
      - tuple/list: (image, mask, y50, yr, yc, *extras)
    """
    if isinstance(batch, dict):
        return batch
    if isinstance(batch, (list, tuple)) and len(batch) >= 5:
        return {"image": batch[0], "mask": batch[1], "y50": batch[2], "yr": batch[3], "yc": batch[4]}
    raise ValueError(
        "Unsupported batch format. Expected dict with keys "
        "('image','mask','y50','yr','yc') or tuple (image, mask, y50, yr, yc, ...)."
    )


def _estimate_epoch_total_samples(loader) -> Optional[int]:
    """
    Best-effort estimate of total samples in the epoch.
    Priority:
      1) len(loader.dataset)
      2) len(loader.sampler)
      3) len(loader) * loader.batch_size (approx)
    """
    ds = getattr(loader, "dataset", None)
    if ds is not None:
        try:
            return int(len(ds))
        except Exception:
            pass

    sampler = getattr(loader, "sampler", None)
    if sampler is not None:
        try:
            return int(len(sampler))
        except Exception:
            pass

    try:
        num_batches = int(len(loader))
        bs = getattr(loader, "batch_size", None)
        if bs is not None:
            return int(num_batches * int(bs))
    except Exception:
        pass

    return None


def _as_float(v: Any) -> float:
    if torch.is_tensor(v):
        return float(v.detach().cpu().item())
    return float(v)


# ----------------------------
# Config
# ----------------------------
@dataclass
class TrainerConfig:
    max_epochs: int = 100

    # Batch print frequency (stdout only; no W&B logging here)
    log_every_n_batches: int = 50  # N1

    # Checkpoint every N2 epochs
    save_every_n_epochs: int = 1   # N2

    early_stopping_patience: int = 10  # Early stopping based on aggregate score plateau for this many epochs

    # AMP / grad clipping
    use_amp: bool = False
    grad_clip_norm: Optional[float] = 1.0

    # Toggle W&B only (setup passed separately)
    use_wandb: bool = True

    # Device
    device: Optional[Union[str, torch.device]] = None

    batch_size: int = 8
    # Wandb entity
    wandb_entity: Optional[str] = None
    # Wandb project name
    wandb_project: Optional[str] = None


# ----------------------------
# Trainer
# ----------------------------
class Trainer:
    """
    Assumptions:
      - model(image, mask) -> (y50_hat, yr_hat, yc_hat)
      - train_loader & val_loader yield (image, mask, y50, yr, yc) as dict or tuple
      - loss_fn returns: (lossVal, lossMetricsDict)
          lossMetricsDict includes:
            {"loss_50": ..., "loss_r": ..., "loss_c": ...}
          (values can be floats or tensors)
      - validator returns: (aggregate_score, metrics_dict), where aggregate_score is higher-is-better.

    Requirements implemented:
      - Epoch-based validation and epoch-based "high level" metric logging (W&B step=epoch)
      - Per-N1-batch prints to stdout with tabs, including:
          N3/N4, BT, DT, component losses, total loss (val and running avg)
      - Save ckpt_epoch_<num>.pth every N2 epochs
      - Save ckpt_best.pth when aggregate_score improves
      - If epoch == 0 at start, save weights before training
      - Log best epoch number
      - Save/restore optimizer + scheduler (+ AMP scaler) in checkpoints
      - Resume-from-checkpoint helper
      - If W&B disabled/unavailable -> JSONL logs written to output_dir/metrics.jsonl
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        validator: SurvivalValidationRunner,
        output_dir: str,
        start_epoch: int = 0,
        cfg: Optional[TrainerConfig] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.validator = validator

        self.start_epoch = int(start_epoch)
        self.cfg = cfg or TrainerConfig()
        self.early_stopping_patience = int(self.cfg.early_stopping_patience)

        self.device = (
            torch.device(self.cfg.device)
            if self.cfg.device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # AMP helpers
        self._scaler = torch.cuda.amp.GradScaler(enabled=(self.cfg.use_amp and self.device.type == "cuda"))
        self._autocast_ctx = torch.cuda.amp.autocast if self.device.type == "cuda" else torch.cpu.amp.autocast

        # Best tracking (aggregate_score higher is better)
        self.best_score: float = float("-inf")
        self.best_epoch: int = self.start_epoch
        self.epochs_since_improve: int = 0

        # We keep this for checkpoint state; we do NOT use it as the W&B step.
        self.global_step: int = 0

        # W&B + JSONL
        self.wandb = _maybe_import_wandb() if self.cfg.use_wandb else None
        self._wandb_run = None
        self.jsonl_logger: Optional[JsonlLogger] = None

    # ----------------------------
    # Unified metrics logger (category + fallback)
    # ----------------------------
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        *,
        category: Optional[str] = None,
        step: Optional[int] = None,
        allow_wandb: bool = True,
    ) -> None:
        """
        Logs metrics to W&B (if enabled/available and allow_wandb=True),
        otherwise logs to JSONL in output_dir/metrics.jsonl.

        Keys are prefixed as "<category>/<key>" when category is provided.
        For your use-case: pass step=epoch for epoch-based logging.
        """
        if not isinstance(metrics, dict) or len(metrics) == 0:
            return

        payload: Dict[str, float] = {}
        for k, v in metrics.items():
            key = f"{category}/{k}" if category else str(k)
            payload[key] = _as_float(v)

        # W&B path
        if allow_wandb and self.wandb is not None and self._wandb_run is not None:
            self.wandb.log(payload, step=step)  # type: ignore
            return

        # JSONL fallback
        if self.jsonl_logger is None:
            self.jsonl_logger = JsonlLogger(os.path.join(self.output_dir, "metrics.jsonl"))
        record = dict(payload)
        if category is not None:
            record["_category"] = category
        self.jsonl_logger.log(record, step=step)


    def save_checkpoint(self, epoch: int, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)

        meta = CheckpointMeta(
            best_score=self.best_score,
            best_epoch=self.best_epoch,
            epochs_since_improve=self.epochs_since_improve,
            trainer_cfg=vars(self.cfg),
        )

        state = make_checkpoint_dict(
            epoch=epoch,
            global_step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            scaler=self._scaler if (self._scaler is not None and self._scaler.is_enabled()) else None,
            meta=meta,
        )
        save_checkpoint(state, path)

        print(f"[CKPT] Saved: {path}")
        self.log_metrics({"saved_path": path, "epoch": epoch}, category="ckpt", step=epoch, allow_wandb=True)
        return path


    # # ----------------------------
    # # Checkpointing
    # # ----------------------------
    # def save_checkpoint(self, epoch: int, filename: str) -> str:
    #     path = os.path.join(self.output_dir, filename)
    #     state = {
    #         "epoch": int(epoch),
    #         "global_step": int(self.global_step),
    #         "model_state_dict": self.model.state_dict(),
    #         "optimizer_state_dict": self.optimizer.state_dict(),
    #         "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
    #         "scaler_state_dict": self._scaler.state_dict() if self._scaler is not None else None,
    #         "best_score": float(self.best_score),
    #         "best_epoch": int(self.best_epoch),
    #         "epochs_since_improve": int(self.epochs_since_improve),
    #         "trainer_cfg": vars(self.cfg),
    #     }
    #     torch.save(state, path)
    #     print(f"[CKPT] Saved: {path}")
    #     self.log_metrics({"saved_path": path, "epoch": epoch}, category="ckpt", step=epoch, allow_wandb=True)
    #     return path

    def save_best_checkpoint(self, epoch: int) -> str:
        # Save "best" with the *current* epoch checkpoint contents
        return self.save_checkpoint(epoch=epoch, filename="ckpt_best.pth")

    # def resume_from_checkpoint(
    #     self,
    #     ckpt_path: str,
    #     map_location: Optional[Union[str, torch.device]] = None,
    #     strict_model: bool = False,
    # ) -> int:
    #     """
    #     Loads model/optimizer/scheduler/scaler + trainer state.
    #     Returns next epoch index (loaded_epoch + 1).
    #     """
    #     if map_location is None:
    #         map_location = self.device

    #     ckpt = torch.load(ckpt_path, map_location=map_location)

    #     self.model.load_state_dict(ckpt["model_state_dict"], strict=strict_model)

    #     opt_state = ckpt.get("optimizer_state_dict", None)
    #     if opt_state is not None:
    #         self.optimizer.load_state_dict(opt_state)

    #     if self.lr_scheduler is not None:
    #         sched_state = ckpt.get("lr_scheduler_state_dict", None)
    #         if sched_state is not None:
    #             self.lr_scheduler.load_state_dict(sched_state)

    #     scaler_state = ckpt.get("scaler_state_dict", None)
    #     if scaler_state is not None and self._scaler is not None:
    #         try:
    #             self._scaler.load_state_dict(scaler_state)
    #         except Exception:
    #             pass

    #     self.best_score = float(ckpt.get("best_score", float("-inf")))
    #     self.best_epoch = int(ckpt.get("best_epoch", 0))
    #     self.epochs_since_improve = int(ckpt.get("epochs_since_improve", 0))
    #     self.global_step = int(ckpt.get("global_step", 0))

    #     loaded_epoch = int(ckpt.get("epoch", 0))
    #     next_epoch = loaded_epoch + 1
    #     print(f"[CKPT] Resumed: {ckpt_path} (epoch={loaded_epoch}) -> next_epoch={next_epoch}")
    #     return next_epoch



    def resume_from_checkpoint(self, ckpt_path: str, map_location=None, strict_model: bool = False) -> int:
        if map_location is None:
            map_location = self.device

        meta = load_training_state_from_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            scaler=self._scaler,
            ckpt=ckpt_path,
            map_location=map_location,
            strict_model=strict_model,
            load_optimizer=True,
            load_scheduler=True,
            load_scaler=True,
        )

        self.best_score = meta.best_score
        self.best_epoch = meta.best_epoch
        self.epochs_since_improve = meta.epochs_since_improve
        self.global_step = meta.global_step

        next_epoch = meta.epoch + 1
        print(f"[CKPT] Resumed: {ckpt_path} (epoch={meta.epoch}) -> next_epoch={next_epoch}")
        return next_epoch



    # ----------------------------
    # Training / validation
    # ----------------------------
    def fit(self) -> Dict[str, Any]:
        self.model = self.model.to(self.device)

        # Init W&B if enabled+available; else JSONL fallback
        if self.wandb is not None:
            # Require an externally initialized run
            if self.wandb.run is None:
                raise RuntimeError(
                    "use_wandb=True, but no active wandb run found. "
                    "Call wandb.init() before creating the Trainer."
                )
            self._wandb_run = self.wandb.run
            print("[W&B] Using externally initialized wandb run.")
        else:
            self.jsonl_logger = self.jsonl_logger or JsonlLogger(
                os.path.join(self.output_dir, "metrics.jsonl")
            )
            print(f"[LOG] W&B disabled/unavailable; writing JSONL logs to {self.jsonl_logger.path}")

        # If epoch==0 start, save weights before training
        if self.start_epoch == 0:
            self.save_checkpoint(epoch=0, filename="ckpt_epoch_0_pretrain.pth")

        for epoch in range(self.start_epoch, self.cfg.max_epochs):
            train_loss_avg, train_loss_dict_avgs = self._train_one_epoch(epoch)

            # Epoch-based train logging (step=epoch)
            self.log_metrics({"loss": train_loss_avg, **train_loss_dict_avgs}, category="train", step=epoch, allow_wandb=True)

            # Validate (epoch-based)
            val_score, val_metrics = self._validate(epoch)

            # Epoch-based validation logging (step=epoch)
            self.log_metrics({**val_metrics, "aggregate_score": val_score}, category="validation", step=epoch, allow_wandb=True)

            # Step LR scheduler per epoch
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # higher-is-better metric => pass negative
                    self.lr_scheduler.step(-float(val_score))
                else:
                    self.lr_scheduler.step()

            # Save periodic checkpoint every N2 epochs
            if (epoch % self.cfg.save_every_n_epochs) == 0:
                self.save_checkpoint(epoch=epoch, filename=f"ckpt_epoch_{epoch}.pth")

            # Best tracking by aggregate_score
            improved = float(val_score) > float(self.best_score)
            if improved:
                self.best_score = float(val_score)
                self.best_epoch = int(epoch)
                self.epochs_since_improve = 0
                self.save_best_checkpoint(epoch=epoch)
            else:
                self.epochs_since_improve += 1

            # Log best epoch number (epoch-based)
            self.log_metrics(
                {
                    "epoch": self.best_epoch,
                    "aggregate_score": self.best_score,
                    "epochs_since_improve": self.epochs_since_improve,
                },
                category="best",
                step=epoch,
                allow_wandb=True,
            )

            # Console summary
            print(
                f"[EpochSummary]\tEpoch {epoch}\t"
                f"train_loss={train_loss_avg:.6f}\t"
                f"val_agg={val_score:.6f}\t"
                f"best_epoch={self.best_epoch}\tbest_agg={self.best_score:.6f}"
            )

            # Early stopping (based on aggregate score plateau)
            if self.epochs_since_improve >= self.early_stopping_patience:
                print(
                    f"[EarlyStopping] No improvement in aggregate score for "
                    f"{self.early_stopping_patience} epochs. Stopping at epoch {epoch}. "
                    f"Best epoch: {self.best_epoch} (best agg={self.best_score:.6f})."
                )
                break

        # Cleanup
        if self.wandb is not None and self._wandb_run is not None:
            try:
                self.wandb.finish()  # type: ignore
            except Exception:
                pass
        if self.jsonl_logger is not None:
            self.jsonl_logger.close()

        return {
            "best_epoch": self.best_epoch,
            "best_aggregate_score": self.best_score,
            "ckpt_best_path": os.path.join(self.output_dir, "ckpt_best.pth"),
            "metrics_jsonl_path": os.path.join(self.output_dir, "metrics.jsonl"),
        }

    def _train_one_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.train()

        # time meters (batch-weighted)
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # loss meters (sample-weighted)
        loss_meter = AverageMeter()
        loss50_meter = AverageMeter()
        lossr_meter = AverageMeter()
        lossc_meter = AverageMeter()

        # dynamic meters for all entries in lossMetricsDict
        metric_meters: Dict[str, AverageMeter] = {}

        use_amp = bool(self.cfg.use_amp and self.device.type == "cuda")

        epoch_total = _estimate_epoch_total_samples(self.train_loader)
        samples_done = 0

        end = time.perf_counter()

        for idx, batch in enumerate(self.train_loader):
            # Data loading time: time since last loop end
            dt = time.perf_counter() - end

            b = _unpack_batch(batch)
            b = _to_device(b, self.device)

            image = b["image"]
            mask = b["mask"]
            y50 = b["y50"]
            yr = b["yr"]
            yc = b["yc"]

            bs = int(image.shape[0])

            data_time.update(dt, n=1)

            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast_ctx(enabled=use_amp):
                y50_hat, yr_hat, yc_hat = self.model(image, mask)

                out = self.loss_fn(y50_hat, y50, yr_hat, yr, yc_hat, yc)
                if not (isinstance(out, (tuple, list)) and len(out) == 3):
                    raise ValueError("loss_fn must return (lossVal, lossMetricsDict, debugInfo).")
                lossVal, lossMetricsDict, _ = out
                if not torch.is_tensor(lossVal):
                    raise ValueError("lossVal must be a scalar torch.Tensor.")

            # backward + step
            if self._scaler.is_enabled():
                self._scaler.scale(lossVal).backward()
                if self.cfg.grad_clip_norm is not None:
                    self._scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                lossVal.backward()
                if self.cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                self.optimizer.step()

            # Batch time includes compute + step (and some python overhead)
            bt = time.perf_counter() - end
            batch_time.update(bt, n=1)

            # Update loss meters (sample-weighted)
            loss_item = float(lossVal.detach().cpu().item())
            loss_meter.update(loss_item, n=bs)

            # Update component meters from lossMetricsDict
            if isinstance(lossMetricsDict, dict):
                for k, v in lossMetricsDict.items():
                    v_f = _as_float(v)
                    if k not in metric_meters:
                        metric_meters[k] = AverageMeter()
                    metric_meters[k].update(v_f, n=bs)

                # Optional: use named meters for print formatting if keys exist
                if "loss_50" in lossMetricsDict:
                    loss50_meter.update(_as_float(lossMetricsDict["loss_50"]), n=bs)
                if "loss_r" in lossMetricsDict:
                    lossr_meter.update(_as_float(lossMetricsDict["loss_r"]), n=bs)
                if "loss_c" in lossMetricsDict:
                    lossc_meter.update(_as_float(lossMetricsDict["loss_c"]), n=bs)

            # Progress N3/N4
            samples_done += bs
            N3 = samples_done
            N4 = epoch_total if epoch_total is not None else -1
            n4_str = str(N4) if N4 >= 0 else "?"

            # Print every N1 batches (stdout only; W&B captures logs text)
            if (idx % self.cfg.log_every_n_batches) == 0:
                print(
                    f"Epoch: [{epoch}][{N3}/{n4_str}]\t"
                    f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"L50 {loss50_meter.val:.4f} ({loss50_meter.avg:.4f})\t"
                    f"Lr {lossr_meter.val:.4f} ({lossr_meter.avg:.4f})\t"
                    f"Lc {lossc_meter.val:.4f} ({lossc_meter.avg:.4f})\t"
                    f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})"
                )

            self.global_step += 1
            end = time.perf_counter()

        # Return epoch averages
        # Total loss avg:
        train_loss_avg = float(loss_meter.avg)

        # Per-component averages: take averages from metric_meters (ensures you log any extra metrics too)
        train_loss_dict_avgs = {k: float(m.avg) for k, m in metric_meters.items()}

        # You may also want time averages at epoch level; include them under train/ by returning them:
        train_loss_dict_avgs.setdefault("batch_time_avg", float(batch_time.avg))
        train_loss_dict_avgs.setdefault("data_time_avg", float(data_time.avg))

        return train_loss_avg, train_loss_dict_avgs

    @torch.no_grad()
    def _validate(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        agg_score, metrics = self.validator.validate(self.model, self.val_loader)

        # normalize validation metrics dict to floats
        val_metrics: Dict[str, float] = {}
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                val_metrics[k] = _as_float(v)

        print(f"[Val]\tEpoch {epoch}\taggregate_score={float(agg_score):.6f}")

        return float(agg_score), val_metrics

