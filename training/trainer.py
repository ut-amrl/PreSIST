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
from training.validator import ValConfig
from training.task import TrainingTask

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
    name: str = "default"
    max_epochs: int = 100

    # Batch print frequency (stdout only; no W&B logging here)
    log_every_n_batches: int = 50  # N1

    # Checkpoint every N2 epochs
    save_every_n_epochs: int = 1   # N2

    img_size: int = 518

    early_stopping_patience: int = 10  # Early stopping based on aggregate score plateau for this many epochs

    # AMP / grad clipping
    use_amp: bool = False
    grad_clip_norm: Optional[float] = 1.0

    # Toggle W&B only (setup passed separately)
    use_wandb: bool = True

    # Device
    device: Optional[Union[str, torch.device]] = None

    batch_size: int = 8
    # Validation batch size. Falls back to `batch_size` when unset/None.
    # Val has no grads so it can usually fit a much larger batch than train.
    val_batch_size: Optional[int] = None
    # DataLoader worker counts. Bump these if val/train batches are CPU/IO
    # bound (GPU under-utilized).
    num_workers: int = 4
    val_num_workers: Optional[int] = None
    # Wandb entity
    wandb_entity: Optional[str] = None
    # Wandb project name
    wandb_project: Optional[str] = None

    # When true, the train DataLoader uses a WeightedRandomSampler with
    # weight = 1/n_instances_in_image, so each image contributes equal
    # expected probability regardless of how many instances it contains.
    # Off by default — existing configs are unaffected.
    image_balanced_sampling: bool = False

    # Cap the number of batches processed when running the validator
    # against the *training* loader (the per-epoch "train-as-val" pass).
    # None = no cap (full epoch over train_loader). Set this to limit
    # train-as-val wall time without touching the actual training loop
    # — the training loop iterates train_loader normally.
    train_eval_max_batches: Optional[int] = None

    # Cap the number of batches processed during the per-epoch val pass.
    # None = no cap (full val_loader). Same early-break mechanism as
    # train_eval_max_batches.
    val_max_batches: Optional[int] = None

    # When set, every this-many training batches, push a sub-epoch
    # snapshot of the running training metrics (loss, components, grad
    # norm, lr) to W&B under the `train_batch/` namespace, indexed by a
    # custom step metric `train_batch/global_step`. This lets you see
    # within-epoch trends live instead of waiting for the next epoch
    # boundary. Falls back to JSONL when W&B is disabled. Set None to
    # disable. Recommended value: same as log_every_n_batches (10) for
    # roughly one W&B point per stdout progress line.
    wandb_log_every_n_batches: Optional[int] = None

    # Threshold for "obviously bad" gradient spikes. When the pre-clip grad
    # norm of a step exceeds this value, the trainer appends a record to
    # <output_dir>/grad_spikes.jsonl with the epoch, batch index, loss
    # components, and per-sample identifiers (image_id, instance_id,
    # image_path) so you can go inspect the offending batch later.
    # Set to None to disable. Defaults to something well above
    # grad_clip_norm so it doesn't fire on every clipped step.
    spike_grad_norm_threshold: Optional[float] = 100.0
    # Cap on number of spike records written per run to keep the sidecar
    # bounded if the run is fully diverged (would otherwise log every step).
    spike_log_max_records: int = 200

    # Number of samples drawn per epoch by the training WeightedRandomSampler.
    # None = len(dataset) (one logical pass per epoch — historical default).
    # Set lower to make each epoch shorter (more frequent val / checkpoint /
    # LR step) without dropping any data: each draw still picks any image
    # with probability 1/N_images, the dataset is unchanged, only the
    # *number of draws per epoch* changes. Use when val cadence matters
    # more than "one full pass = one epoch" semantics. Only takes effect
    # when image_balanced_sampling=True.
    train_samples_per_epoch: Optional[int] = None

    # When True on a resumed run, run one validation+val_on_train pass at
    # the resumed epoch *before* training continues, and log the results
    # under the existing `validation/` and `val_on_train/` categories.
    # Use this after changing something that affects metric values without
    # changing the weights — e.g. wiring up the denormalizing validation
    # wrapper so a `_norm` run's val metrics now land in log-space rather
    # than normalized space. Off by default: a normal resume should not
    # spend the extra val time. Override at the CLI with
    # `trainer.loop.revalidate_on_resume=true`.
    revalidate_on_resume: bool = False

    validationConfig: ValConfig = ValConfig()


# ----------------------------
# Trainer
# ----------------------------
class Trainer:
    """
    Assumptions:
      - task.compute_loss(batch) -> (lossVal, lossMetricsDict)
          lossMetricsDict includes (for the per-mask task):
            {"loss_50": ..., "loss_r": ..., "loss_c": ...}
          (values can be floats or tensors)
      - train_loader & val_loader yield batches consumable by the task
        (dict or tuple; per-mask task expects keys
        image, mask, y50, yr, yc)
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
        task: TrainingTask,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        validator: SurvivalValidationRunner,
        output_dir: str,
        start_epoch: int = 0,
        cfg: Optional[TrainerConfig] = None,
        full_cfg_yaml: Optional[str] = None,
        val_on_train_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
        self.task = task
        # Convenience alias so checkpointing / device moves / eval mode
        # continue to address the underlying nn.Module directly.
        self.model = task.model
        self.loss_fn = task.loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Optional dedicated val-on-train loader. When None, reuse the
        # train loader (preserves the old behavior; works for per-mask
        # training where the train loader yields per-instance batches
        # the validator can score directly). For dense training the
        # train loader yields dense batches that the per-instance
        # validator cannot consume, so we always supply a separate
        # per-instance loader here.
        self.val_on_train_loader = val_on_train_loader if val_on_train_loader is not None else train_loader
        self.validator = validator
        # Lazily built in train(); the dense task swaps in a pooling
        # facade so the existing per-instance validator can score the
        # dense model on the same val population as the per-mask runs.
        self._validation_model: Optional[nn.Module] = None

        self.start_epoch = int(start_epoch)
        self.cfg = cfg or TrainerConfig()
        self.full_cfg_yaml = full_cfg_yaml or ""
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

        # W&B path. We deliberately do not pass step= here: the batch
        # stream auto-increments wandb's internal step via its own
        # train_batch/global_step axis, so passing a small step=epoch
        # would be rejected as non-monotonic. Epoch metrics land at
        # whatever internal step wandb is currently on.
        if allow_wandb and self.wandb is not None and self._wandb_run is not None:
            self.wandb.log(payload)  # type: ignore
            return

        # JSONL fallback
        if self.jsonl_logger is None:
            self.jsonl_logger = JsonlLogger(os.path.join(self.output_dir, "metrics.jsonl"))
        record = dict(payload)
        if category is not None:
            record["_category"] = category
        self.jsonl_logger.log(record, step=step)


    def save_checkpoint(self, epoch: int, filename: str) -> str:
        path = os.path.join(self.output_dir, "checkpoints", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        meta = CheckpointMeta(
            best_score=self.best_score,
            best_epoch=self.best_epoch,
            epochs_since_improve=self.epochs_since_improve,
            trainer_cfg=self.full_cfg_yaml,
            # trainer_cfg=vars(self.cfg),
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
        # self.log_metrics({"saved_path": path, "epoch": epoch}, category="ckpt", step=epoch, allow_wandb=True)
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

        self.start_epoch = meta.epoch + 1
        print(f"[CKPT] Resumed: {ckpt_path} (epoch={meta.epoch}) -> next_epoch={self.start_epoch}")
        return self.start_epoch



    # ----------------------------
    # Training / validation
    # ----------------------------
    def train(self) -> Dict[str, Any]:
        self.model = self.model.to(self.device)
        # The validator always receives a per-mask-style forward; the
        # task constructs it (identity for per-mask, pooling facade for
        # dense). Building once and reusing keeps the facade's
        # ``.eval()`` state consistent across val passes.
        self._validation_model = self.task.make_validation_model().to(self.device)

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
            if getattr(self.cfg, "wandb_log_every_n_batches", None):
                try:
                    self.wandb.define_metric("train_batch/global_step")
                    self.wandb.define_metric(
                        "train_batch/*",
                        step_metric="train_batch/global_step",
                    )
                except Exception as e:
                    print(f"[W&B] WARN failed to define_metric for train_batch/*: {e}")
        else:
            self.jsonl_logger = self.jsonl_logger or JsonlLogger(
                os.path.join(self.output_dir, "metrics.jsonl")
            )
            print(f"[LOG] W&B disabled/unavailable; writing JSONL logs to {self.jsonl_logger.path}")

        # If epoch==0 start, save weights before training
        if self.start_epoch == 0:
            # Log initial validation metrics before training starts (epoch=0)
            print("Running initial validation before training starts...")
            val_score, val_metrics = self._validate(epoch=0, useValSplit=True, batchUseFreq=1)
            train_score, train_metrics = self._validate(epoch=0, useValSplit=False, batchUseFreq=1)
            self.log_metrics({"aggregate_score": val_score, **val_metrics}, category="validation", step=0, allow_wandb=True)
            self.log_metrics({"aggregate_score": train_score, **train_metrics}, category="val_on_train", step=0, allow_wandb=True)

            # Set best_score based on initial validation before saving the initial checkpoint
            self.best_score = float(val_score)
            # self.best_score = float("-inf")  # Don't set best_score based on initial validation; only update after first epoch
            self.save_checkpoint(epoch=0, filename="ckpt_epoch_0_pretrain.pth")

        elif getattr(self.cfg, "revalidate_on_resume", False):
            # Resumed run, and the user asked us to re-log validation metrics
            # before training continues — typically because something about
            # the validation path changed without changing the weights (e.g.
            # the denormalizing wrapper was added, so a `_norm` run's metrics
            # now land in log-space). Log at step = start_epoch - 1 so the
            # new metrics overwrite the same x-axis position as the last
            # completed epoch in JSONL; W&B ignores the `step` arg and just
            # appends at the current internal step, which is also what we
            # want (a single re-validation point in the timeline).
            revalidate_at = self.start_epoch - 1
            print(
                f"[CKPT] revalidate_on_resume=true; running validation at "
                f"epoch {revalidate_at} before resuming training..."
            )
            val_score, val_metrics = self._validate(
                epoch=revalidate_at, useValSplit=True, batchUseFreq=1
            )
            train_score, train_metrics = self._validate(
                epoch=revalidate_at, useValSplit=False, batchUseFreq=1
            )
            self.log_metrics(
                {"aggregate_score": val_score, **val_metrics},
                category="validation",
                step=revalidate_at,
                allow_wandb=True,
            )
            self.log_metrics(
                {"aggregate_score": train_score, **train_metrics},
                category="val_on_train",
                step=revalidate_at,
                allow_wandb=True,
            )
            # Re-seat best_score so early stopping uses the same metric space
            # as the new (post-revalidate) curve, not the stale pre-resume value.
            self.best_score = float(val_score)
            self.best_epoch = revalidate_at

        # Epoch 0 is pretrain-only (saved above). Training epochs start at 1.
        # When resuming, start_epoch is already the correct next epoch (never 0).
        loop_start = max(self.start_epoch, 1)
        for epoch in range(loop_start, self.cfg.max_epochs + 1):
            train_loss_avg, train_loss_dict_avgs = self._train_one_epoch(epoch)

            # Epoch-based train logging (step=epoch)
            self.log_metrics({"loss": train_loss_avg, **train_loss_dict_avgs}, category="train", step=epoch, allow_wandb=True)

            # Validate (epoch-based)
            val_score, val_metrics = self._validate(epoch, useValSplit=True, batchUseFreq=1)
            # Also compute the merics on the training set for this epoch to see train vs val metric gap (epoch-based)
            train_score, train_metrics = self._validate(epoch, useValSplit=False, batchUseFreq=1)
            # Log train metrics on the training set as well (step=epoch)
            self.log_metrics({"aggregate_score": train_score, **train_metrics}, category="val_on_train", step=epoch, allow_wandb=True)

            # Epoch-based validation logging (step=epoch)
            self.log_metrics({**val_metrics, "aggregate_score": val_score}, category="validation", step=epoch, allow_wandb=True)

            # Step LR scheduler per epoch
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # higher-is-better metric => pass negative
                    self.lr_scheduler.step(-float(val_score))
                else:
                    self.lr_scheduler.step()

            # Always overwrite ckpt_latest.pth so a crash recovery point is
            # at-most-one-epoch stale, regardless of save_every_n_epochs.
            # save_checkpoint writes via tmp+rename so the previous good
            # file is never truncated.
            self.save_checkpoint(epoch=epoch, filename="ckpt_latest.pth")

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

        # Optimization-health meters: grad norm is measured *before* clipping
        # (the value returned by clip_grad_norm_ is the pre-clip total norm).
        # Rising train loss with rising grad-norm = optimization blowing up;
        # rising train loss with steady/falling grad-norm = label/overfit issue.
        gradnorm_meter = AverageMeter()
        gradnorm_max = 0.0
        gradnorm_samples: list = []  # for p95 at epoch end
        n_clipped = 0
        n_steps = 0
        loss_max = float("-inf")

        # Lazily-created sidecar logger for grad-norm spikes. We hold the
        # path on `self` so the cap persists across epochs.
        if not hasattr(self, "_spike_records_written"):
            self._spike_records_written = 0
        spike_thr = getattr(self.cfg, "spike_grad_norm_threshold", None)
        spike_cap = int(getattr(self.cfg, "spike_log_max_records", 0) or 0)
        spike_log_path = os.path.join(self.output_dir, "grad_spikes.jsonl")

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

            # Per-mask task expects image/mask/y50/yr/yc; other tasks (e.g.
            # the dense task) accept different keys. Batch size is taken
            # from "image" since every task we have ships images.
            bs = int(b["image"].shape[0])

            data_time.update(dt, n=1)

            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast_ctx(enabled=use_amp):
                lossVal, lossMetricsDict = self.task.compute_loss(b)

            # backward + step. clip_grad_norm_ returns the pre-clip total
            # norm; we capture it regardless of whether clipping is enabled
            # so we can tell whether updates are being magnitude-capped.
            grad_norm_val: Optional[float] = None
            if self._scaler.is_enabled():
                self._scaler.scale(lossVal).backward()
                self._scaler.unscale_(self.optimizer)
                clip_to = self.cfg.grad_clip_norm if self.cfg.grad_clip_norm is not None else float("inf")
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_to)
                grad_norm_val = float(gn)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                lossVal.backward()
                clip_to = self.cfg.grad_clip_norm if self.cfg.grad_clip_norm is not None else float("inf")
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_to)
                grad_norm_val = float(gn)
                self.optimizer.step()

            n_steps += 1
            if math.isfinite(grad_norm_val):
                gradnorm_meter.update(grad_norm_val, n=1)
                gradnorm_samples.append(grad_norm_val)
                if grad_norm_val > gradnorm_max:
                    gradnorm_max = grad_norm_val
                if (self.cfg.grad_clip_norm is not None
                        and grad_norm_val > float(self.cfg.grad_clip_norm)):
                    n_clipped += 1

            # Spike logger: dump per-sample identifiers when this step's
            # pre-clip grad norm crosses the configured threshold. Keeps
            # a per-run cap so a fully-diverged run can't generate
            # gigabytes of JSONL.
            if (spike_thr is not None
                    and grad_norm_val is not None
                    and math.isfinite(grad_norm_val)
                    and grad_norm_val > float(spike_thr)
                    and self._spike_records_written < spike_cap):
                try:
                    rec: Dict[str, Any] = {
                        "epoch": int(epoch),
                        "batch_idx": int(idx),
                        "global_step": int(self.global_step),
                        "grad_norm": float(grad_norm_val),
                        "loss": float(loss_item),
                    }
                    if isinstance(lossMetricsDict, dict):
                        for k, v in lossMetricsDict.items():
                            rec[f"loss/{k}"] = _as_float(v)
                    # Per-sample identifiers. The per-mask dataset emits
                    # image_id (str), instance_id (LongTensor), image_path
                    # (str). The dense dataset emits image_id / image_path.
                    # We tolerate missing keys so the same code works for
                    # both task formats.
                    for key in ("image_id", "instance_id", "image_path"):
                        if key in b:
                            v = b[key]
                            if torch.is_tensor(v):
                                rec[key] = v.detach().cpu().tolist()
                            elif isinstance(v, (list, tuple)):
                                rec[key] = list(v)
                            else:
                                rec[key] = v
                    os.makedirs(os.path.dirname(spike_log_path), exist_ok=True)
                    with open(spike_log_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, default=str) + "\n")
                    self._spike_records_written += 1
                    print(
                        f"[GradSpike] epoch={epoch} batch={idx} "
                        f"grad_norm={grad_norm_val:.1f} loss={loss_item:.4f} "
                        f"(logged to {spike_log_path}; "
                        f"{self._spike_records_written}/{spike_cap})"
                    )
                except Exception as e:
                    print(f"[GradSpike] WARN failed to log spike: {e}")

            # Batch time includes compute + step (and some python overhead)
            bt = time.perf_counter() - end
            batch_time.update(bt, n=1)

            # Update loss meters (sample-weighted)
            loss_item = float(lossVal.detach().cpu().item())
            loss_meter.update(loss_item, n=bs)
            if loss_item > loss_max:
                loss_max = loss_item

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
                gn_str = f"{grad_norm_val:.3f}" if grad_norm_val is not None else "n/a"
                gn_avg_str = f"{gradnorm_meter.avg:.3f}" if gradnorm_meter.count > 0 else "n/a"
                print(
                    f"Epoch: [{epoch}][{N3}/{n4_str}]\t"
                    f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"L50 {loss50_meter.val:.4f} ({loss50_meter.avg:.4f})\t"
                    f"Lr {lossr_meter.val:.4f} ({lossr_meter.avg:.4f})\t"
                    f"Lc {lossc_meter.val:.4f} ({lossc_meter.avg:.4f})\t"
                    f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                    f"GN {gn_str} ({gn_avg_str})"
                )

            # Sub-epoch W&B logging (independent cadence from the stdout
            # print). Pushes the *running* averages so each point is a
            # smoothed view of the epoch so far — easier to read than
            # raw per-batch values. Uses commit=True so the dashboard
            # updates live instead of buffering until the next call.
            n_every = getattr(self.cfg, "wandb_log_every_n_batches", None)
            if n_every and (idx % int(n_every)) == 0:
                batch_payload: Dict[str, float] = {
                    "train_batch/global_step": float(self.global_step),
                    "train_batch/epoch": float(epoch),
                    "train_batch/loss": float(loss_meter.avg),
                    "train_batch/loss_inst": float(loss_item),
                    "train_batch/loss_50": float(loss50_meter.avg),
                    "train_batch/loss_r": float(lossr_meter.avg),
                    "train_batch/loss_c": float(lossc_meter.avg),
                }
                if grad_norm_val is not None and math.isfinite(grad_norm_val):
                    batch_payload["train_batch/grad_norm"] = float(grad_norm_val)
                if gradnorm_meter.count > 0:
                    batch_payload["train_batch/grad_norm_avg"] = float(gradnorm_meter.avg)
                try:
                    for gi, pg in enumerate(self.optimizer.param_groups):
                        lr_key = ("train_batch/lr" if len(self.optimizer.param_groups) == 1
                                  else f"train_batch/lr_group{gi}")
                        batch_payload[lr_key] = float(pg.get("lr", 0.0))
                except Exception:
                    pass
                if self._wandb_run is not None:
                    try:
                        self._wandb_run.log(batch_payload, commit=True)
                    except Exception as e:
                        print(f"[W&B] WARN batch log failed: {e}")
                else:
                    # JSONL fallback when W&B is off; reuse the same logger
                    if self.jsonl_logger is None:
                        self.jsonl_logger = JsonlLogger(
                            os.path.join(self.output_dir, "metrics.jsonl")
                        )
                    self.jsonl_logger.log(batch_payload, step=self.global_step)

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

        # Optimization-health summary (one row per epoch under train/ in W&B).
        # frac_clipped tells you how often pre-clip norm exceeded grad_clip_norm.
        # If it climbs over epochs, the optimizer is being held back more and
        # more — i.e., gradients are still growing even though clipping hides
        # it in the parameter update.
        if n_steps > 0 and len(gradnorm_samples) > 0:
            gns = sorted(gradnorm_samples)
            p95_idx = max(0, min(len(gns) - 1, int(0.95 * (len(gns) - 1))))
            train_loss_dict_avgs.setdefault("grad_norm_avg", float(gradnorm_meter.avg))
            train_loss_dict_avgs.setdefault("grad_norm_max", float(gradnorm_max))
            train_loss_dict_avgs.setdefault("grad_norm_p95", float(gns[p95_idx]))
            train_loss_dict_avgs.setdefault("grad_norm_frac_clipped", float(n_clipped) / float(n_steps))
        train_loss_dict_avgs.setdefault("loss_max_batch", float(loss_max) if math.isfinite(loss_max) else 0.0)

        # Current LR(s). For single-group optimizers this is one value; for
        # param-grouped optimizers (e.g. decoupled head/backbone LRs) we log
        # each group so a stale or misconfigured scheduler is visible.
        try:
            for gi, pg in enumerate(self.optimizer.param_groups):
                lr_key = "lr" if len(self.optimizer.param_groups) == 1 else f"lr_group{gi}"
                train_loss_dict_avgs.setdefault(lr_key, float(pg.get("lr", 0.0)))
        except Exception:
            pass

        return train_loss_avg, train_loss_dict_avgs

    @torch.no_grad()
    def _validate(self, epoch: int, useValSplit: bool=True, batchUseFreq: int = 1) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        # Validator always sees a per-mask-style forward via the task's
        # validation model (identity for per-mask tasks; pooling facade
        # for the dense task).
        if self._validation_model is None:
            self._validation_model = self.task.make_validation_model().to(self.device)
        val_model = self._validation_model
        val_model.eval()
        if (useValSplit):
            agg_score, metrics = self.validator.validate(
                val_model, self.val_loader, batchUseFreq=batchUseFreq,
                max_batches=getattr(self.cfg, "val_max_batches", None),
            )
        else:
            # Train-as-val pass: cap the number of batches we process
            # via TrainerConfig.train_eval_max_batches so this stays
            # bounded even when the val-on-train loader is huge. Caps
            # via an early break in the validator so DataLoader workers
            # don't keep producing batches we'd discard. Does NOT modify
            # the training loop's use of train_loader.
            agg_score, metrics = self.validator.validate(
                val_model,
                self.val_on_train_loader,
                batchUseFreq=batchUseFreq,
                max_batches=getattr(self.cfg, "train_eval_max_batches", None),
            )

        # normalize validation metrics dict to floats
        val_metrics: Dict[str, float] = {}
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                val_metrics[k] = _as_float(v)

        if (useValSplit):
            print(f"[Val]\tEpoch {epoch}\taggregate_score={float(agg_score):.6f}")
        else:
            print(f"[Val on train]\tEpoch {epoch}\taggregate_score={float(agg_score):.6f}")

        return float(agg_score), val_metrics

