# checkpointing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


@dataclass
class CheckpointMeta:
    epoch: int = 0
    global_step: int = 0
    best_score: float = float("-inf")
    best_epoch: int = 0
    epochs_since_improve: int = 0
    # trainer_cfg: Optional[Dict[str, Any]] = None
    trainer_cfg: Optional[str] = None  # YAML string of the full trainer config for easier loading and reproducibility


def make_checkpoint_dict(
    *,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,  # torch scheduler types vary
    scaler: Optional[Any] = None,        # torch.cuda.amp.GradScaler
    meta: Optional[CheckpointMeta] = None,
) -> Dict[str, Any]:
    meta = meta or CheckpointMeta()
    return {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_score": float(meta.best_score),
        "best_epoch": int(meta.best_epoch),
        "epochs_since_improve": int(meta.epochs_since_improve),
        "trainer_cfg": meta.trainer_cfg
    }


def save_checkpoint(state: Dict[str, Any], path: str) -> str:
    torch.save(state, path)
    return path


def load_raw_checkpoint(
    path: str,
    *,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def load_model_from_checkpoint(
    model: nn.Module,
    ckpt: Union[str, Dict[str, Any]],
    *,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> CheckpointMeta:
    if isinstance(ckpt, str):
        ckpt = load_raw_checkpoint(ckpt, map_location=map_location)

    model.load_state_dict(ckpt["model_state_dict"], strict=strict)

    return CheckpointMeta(
        epoch=int(ckpt.get("epoch", 0)),
        global_step=int(ckpt.get("global_step", 0)),
        best_score=float(ckpt.get("best_score", float("-inf"))),
        best_epoch=int(ckpt.get("best_epoch", 0)),
        epochs_since_improve=int(ckpt.get("epochs_since_improve", 0)),
        trainer_cfg=ckpt.get("trainer_cfg", None),
    )


def load_training_state_from_checkpoint(
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    lr_scheduler: Optional[Any],
    scaler: Optional[Any],
    ckpt: Union[str, Dict[str, Any]],
    map_location: Optional[Union[str, torch.device]] = None,
    strict_model: bool = False,
    load_optimizer: bool = True,
    load_scheduler: bool = True,
    load_scaler: bool = True,
) -> CheckpointMeta:
    """
    Trainer-oriented loader. Lets you selectively ignore optimizer/scheduler/scaler
    (useful when changing AMP mode, LR schedule, etc.).
    """
    if isinstance(ckpt, str):
        ckpt = load_raw_checkpoint(ckpt, map_location=map_location)

    model.load_state_dict(ckpt["model_state_dict"], strict=strict_model)

    if load_optimizer and optimizer is not None:
        opt_state = ckpt.get("optimizer_state_dict", None)
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)

    if load_scheduler and lr_scheduler is not None:
        sched_state = ckpt.get("lr_scheduler_state_dict", None)
        if sched_state is not None:
            lr_scheduler.load_state_dict(sched_state)

    if load_scaler and scaler is not None:
        scaler_state = ckpt.get("scaler_state_dict", None)
        if scaler_state is not None:
            try:
                scaler.load_state_dict(scaler_state)
            except Exception:
                # e.g. switching amp on/off or scaler version mismatch
                pass

    return CheckpointMeta(
        epoch=int(ckpt.get("epoch", 0)),
        global_step=int(ckpt.get("global_step", 0)),
        best_score=float(ckpt.get("best_score", float("-inf"))),
        best_epoch=int(ckpt.get("best_epoch", 0)),
        epochs_since_improve=int(ckpt.get("epochs_since_improve", 0)),
        trainer_cfg=ckpt.get("trainer_cfg", None),
    )
