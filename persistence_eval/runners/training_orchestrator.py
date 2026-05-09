"""Orchestrator for training persistence parameters.

Builds a TrainingDataset from a Hydra config, instantiates a Trainer,
runs `trainer.train(dataset)`, and writes:
  - <output_dir>/persistence_params.json   — bucket-keyed param entries
  - <output_dir>/training_metadata.json    — sweep params + bucket sizes

The output schema mirrors SemanticClassPersistenceParamLookup's JSON exactly,
so existing class-keyed inferencers consume per-class outputs without change,
and LocationPersistenceParamLookup consumes per-location outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from hydra.utils import instantiate

from persistence_eval.methods.training.base import (
    AbsTrainer,
    TrainingDataset,
    bucket_tracks,
    load_manifest_with_resolved_paths,
)


class TrainingOrchestrator:
    def __init__(
        self,
        dataset: TrainingDataset,
        trainer: AbsTrainer,
        output_dir: Path,
    ) -> None:
        self.dataset = dataset
        self.trainer = trainer
        self.output_dir = Path(output_dir)

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "TrainingOrchestrator":
        manifest_path = Path(cfg.train_dataset.manifest_path)
        raw_tracks = load_manifest_with_resolved_paths(
            manifest_path,
            host_root=cfg.train_dataset.get("host_root"),
            container_root=cfg.train_dataset.get("container_root"),
        )
        dataset = TrainingDataset(
            raw_tracks=raw_tracks,
            cell_size_px=int(cfg.train_dataset.get("cell_size_px", 64)),
            class_id=int(cfg.train_dataset.get("class_id", 0)),
            subsample_fraction=float(cfg.experiment.subsample_fraction),
            subsample_seed=int(cfg.experiment.subsample_seed),
            val_track_fraction=float(cfg.experiment.get("val_track_fraction", 0.0)),
            val_split_seed=int(cfg.experiment.get("val_split_seed", 0)),
            manifest_dir=manifest_path.parent,
        )
        trainer_kwargs: dict[str, Any] = {"grouping": cfg.grouping}
        # Vision trainers consume a pre-instantiated model from cfg.model;
        # distribution-fit trainers don't have cfg.model in their experiment
        # presets, so this is conditional.
        if "model" in cfg:
            trainer_kwargs["model"] = instantiate(cfg.model, _convert_="all")
        trainer = instantiate(cfg.trainer, **trainer_kwargs, _convert_="all")
        # Trained perpetua models live under `paths.model_dir`. (Eval-side
        # `paths.output_root` points at the standard
        # /workspace/data/output/persistence_eval_outputs and is unused here.)
        output_dir = Path(cfg.paths.model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Vision trainers need to write checkpoints; expose output_dir on
        # trainers that opt in (distribution-fit trainers do not have this
        # attribute and are unaffected).
        if hasattr(trainer, "output_dir"):
            trainer.output_dir = output_dir
        # Vision trainers may consume the resolved cfg as wandb config; fill
        # in a default run name if one wasn't supplied.
        if hasattr(trainer, "wandb_config"):
            from omegaconf import OmegaConf
            trainer.wandb_config = OmegaConf.to_container(cfg, resolve=True)
        if (
            getattr(trainer, "_wandb_run_name", None) is None
            and getattr(trainer, "_use_wandb", False)
        ):
            train_ds_name = cfg.train_dataset.get("name", "train")
            trainer._wandb_run_name = (
                f"{train_ds_name}__{getattr(trainer, 'name', 'trainer')}"
                f"__frac{cfg.experiment.subsample_fraction}"
            )
        return cls(dataset=dataset, trainer=trainer, output_dir=output_dir)

    def run(self) -> dict[str, Any]:
        entries = self.trainer.train(self.dataset)
        params_path = self.output_dir / "persistence_params.json"
        with params_path.open("w") as f:
            json.dump(entries, f, indent=2, sort_keys=True)

        # Recompute bucket sizes after subsample for diagnostics.
        tracks = list(self.dataset.iter_tracks())
        buckets = bucket_tracks(tracks, self.trainer.grouping)
        bucket_sizes = {k: len(v) for k, v in buckets.items()}

        metadata = {
            "trainer_name": getattr(self.trainer, "name", type(self.trainer).__name__),
            "trainer_class": type(self.trainer).__name__,
            "grouping": self.trainer.grouping,
            "subsample_fraction": self.dataset.subsample_fraction,
            "subsample_seed": self.dataset.subsample_seed,
            "n_tracks_used": len(tracks),
            "n_buckets": len(bucket_sizes),
            "bucket_sizes": bucket_sizes,
            "params_path": str(params_path),
        }
        meta_path = self.output_dir / "training_metadata.json"
        with meta_path.open("w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        return metadata
