"""Hydra entry point for training Perpetua-style persistence models.

Mirrors eval_persistence.py. Selects a trainer + train_dataset + experiment
namespace via Hydra; writes persistence_params.json and training_metadata.json
into ${paths.output_dir}.
"""

import random

import hydra
import numpy as np
from omegaconf import DictConfig

# Importing this module registers the OmegaConf resolvers used in our path
# templates (e.g. ${dot_to_p:...}). Must come before @hydra.main runs so the
# resolvers are in place when Hydra composes the config.
import persistence_eval.utils.hydra_resolvers  # noqa: F401

from persistence_eval.runners.training_orchestrator import TrainingOrchestrator


@hydra.main(
    config_path="configs/train_confs",
    config_name="train_config_perpetua_single_exp",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    seed = cfg.runtime.get("seed") if "runtime" in cfg else None
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print("Starting persistence-model training with config:")
    print(cfg)

    orchestrator = TrainingOrchestrator.from_cfg(cfg)
    metadata = orchestrator.run()
    print("Training complete.")
    print(f"  trainer:           {metadata['trainer_name']} ({metadata['trainer_class']})")
    print(f"  grouping:          {metadata['grouping']}")
    print(f"  n tracks used:     {metadata['n_tracks_used']}")
    print(f"  n buckets:         {metadata['n_buckets']}")
    print(f"  params written to: {metadata['params_path']}")


if __name__ == "__main__":
    main()
