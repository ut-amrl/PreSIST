import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

# Registers OmegaConf resolvers used in our config interpolations
# (e.g. ${dot_to_p:...}). Must precede @hydra.main.
import persistence_eval.utils.hydra_resolvers  # noqa: F401

from persistence_eval.runners.orchestrator import BenchmarkOrchestrator


@hydra.main(config_path="configs/train_confs", config_name="eval_config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed = cfg.runtime.get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print("Starting evaluation with config:")
    print(cfg)

    orchestrator = BenchmarkOrchestrator.from_cfg(cfg)
    if cfg.mode == "evaluate":
        batch, metrics = orchestrator.run_evaluation()
        print(f"Evaluation complete: {len(batch)} prediction records, {len(metrics)} metric results")
    elif cfg.mode == "run_all":
        orchestrator.run_all()
        print("run_all complete")
    elif cfg.mode == "infer":
        artifacts, run_dir = orchestrator.run_inference()
        print(f"Inference complete: {len(artifacts)} artifacts in {run_dir}")
    else:
        raise ValueError(f"eval_persistence.py: unexpected mode '{cfg.mode}'. Use 'evaluate', 'run_all', or 'infer'.")


if __name__ == "__main__":
    main()
