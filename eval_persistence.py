import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from persistence_eval.runners.orchestrator import BenchmarkOrchestrator


@hydra.main(config_path="configs/train_confs", config_name="eval_config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed = cfg.runtime.get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    orchestrator = BenchmarkOrchestrator.from_cfg(cfg)
    if cfg.mode == "evaluate":
        batch, metrics = orchestrator.run_evaluation()
        print(f"Evaluation complete: {len(batch)} prediction records, {len(metrics)} metric results")
    elif cfg.mode == "run_all":
        orchestrator.run_all()
        print("run_all complete")
    else:
        raise ValueError(f"eval_persistence.py: unexpected mode '{cfg.mode}'. Use 'evaluate' or 'run_all'.")


if __name__ == "__main__":
    main()
