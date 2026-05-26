import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

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

    # Optional top-level `gemini:` block configures the process-wide Gemini
    # singleton (API client, cumulative token-usage tracker, snapshot file,
    # safety-prompt opt-out). Configs without a `gemini:` section fall back
    # to lazy env-var init on first engine construction. The pre-first-call
    # safety prompt is owned by the engine itself (see _shared.maybe_confirm_first_call).
    gemini_cfg = cfg.get("gemini") if isinstance(cfg, DictConfig) else None
    if gemini_cfg is not None:
        from dataset_generation.llm_persistence_querying.gemini._shared import (
            configure_gemini_shared,
        )
        configure_gemini_shared(**OmegaConf.to_container(gemini_cfg, resolve=True))

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
