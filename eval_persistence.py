import hydra
from omegaconf import DictConfig

from persistence_eval.runners.orchestrator import BenchmarkOrchestrator


@hydra.main(config_path="configs/train_confs", config_name="eval_config", version_base=None)
def main(cfg: DictConfig) -> None:
    orchestrator = BenchmarkOrchestrator.from_cfg(cfg)
    if cfg.mode == "evaluate":
        records, metrics = orchestrator.run_evaluation()
        print(f"Evaluation complete: {len(records)} prediction records, {len(metrics)} metric results")
    elif cfg.mode == "run_all":
        orchestrator.run_all()
        print("run_all complete")
    else:
        raise ValueError(f"eval_persistence.py: unexpected mode '{cfg.mode}'. Use 'evaluate' or 'run_all'.")


if __name__ == "__main__":
    main()
