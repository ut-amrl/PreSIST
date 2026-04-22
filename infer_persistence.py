import hydra
from omegaconf import DictConfig

from persistence_eval.runners.orchestrator import BenchmarkOrchestrator


@hydra.main(config_path="configs/train_confs", config_name="eval_config", version_base=None)
def main(cfg: DictConfig) -> None:
    orchestrator = BenchmarkOrchestrator.from_cfg(cfg)
    artifacts, run_dir = orchestrator.run_inference()
    print(f"Wrote {len(artifacts)} inference artifacts to {run_dir}")


if __name__ == "__main__":
    main()
