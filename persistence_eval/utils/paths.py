from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class BenchmarkLayout:
    """Resolved directory layout for one (dataset, method, benchmark) combination."""
    root: Path  # <output_root>/<dataset>/<method>/<benchmark>/

    def inference_runs_dir(self) -> Path:
        return self.root / "inference_runs"

    def inference_run_dir(self, timestamp: str) -> Path:
        return self.inference_runs_dir() / timestamp

    def latest_inference_run(self) -> Path:
        runs = sorted(self.inference_runs_dir().iterdir())
        if not runs:
            raise FileNotFoundError(f"No inference runs found under {self.inference_runs_dir()}")
        return runs[-1]


def resolve_output_layout(
    output_root: Path,
    dataset_name: str,
    method_name: str,
    benchmark_name: str,
) -> BenchmarkLayout:
    root = output_root / dataset_name / method_name / benchmark_name
    return BenchmarkLayout(root=root)
