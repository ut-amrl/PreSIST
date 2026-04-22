from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from persistence_eval.core.domain import MetricResult, PredictionRecord


class MetricsAnalyzer(ABC):
    @abstractmethod
    def analyze(
        self,
        metrics: list[MetricResult],
        records: list[PredictionRecord],
        output_dir: Path,
    ) -> None: ...


class BasicMetricsAnalyzer(MetricsAnalyzer):
    """
    Groups PredictionRecords by configurable metadata keys and writes:
      metric_results.json, prediction_records.json, group_summary.csv
    """

    def __init__(self, group_by: list[str] | None = None) -> None:
        self._group_by = group_by or ["method_name", "dataset_name", "split"]

    def analyze(
        self,
        metrics: list[MetricResult],
        records: list[PredictionRecord],
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # metric_results.json
        metric_data = [
            {"name": m.name, "value": m.value, "count": m.count, "group": m.group}
            for m in metrics
        ]
        (output_dir / "metric_results.json").write_text(json.dumps(metric_data, indent=2))

        # prediction_records.json
        records_data = [
            {
                "episode_id": r.episode_id,
                "query_time": r.query_time,
                "predicted_prob": r.predicted_prob,
                "ground_truth": r.ground_truth,
                "inference_latency_ms": r.inference_latency_ms,
                "filter_latency_ms": r.filter_latency_ms,
                "metadata": r.metadata,
            }
            for r in records
        ]
        (output_dir / "prediction_records.json").write_text(json.dumps(records_data, indent=2))

        # group_summary.csv — aggregate predicted_prob and ground_truth per group
        groups: dict[tuple, list[PredictionRecord]] = defaultdict(list)
        for r in records:
            key = tuple(r.metadata.get(k, "unknown") for k in self._group_by)
            groups[key].append(r)

        rows = []
        for key, group_records in groups.items():
            labeled = [r for r in group_records if r.ground_truth is not None]
            row = dict(zip(self._group_by, key))
            row["n_records"] = len(group_records)
            row["n_labeled"] = len(labeled)
            if labeled:
                row["mean_predicted_prob"] = sum(r.predicted_prob for r in labeled) / len(labeled)
                row["mean_ground_truth"] = sum(float(r.ground_truth) for r in labeled) / len(labeled)
            else:
                row["mean_predicted_prob"] = None
                row["mean_ground_truth"] = None
            rows.append(row)

        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_dir / "group_summary.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
