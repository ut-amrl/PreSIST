from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

from persistence_eval.core.domain import EvaluationBatch, MetricResult


class MetricsAnalyzer(ABC):
    @abstractmethod
    def analyze(
        self,
        metrics: list[MetricResult],
        batch: EvaluationBatch,
        output_dir: Path,
    ) -> None: ...


class BasicMetricsAnalyzer(MetricsAnalyzer):
    """
    Groups EvaluationBatch records by configurable metadata keys and writes:
      metric_results.json, prediction_records.json, group_summary.csv
    """

    def __init__(self, group_by: list[str] | None = None) -> None:
        self._group_by = group_by or ["method_name", "dataset_name", "split"]

    def analyze(
        self,
        metrics: list[MetricResult],
        batch: EvaluationBatch,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # metric_results.json
        metric_data = [
            {"name": m.name, "value": m.value, "count": m.count, "group": m.group}
            for m in metrics
        ]
        (output_dir / "metric_results.json").write_text(json.dumps(metric_data, indent=2))

        # prediction_records.json — iterate over the batch as plain dicts
        gt_list = batch.ground_truth.tolist()
        records_data = [
            {
                "episode_id": batch.episode_ids[i],
                "query_time": batch.query_times[i].item(),
                "predicted_prob": batch.predicted_probs[i].item(),
                "ground_truth": None if gt_list[i] != gt_list[i] else bool(gt_list[i]),  # NaN check
                "inference_latency_ms": batch.inference_latency_ms[i].item(),
                "filter_latency_ms": batch.filter_latency_ms[i].item(),
                "metadata": batch.metadata[i],
            }
            for i in range(len(batch))
        ]
        (output_dir / "prediction_records.json").write_text(json.dumps(records_data, indent=2))

        # group_summary.csv — aggregate per metadata group using tensor ops
        labeled_mask = batch.labeled_mask()
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i in range(len(batch)):
            key = tuple(batch.metadata[i].get(k, "unknown") for k in self._group_by)
            groups[key].append(i)

        rows = []
        for key, indices in groups.items():
            import torch
            idx_t = torch.tensor(indices, dtype=torch.long)
            labeled_idx = idx_t[labeled_mask[idx_t]]

            row = dict(zip(self._group_by, key))
            row["n_records"] = len(indices)
            row["n_labeled"] = int(labeled_idx.numel())
            if labeled_idx.numel() > 0:
                row["mean_predicted_prob"] = batch.predicted_probs[labeled_idx].mean().item()
                row["mean_ground_truth"] = batch.ground_truth[labeled_idx].mean().item()
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
