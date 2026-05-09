from __future__ import annotations

import csv
import json
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import torch

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

    For evaluations that populate per-row effective horizons (e.g. the
    scaled-horizon alt eval), additionally writes effective_horizon_summary.csv
    binned by h_eff. Bin edges default to `effective_horizon_bins` linear bins
    from 0 to the observed max h_eff (plus an overflow bin).
    """

    def __init__(
        self,
        group_by: list[str] | None = None,
        effective_horizon_bins: int = 10,
    ) -> None:
        self._group_by = group_by or ["method_name", "dataset_name", "split"]
        self._h_eff_bins = max(int(effective_horizon_bins), 1)

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
        h_eff_list = batch.effective_horizons.tolist()
        records_data = [
            {
                "episode_id": batch.episode_ids[i],
                "query_time": batch.query_times[i].item(),
                "effective_horizon": None if math.isnan(h_eff_list[i]) else h_eff_list[i],
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

        # effective_horizon_summary.csv — only when h_eff is populated.
        self._write_effective_horizon_summary(batch, labeled_mask, output_dir)

    def _write_effective_horizon_summary(
        self,
        batch: EvaluationBatch,
        labeled_mask: torch.Tensor,
        output_dir: Path,
    ) -> None:
        h = batch.effective_horizons
        valid = ~torch.isnan(h)
        if not bool(valid.any()):
            return  # primary anchor-only eval; nothing to bin

        valid_h = h[valid]
        h_max = float(valid_h.max().item())
        if h_max <= 0.0:
            return

        n_bins = self._h_eff_bins
        edges = torch.linspace(0.0, h_max, n_bins + 1, dtype=torch.float64)
        # bucketize returns 1..n_bins for in-range, n_bins+1 for overflow
        bin_idx = torch.bucketize(h, edges[1:], right=False)  # 0..n_bins
        bin_idx = bin_idx.clamp(max=n_bins)  # n_bins is the overflow bucket

        rows = []
        for b in range(n_bins + 1):
            sel = (bin_idx == b) & valid
            n_records = int(sel.sum().item())
            if n_records == 0:
                continue
            sel_labeled = sel & labeled_mask
            n_labeled = int(sel_labeled.sum().item())

            if b < n_bins:
                lo = float(edges[b].item())
                hi = float(edges[b + 1].item())
                bin_label = f"[{lo:.4g}, {hi:.4g})"
            else:
                bin_label = f"[{h_max:.4g}, +inf)"

            row = {
                "bin": b,
                "h_eff_range": bin_label,
                "n_records": n_records,
                "n_labeled": n_labeled,
                "mean_h_eff": float(h[sel].mean().item()),
            }
            if n_labeled > 0:
                row["mean_predicted_prob"] = float(batch.predicted_probs[sel_labeled].mean().item())
                row["mean_ground_truth"] = float(batch.ground_truth[sel_labeled].mean().item())
                # MAE between predicted_prob and ground_truth on this slice.
                row["mae"] = float(
                    (batch.predicted_probs[sel_labeled] - batch.ground_truth[sel_labeled])
                    .abs()
                    .mean()
                    .item()
                )
            else:
                row["mean_predicted_prob"] = None
                row["mean_ground_truth"] = None
                row["mae"] = None
            rows.append(row)

        if not rows:
            return

        fieldnames = list(rows[0].keys())
        with open(output_dir / "effective_horizon_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
