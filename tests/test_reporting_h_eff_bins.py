"""Tests for BasicMetricsAnalyzer's effective-horizon binning (Phase 4)."""

from __future__ import annotations

import csv
import json
import math
import tempfile
from pathlib import Path

import torch

from persistence_eval.analysis.reporting import BasicMetricsAnalyzer
from persistence_eval.core.domain import EvaluationBatch


def _make_batch(query_times, h_effs, gt, preds):
    n = len(query_times)
    return EvaluationBatch(
        episode_ids=[f"ep:{i}" for i in range(n)],
        query_times=torch.tensor(query_times, dtype=torch.float64),
        predicted_probs=torch.tensor(preds, dtype=torch.float64),
        ground_truth=torch.tensor(gt, dtype=torch.float64),
        inference_latency_ms=torch.zeros(n, dtype=torch.float64),
        filter_latency_ms=torch.zeros(n, dtype=torch.float64),
        effective_horizons=torch.tensor(h_effs, dtype=torch.float64),
        metadata=[{"method_name": "m", "dataset_name": "d", "split": "s"} for _ in range(n)],
    )


def test_no_h_eff_bin_file_for_anchor_only():
    """When all effective_horizons are NaN, the binned summary file isn't written."""
    nan = float("nan")
    batch = _make_batch(
        query_times=[1.0, 2.0],
        h_effs=[nan, nan],
        gt=[1.0, 0.0],
        preds=[0.9, 0.1],
    )
    analyzer = BasicMetricsAnalyzer(group_by=["method_name"])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        analyzer.analyze([], batch, out)
        assert not (out / "effective_horizon_summary.csv").exists()


def test_h_eff_summary_groups_records_into_bins():
    batch = _make_batch(
        query_times=[1.0, 2.0, 4.0, 10.0, 20.0],
        h_effs=[0.5, 1.5, 3.0, 5.0, 9.0],
        gt=[1.0, 1.0, 0.0, 0.0, 0.0],
        preds=[0.9, 0.8, 0.6, 0.3, 0.1],
    )
    analyzer = BasicMetricsAnalyzer(group_by=["method_name"], effective_horizon_bins=3)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        analyzer.analyze([], batch, out)
        path = out / "effective_horizon_summary.csv"
        assert path.exists()
        rows = list(csv.DictReader(path.open()))

    # max h_eff = 9.0, 3 bins ⇒ edges [0, 3, 6, 9]; overflow has nothing
    # (since max is 9.0 inclusive ⇒ falls into the overflow bucket per torch.bucketize).
    # Records counted by sum.
    total = sum(int(r["n_records"]) for r in rows)
    assert total == 5


def test_per_record_json_includes_effective_horizon():
    batch = _make_batch(
        query_times=[1.0, 2.0],
        h_effs=[0.5, float("nan")],
        gt=[1.0, 0.0],
        preds=[0.9, 0.1],
    )
    analyzer = BasicMetricsAnalyzer(group_by=["method_name"])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        analyzer.analyze([], batch, out)
        records = json.loads((out / "prediction_records.json").read_text())

    assert math.isclose(records[0]["effective_horizon"], 0.5)
    assert records[1]["effective_horizon"] is None
