from __future__ import annotations

import torch

from persistence_eval.core.domain import EvaluationBatch, MetricResult
from persistence_eval.metrics.base import Metric


class ThresholdAccuracy(Metric):
    """
    Classification metrics at a fixed decision threshold.
    Returns accuracy, balanced accuracy, F1, precision, and recall.
    Skips records with unknown ground truth (NaN).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def compute(self, batch: EvaluationBatch) -> list[MetricResult]:
        mask = batch.labeled_mask()
        if not mask.any():
            return []

        preds = (batch.predicted_probs[mask] >= self._threshold)
        labels = batch.ground_truth[mask].bool()
        n = int(mask.sum().item())
        t = self._threshold

        tp = float((preds & labels).sum().item())
        tn = float((~preds & ~labels).sum().item())
        fp = float((preds & ~labels).sum().item())
        fn = float(( ~preds & labels).sum().item())

        accuracy = (tp + tn) / n
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_accuracy = (tpr + tnr) / 2.0
        f1_denom = 2 * tp + fp + fn
        f1 = (2 * tp) / f1_denom if f1_denom > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr

        return [
            MetricResult(name=f"accuracy@{t}", value=accuracy, count=n),
            MetricResult(name=f"balanced_accuracy@{t}", value=balanced_accuracy, count=n),
            MetricResult(name=f"f1@{t}", value=f1, count=n),
            MetricResult(name=f"precision@{t}", value=precision, count=n),
            MetricResult(name=f"recall@{t}", value=recall, count=n),
            MetricResult(name=f"tp@{t}", value=tp, count=n),
            MetricResult(name=f"tn@{t}", value=tn, count=n),
            MetricResult(name=f"fp@{t}", value=fp, count=n),
            MetricResult(name=f"fn@{t}", value=fn, count=n),
            MetricResult(name=f"tpr@{t}", value=tp / n, count=n),
            MetricResult(name=f"tnr@{t}", value=tn / n, count=n),
            MetricResult(name=f"fpr@{t}", value=fp / n, count=n),
            MetricResult(name=f"fnr@{t}", value=fn / n, count=n),
        ]
