from __future__ import annotations

from persistence_eval.core.domain import EvaluationBatch, MetricResult
from persistence_eval.metrics.base import Metric


class ExistenceMAE(Metric):
    """Mean absolute error between predicted_prob and 0/1 ground truth. Skips unknown labels."""

    def compute(self, batch: EvaluationBatch) -> list[MetricResult]:
        mask = batch.labeled_mask()
        if not mask.any():
            return []
        mae = (batch.predicted_probs[mask] - batch.ground_truth[mask]).abs().mean().item()
        return [MetricResult(
            name="existence_mae",
            value=mae,
            count=int(mask.sum().item()),
        )]
