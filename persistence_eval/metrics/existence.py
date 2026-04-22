from __future__ import annotations

from persistence_eval.core.domain import MetricResult, PredictionRecord
from persistence_eval.metrics.base import Metric


class ExistenceMAE(Metric):
    """Mean absolute error between predicted_prob and 0/1 ground truth. Skips unknown labels."""

    def compute(self, records: list[PredictionRecord]) -> list[MetricResult]:
        raise NotImplementedError
