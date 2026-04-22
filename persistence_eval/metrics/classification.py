from __future__ import annotations

from persistence_eval.core.domain import MetricResult, PredictionRecord
from persistence_eval.metrics.base import Metric


class ThresholdAccuracy(Metric):
    """Binary accuracy at a configurable decision threshold. Foundation for F1/P/R later."""

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def compute(self, records: list[PredictionRecord]) -> list[MetricResult]:
        raise NotImplementedError
