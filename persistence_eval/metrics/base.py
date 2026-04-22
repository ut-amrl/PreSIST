from __future__ import annotations

from abc import ABC, abstractmethod

from persistence_eval.core.domain import MetricResult, PredictionRecord


class Metric(ABC):
    @abstractmethod
    def compute(self, records: list[PredictionRecord]) -> list[MetricResult]: ...


class CompositeMetricSuite(Metric):
    """Calls all child metrics and returns their concatenated results."""

    def __init__(self, metrics: list[Metric]) -> None:
        self._metrics = metrics

    def compute(self, records: list[PredictionRecord]) -> list[MetricResult]:
        results: list[MetricResult] = []
        for m in self._metrics:
            results.extend(m.compute(records))
        return results
