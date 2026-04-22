from __future__ import annotations

from persistence_eval.core.domain import MetricResult, PredictionRecord
from persistence_eval.metrics.base import Metric


class MeanQueryLatencyMs(Metric):
    """Mean filter_latency_ms across all prediction records."""

    def compute(self, records: list[PredictionRecord]) -> list[MetricResult]:
        raise NotImplementedError


class MeanImageTotalLatencyMs(Metric):
    """Per-anchor mean of (inference_latency_ms + sum of filter_latency_ms for all queries)."""

    def compute(self, records: list[PredictionRecord]) -> list[MetricResult]:
        raise NotImplementedError
