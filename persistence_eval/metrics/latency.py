from __future__ import annotations

from collections import defaultdict

from persistence_eval.core.domain import EvaluationBatch, MetricResult
from persistence_eval.metrics.base import Metric


class MeanQueryLatencyMs(Metric):
    """Mean filter_latency_ms across all prediction records."""

    def compute(self, batch: EvaluationBatch) -> list[MetricResult]:
        if len(batch) == 0:
            return []
        return [MetricResult(
            name="mean_query_latency_ms",
            value=batch.filter_latency_ms.mean().item(),
            count=len(batch),
        )]


class MeanImageTotalLatencyMs(Metric):
    """Per-episode mean of (inference_latency_ms + total filter_latency_ms across all queries)."""

    def compute(self, batch: EvaluationBatch) -> list[MetricResult]:
        if len(batch) == 0:
            return []

        # Aggregate per episode: inference latency is constant per episode;
        # filter latencies are per-query shares, so summing recovers the episode total.
        episode_inference: dict[str, float] = {}
        episode_filter_total: dict[str, float] = defaultdict(float)
        for i, eid in enumerate(batch.episode_ids):
            if eid not in episode_inference:
                episode_inference[eid] = batch.inference_latency_ms[i].item()
            episode_filter_total[eid] += batch.filter_latency_ms[i].item()

        totals = [
            episode_inference[eid] + episode_filter_total[eid]
            for eid in episode_inference
        ]
        mean_total = sum(totals) / len(totals)
        return [MetricResult(
            name="mean_image_total_latency_ms",
            value=mean_total,
            count=len(episode_inference),
        )]
