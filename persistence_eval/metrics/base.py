from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from persistence_eval.core.domain import EvaluationBatch, MetricResult


class Metric(ABC):
    """Legacy batch-at-once metric interface. Wrap with BatchingMetricAccumulator for streaming use."""
    @abstractmethod
    def compute(self, batch: EvaluationBatch) -> list[MetricResult]: ...


class CompositeMetricSuite(Metric):
    """Calls all child metrics and returns their concatenated results."""

    def __init__(self, metrics: list[Metric]) -> None:
        self._metrics = metrics

    def compute(self, batch: EvaluationBatch) -> list[MetricResult]:
        results: list[MetricResult] = []
        for m in self._metrics:
            results.extend(m.compute(batch))
        return results


class AbsMetricAccumulator(ABC):
    """
    Streaming accumulator: update() once per (artifact, context) pair, compute() at end.
    Implementations may stratify results by bucketizing query_times, num_prior_observations,
    prior_observation_timespan_s, or metadata keys and returning multiple MetricResults
    with different MetricResult.group dicts.
    """

    @abstractmethod
    def update(
        self,
        predicted_probs: torch.Tensor,             # (Q,) float64
        ground_truth: torch.Tensor,                # (Q,) float64, NaN if unknown
        query_times: torch.Tensor,                 # (Q,) float64, dt from anchor_time (s)
        num_prior_observations: int,
        prior_observation_timespan_s: float,       # anchor_time - first_prior_obs.timestamp
        inference_latency_ms: float,
        filter_latency_ms: float,
        metadata: dict,
    ) -> None: ...

    @abstractmethod
    def compute(self) -> list[MetricResult]: ...

    @abstractmethod
    def reset(self) -> None: ...


class BatchingMetricAccumulator(AbsMetricAccumulator):
    """
    Adapts a legacy Metric to the streaming accumulator interface by buffering all
    per-episode records and constructing an EvaluationBatch on compute().
    """

    def __init__(self, metric: Metric) -> None:
        self._metric = metric
        self._episode_ids: list[str] = []
        self._query_times: list[float] = []
        self._predicted_probs: list[float] = []
        self._ground_truth: list[float] = []
        self._inference_latency: list[float] = []
        self._filter_latency: list[float] = []
        self._metadata: list[dict] = []

    def update(
        self,
        predicted_probs: torch.Tensor,
        ground_truth: torch.Tensor,
        query_times: torch.Tensor,
        num_prior_observations: int,
        prior_observation_timespan_s: float,
        inference_latency_ms: float,
        filter_latency_ms: float,
        metadata: dict,
    ) -> None:
        per_q_lat = filter_latency_ms / max(len(query_times), 1)
        episode_id = f"{metadata.get('track_id', '')}:{metadata.get('anchor_time', '')}"
        for i in range(len(query_times)):
            self._episode_ids.append(episode_id)
            self._query_times.append(query_times[i].item())
            self._predicted_probs.append(predicted_probs[i].item())
            self._ground_truth.append(ground_truth[i].item())
            self._inference_latency.append(inference_latency_ms)
            self._filter_latency.append(per_q_lat)
            self._metadata.append(metadata)

    def compute(self) -> list[MetricResult]:
        if not self._episode_ids:
            return []
        batch = EvaluationBatch(
            episode_ids=self._episode_ids,
            query_times=torch.tensor(self._query_times, dtype=torch.float64),
            predicted_probs=torch.tensor(self._predicted_probs, dtype=torch.float64),
            ground_truth=torch.tensor(self._ground_truth, dtype=torch.float64),
            inference_latency_ms=torch.tensor(self._inference_latency, dtype=torch.float64),
            filter_latency_ms=torch.tensor(self._filter_latency, dtype=torch.float64),
            metadata=self._metadata,
        )
        return self._metric.compute(batch)

    def reset(self) -> None:
        self._episode_ids.clear()
        self._query_times.clear()
        self._predicted_probs.clear()
        self._ground_truth.clear()
        self._inference_latency.clear()
        self._filter_latency.clear()
        self._metadata.clear()
