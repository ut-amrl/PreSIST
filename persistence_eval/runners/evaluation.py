from __future__ import annotations

import time

from persistence_eval.core.domain import (
    InferenceArtifact,
    MetricResult,
    PersistenceFilterInput,
    PredictionRecord,
    TrackTemporalBounds,
)
from persistence_eval.core.protocols import Metric
from persistence_eval.methods.evaluation_method import EvaluationMethod


class EvaluationRunner:
    """
    Runs the evaluation pipeline over pre-computed inference artifacts.
    Accepts EvaluationMethod (not BenchmarkMethod) so inference and evaluation
    remain fully decoupled — artifacts can come from disk or a prior InferenceRunner run.
    """

    def run(
        self,
        episode_pairs: list[tuple[PersistenceFilterInput, TrackTemporalBounds]],
        artifacts: dict[str, InferenceArtifact],
        method: EvaluationMethod,
        metrics: list[Metric],
    ) -> tuple[list[PredictionRecord], list[MetricResult]]:
        records: list[PredictionRecord] = []

        for episode, bounds in episode_pairs:
            artifact = artifacts.get(episode.episode_id)
            if artifact is None:
                continue

            survival_fn = method.survival_generator.build(artifact.estimate)

            for query_time in episode.query_times:
                t0 = time.perf_counter()
                predicted_prob = method.persistence_filter.predict(
                    survival_fn=survival_fn,
                    prior_observations=episode.prior_observations,
                    query_time=query_time,
                )
                filter_latency_ms = (time.perf_counter() - t0) * 1000.0

                abs_time = episode.anchor.anchor_time + query_time
                ground_truth = bounds.exists_at(abs_time)

                records.append(PredictionRecord(
                    episode_id=episode.episode_id,
                    query_time=query_time,
                    predicted_prob=predicted_prob,
                    ground_truth=ground_truth,
                    inference_latency_ms=artifact.inference_latency_ms,
                    filter_latency_ms=filter_latency_ms,
                    metadata={
                        "method_name": method.name,
                        **episode.metadata,
                    },
                ))

        metric_results: list[MetricResult] = []
        for metric in metrics:
            metric_results.extend(metric.compute(records))

        return records, metric_results
