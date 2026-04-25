from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

import torch

from persistence_eval.core.domain import (
    AnchorObservation,
    BinaryObservationEvent,
    EvaluationBatch,
    EvaluationContext,
    ImageMaskGroup,
    InferenceArtifact,
    MetricResult,
    PersistenceParameterEstimate,
    TrackTemporalBounds,
)


class BackingDataset(Protocol):
    """Raw data access for a dataset — the underlying source for both dataset views."""
    name: str


class ImageMaskDataset(Protocol):
    """View of a BackingDataset providing AnchorObservations for inferencer input."""
    name: str

    def iter_anchor_observations(self) -> Iterable[AnchorObservation]: ...
    def iter_image_groups(self) -> Iterable[ImageMaskGroup]: ...


class TrackDataset(Protocol):
    """View of a BackingDataset providing per-track observation sequences."""
    name: str

    def iter_tracks(self) -> Iterable[tuple[str, list[BinaryObservationEvent], TrackTemporalBounds]]: ...


class PersistenceParameterInferencer(Protocol):
    """
    Estimates persistence parameters for all objects in an ImageMaskGroup.
    Returns one PersistenceParameterEstimate per mask (same order as group.track_ids).
    """
    def infer(self, group: ImageMaskGroup) -> list[PersistenceParameterEstimate]: ...


class EpisodeProvider(Protocol):
    """
    Given a (track_id, anchor_time) from an InferenceArtifact, returns one or more
    EvaluationContexts supplying prior observations, query horizons, and ground truth.
    Multiple contexts per artifact can represent different prediction horizon sets.
    """
    def get_contexts(self, track_id: str, anchor_time: float) -> Iterable[EvaluationContext]: ...


class SurvivalFunction(Protocol):
    """
    Tensor-native survival function: S(dt) = P(object survives past dt seconds from anchor).
    dt <= 0 must return 1.0 (object certain to exist at or before anchor).
    """
    def __call__(self, dt: torch.Tensor) -> torch.Tensor: ...
    def log_survival(self, dt: torch.Tensor) -> torch.Tensor: ...


class SurvivalFunctionGenerator(Protocol):
    def build(self, estimate: PersistenceParameterEstimate) -> SurvivalFunction: ...


class PersistenceFilter(Protocol):
    def predict(
        self,
        survival_fn: SurvivalFunction,
        prior_observations: list[BinaryObservationEvent],
        query_times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns predicted_prob for each query_time (shape (Q,) → shape (Q,)).
        query_times are dt in seconds from anchor_time.
        """
        ...


class MetricAccumulator(Protocol):
    """
    Streaming metric accumulator: update() is called once per (artifact, context) pair,
    compute() returns final MetricResults after all updates. Accumulators may stratify
    results by query horizon, prior observation count/span, or metadata keys by returning
    multiple MetricResults with different MetricResult.group dicts.
    """
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

    def compute(self) -> list[MetricResult]: ...
    def reset(self) -> None: ...


class Metric(Protocol):
    """Legacy batch-at-once metric. Wrap with BatchingMetricAccumulator for the new pipeline."""
    def compute(self, batch: EvaluationBatch) -> list[MetricResult]: ...


class MetricsAnalyzer(Protocol):
    def analyze(
        self,
        metrics: list[MetricResult],
        batch: EvaluationBatch,
        output_dir: Path,
    ) -> None: ...


class ArtifactStore(Protocol):
    def save(self, artifacts: list[InferenceArtifact], run_dir: Path) -> None: ...
    def append(self, artifacts: list[InferenceArtifact], run_dir: Path) -> None: ...
    def load(self, run_dir: Path) -> list[InferenceArtifact]: ...
