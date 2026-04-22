from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from persistence_eval.core.domain import (
    AnchorObservation,
    BinaryObservationEvent,
    InferenceArtifact,
    MetricResult,
    PersistenceFilterInput,
    PersistenceParameterEstimate,
    PredictionRecord,
    TrackTemporalBounds,
)


class BackingDataset(Protocol):
    """Raw data access for a dataset — the underlying source for both dataset views."""
    name: str


class ImageMaskDataset(Protocol):
    """View of a BackingDataset providing AnchorObservations for inferencer input."""
    name: str

    def iter_anchor_observations(self) -> Iterable[AnchorObservation]: ...


class TrackDataset(Protocol):
    """View of a BackingDataset providing per-track observation sequences for episode sampling."""
    name: str

    def iter_tracks(self) -> Iterable[tuple[str, list[BinaryObservationEvent], TrackTemporalBounds]]: ...


class EpisodeSampler(Protocol):
    """Samples (filter_input, ground_truth) pairs from a TrackDataset."""

    def sample(
        self, dataset: TrackDataset
    ) -> list[tuple[PersistenceFilterInput, TrackTemporalBounds]]: ...


class PersistenceParameterInferencer(Protocol):
    def infer(self, anchor: AnchorObservation) -> PersistenceParameterEstimate: ...

    def infer_batch(
        self,
        image: np.ndarray,
        masks: list[np.ndarray],
        anchor_time: float,
        track_ids: list[str],
    ) -> list[PersistenceParameterEstimate]:
        """Infer for multiple masks on the same image in a single forward pass."""
        ...


class SurvivalFunction(Protocol):
    def __call__(self, dt: float) -> float: ...  # P(survive past dt)


class SurvivalFunctionGenerator(Protocol):
    def build(self, estimate: PersistenceParameterEstimate) -> SurvivalFunction: ...


class PersistenceFilter(Protocol):
    def predict(
        self,
        survival_fn: SurvivalFunction,
        prior_observations: list[BinaryObservationEvent],
        query_time: float,
    ) -> float: ...  # returns predicted_prob; latency is measured by the runner


class Metric(Protocol):
    def compute(self, records: list[PredictionRecord]) -> list[MetricResult]: ...


class MetricsAnalyzer(Protocol):
    def analyze(
        self,
        metrics: list[MetricResult],
        records: list[PredictionRecord],
        output_dir: Path,
    ) -> None: ...


class ArtifactStore(Protocol):
    def save(self, artifacts: list[InferenceArtifact], run_dir: Path) -> None: ...
    def load(self, run_dir: Path) -> list[InferenceArtifact]: ...
