from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


@dataclass
class BinaryObservationEvent:
    """A single observation of an object's presence/absence at a point in time."""
    timestamp: float  # seconds (absolute)
    is_positive: bool


@dataclass
class TrackTemporalBounds:
    """
    Ground-truth temporal extent of an object track.
    Captures both ends of uncertainty: confirmed absence before first observation
    (if known) and confirmed disappearance after last observation (if observed).
    Not an input to the persistence filter — used only for computing ground truth labels.
    """
    first_observed_time: float
    last_observed_time: float
    first_absent_before_observation: float | None  # None → pre-observation absence unknown
    first_missing_time: float | None               # None → disappearance not observed

    def exists_at(self, t: float) -> bool | None:
        """
        Returns True if t is within the confirmed existence window,
        False if t is in a confirmed absence window, None if label is unknown.
        """
        if self.first_observed_time <= t <= self.last_observed_time:
            return True
        if self.first_absent_before_observation is not None and t <= self.first_absent_before_observation:
            return False
        if self.first_missing_time is not None and t >= self.first_missing_time:
            return False
        return None


@dataclass
class AnchorObservation:
    """
    A single image frame + segmentation mask for a tracked object at a specific time.
    This is the sole input to the persistence parameter inferencer: given this
    visual snapshot, the inferencer predicts how long the object is likely to persist.
    anchor_time is the reference point from which query_times (dt) are measured.
    """
    image: np.ndarray   # H×W×3, uint8
    mask: np.ndarray    # H×W, bool
    anchor_time: float  # absolute seconds
    track_id: str       # back-reference to the source track


@dataclass
class PersistenceFilterInput:
    """
    Inputs needed to run the persistence filter for one (track, anchor, query-set) tuple.
    Constructed from track data + a sampled anchor; one track may yield multiple inputs.
    Contains only filter inputs — ground truth (TrackTemporalBounds) is kept separate
    and paired with this object by the sampler so filter logic stays decoupled from labels.
    """
    episode_id: str                                        # "{dataset.name}:{track_id}:{anchor_idx}"
    anchor: AnchorObservation
    prior_observations: list[BinaryObservationEvent]       # history before anchor_time
    query_times: list[float]                               # dt from anchor_time (seconds)
    metadata: dict = field(default_factory=dict)           # dataset_name, split, etc.


@dataclass
class PersistenceParameterEstimate:
    """
    Generic key→time mapping output by an inferencer, e.g.:
      {"t50": 100.0, "t90": 300.0}  — two-percentile Weibull fit
      {"mean": 80.0}                — exponential (mean lifetime) fit
    Survival generators declare which keys they require and raise
    IncompatibleEstimateError if keys are absent.
    """
    params: dict[str, float]
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class InferenceArtifact:
    """
    Persisted output of one inferencer call for one episode.
    Stored to disk so evaluation can be re-run without re-running inference.
    Indexed by episode_id; a dict[episode_id, InferenceArtifact] is passed to EvaluationRunner.
    """
    episode_id: str
    estimate: PersistenceParameterEstimate
    inference_latency_ms: float


@dataclass
class PredictionRecord:
    """
    One filter prediction for one (episode, query_time) pair.
    Links filter output to ground truth for metric computation.
    ground_truth=None means the label is unknown; this record is skipped in metrics.
    """
    episode_id: str
    query_time: float          # dt from anchor_time
    predicted_prob: float
    ground_truth: bool | None  # derived from TrackTemporalBounds.exists_at(); not a filter input
    inference_latency_ms: float
    filter_latency_ms: float
    metadata: dict = field(default_factory=dict)  # method_name, dataset_name, split, horizon, etc.


@dataclass
class MetricResult:
    name: str
    value: float
    count: int
    group: dict = field(default_factory=dict)
