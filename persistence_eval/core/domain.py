from __future__ import annotations

from dataclasses import dataclass, field

import torch


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
    image: torch.Tensor  # C×H×W, uint8
    mask: torch.Tensor   # H×W, bool
    anchor_time: float  # absolute seconds
    track_id: str       # back-reference to the source track


@dataclass
class ImageMaskGroup:
    """
    A single image frame paired with all object masks visible in that frame.
    Enables batch inference: one forward pass for all objects in a frame rather
    than one pass per object. masks is (K, H, W) bool — stacked so the batch
    is fully tensor-native and compatible with (B, K, 1, H, W) model inputs via
    group.masks.unsqueeze(0).unsqueeze(2).
    """
    image: torch.Tensor    # C×H×W, uint8
    masks: torch.Tensor    # K×H×W bool, one slice per tracked object
    track_ids: list[str]
    anchor_time: float     # absolute seconds (same for all masks — same frame)
    image_path: str | None = None


@dataclass
class EvaluationContext:
    """
    Everything needed to run the persistence filter for one (artifact, query-set) pair.
    Produced by EpisodeProvider.get_contexts() from an InferenceArtifact's track_id and
    anchor_time. Ground truth (bounds) is carried here but never passed to the filter —
    it is only used to compute labels after prediction.
    """
    prior_observations: list[BinaryObservationEvent]   # history strictly before anchor_time
    query_times: list[float]                           # dt from anchor_time (seconds)
    bounds: TrackTemporalBounds
    metadata: dict = field(default_factory=dict)


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
    Persisted output of one inferencer call for one anchor observation.
    Stored to disk so evaluation can be re-run without re-running inference.
    Keyed by (track_id, anchor_time); a dict[(track_id, anchor_time), InferenceArtifact]
    is passed to EvaluationRunner.
    """
    track_id: str
    anchor_time: float
    estimate: PersistenceParameterEstimate
    inference_latency_ms: float



@dataclass
class MetricResult:
    name: str
    value: float
    count: int
    group: dict = field(default_factory=dict)


@dataclass
class EvaluationBatch:
    """
    Flat tensor representation of all prediction records from one evaluation run.
    All tensor fields have shape (N,) and are float64.
    ground_truth is NaN where the label was unknown (TrackTemporalBounds.exists_at returned None).
    metadata is a parallel list of per-record dicts for grouping / reporting.
    """
    episode_ids: list[str]
    query_times: torch.Tensor        # dt from anchor_time (seconds)
    predicted_probs: torch.Tensor
    ground_truth: torch.Tensor       # 0.0 / 1.0; NaN = unknown
    inference_latency_ms: torch.Tensor
    filter_latency_ms: torch.Tensor
    metadata: list[dict]

    def __len__(self) -> int:
        return len(self.episode_ids)

    def labeled_mask(self) -> torch.Tensor:
        """Boolean mask selecting records with known ground truth."""
        return ~torch.isnan(self.ground_truth)
