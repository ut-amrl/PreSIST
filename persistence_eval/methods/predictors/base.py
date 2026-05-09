from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from persistence_eval.core.domain import BinaryObservationEvent, InferenceArtifact


class AbsPersistencePredictor(ABC):
    """
    Top-level abstraction owning the artifact -> per-query-time probability step.

    Subsumes the (SurvivalFunctionGenerator, AbsPersistenceFilter) pair by giving
    implementations access to the full InferenceArtifact, so per-instance context
    that doesn't fit a parametric survival form (e.g. raw image references for a
    direct-VLM query) can flow through artifact.estimate.metadata.
    """

    @abstractmethod
    def predict(
        self,
        artifact: InferenceArtifact,
        query_times: torch.Tensor,
        observations: list[BinaryObservationEvent] | None = None,
    ) -> torch.Tensor:
        """Return predicted P(exists at anchor + query_times[i]); shape (Q,) -> (Q,).

        observations is the (post-anchor) evidence visible to every query in this
        call; the caller (episode provider) handles causal grouping.
        """
        ...
