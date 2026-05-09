from __future__ import annotations

import torch

from persistence_eval.core.domain import BinaryObservationEvent, InferenceArtifact
from persistence_eval.methods.filters.base import AbsPersistenceFilter
from persistence_eval.methods.predictors.base import AbsPersistencePredictor
from persistence_eval.methods.survival.base import AbsSurvivalFunctionGenerator


class SurvivalBasedPredictor(AbsPersistencePredictor):
    """
    Composes a SurvivalFunctionGenerator with an AbsPersistenceFilter — the
    pre-existing two-stage pipeline, adapted to the AbsPersistencePredictor
    interface. Both inner abstractions remain independently swappable.
    """

    def __init__(
        self,
        survival_generator: AbsSurvivalFunctionGenerator,
        persistence_filter: AbsPersistenceFilter,
        name: str | None = None,
    ) -> None:
        self._survival_generator = survival_generator
        self._persistence_filter = persistence_filter
        self.name = name

    def predict(
        self,
        artifact: InferenceArtifact,
        query_times: torch.Tensor,
        observations: list[BinaryObservationEvent] | None = None,
    ) -> torch.Tensor:
        survival_fn = self._survival_generator.build(artifact.estimate)
        return self._persistence_filter.predict(
            survival_fn,
            artifact.anchor_time,
            query_times,
            observations,
        )
