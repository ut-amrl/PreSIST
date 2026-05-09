from __future__ import annotations

import torch

from persistence_eval.core.domain import (
    AnchorObservation,
    BinaryObservationEvent,
    PersistenceParameterEstimate,
)
from persistence_eval.methods.inferencers.base import AbsPersistenceParameterInferencer
from persistence_eval.methods.survival.base import AbsSurvivalFunction, AbsSurvivalFunctionGenerator
from persistence_eval.methods.filters.base import AbsPersistenceFilter


class DummyInferencer(AbsPersistenceParameterInferencer):
    """Returns fixed estimates; used in smoke tests to verify runner plumbing."""

    def infer(self, anchor: AnchorObservation) -> PersistenceParameterEstimate:
        return PersistenceParameterEstimate(params={"t50": 60.0, "t90": 300.0})


class _ConstantSurvivalFunction(AbsSurvivalFunction):
    """Returns 0.5 for all dt > 0, 1.0 for dt <= 0."""

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.where(dt <= 0, torch.ones_like(dt), torch.full_like(dt, 0.5))

    def log_survival(self, dt: torch.Tensor) -> torch.Tensor:
        import math
        log_half = math.log(0.5)
        return torch.where(dt <= 0, torch.zeros_like(dt), torch.full_like(dt, log_half))


class DummySurvivalFunctionGenerator(AbsSurvivalFunctionGenerator):
    """Returns a constant 0.5 survival function; used in smoke tests."""

    def build(self, estimate: PersistenceParameterEstimate) -> AbsSurvivalFunction:
        return _ConstantSurvivalFunction()


class DummyPersistenceFilter(AbsPersistenceFilter):
    """Returns survival_fn(query_times) directly; used in smoke tests."""

    def predict(
        self,
        survival_fn: AbsSurvivalFunction,
        anchor_time: float,
        query_times: torch.Tensor,
        observations: list[BinaryObservationEvent] | None = None,
    ) -> torch.Tensor:
        del anchor_time, observations
        return survival_fn(query_times)
