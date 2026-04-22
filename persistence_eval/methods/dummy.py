from __future__ import annotations

from persistence_eval.core.domain import (
    AnchorObservation,
    BinaryObservationEvent,
    PersistenceParameterEstimate,
)
from persistence_eval.core.protocols import SurvivalFunction
from persistence_eval.methods.inferencers.base import AbsPersistenceParameterInferencer
from persistence_eval.methods.survival.base import AbsSurvivalFunction, AbsSurvivalFunctionGenerator
from persistence_eval.methods.filters.base import AbsPersistenceFilter


class DummyInferencer(AbsPersistenceParameterInferencer):
    """Returns fixed estimates; used in smoke tests to verify runner plumbing."""

    def infer(self, anchor: AnchorObservation) -> PersistenceParameterEstimate:
        return PersistenceParameterEstimate(params={"t50": 60.0, "t90": 300.0})


class _ConstantSurvivalFunction(AbsSurvivalFunction):
    def __call__(self, dt: float) -> float:
        return 0.5


class DummySurvivalFunctionGenerator(AbsSurvivalFunctionGenerator):
    """Returns a constant 0.5 survival function; used in smoke tests."""

    def build(self, estimate: PersistenceParameterEstimate) -> AbsSurvivalFunction:
        return _ConstantSurvivalFunction()


class DummyPersistenceFilter(AbsPersistenceFilter):
    """Returns survival_fn(query_time) directly; used in smoke tests."""

    def predict(
        self,
        survival_fn: SurvivalFunction,
        prior_observations: list[BinaryObservationEvent],
        query_time: float,
    ) -> float:
        return survival_fn(query_time)
