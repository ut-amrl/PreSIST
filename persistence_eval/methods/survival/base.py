from __future__ import annotations

from abc import ABC, abstractmethod

from persistence_eval.core.domain import PersistenceParameterEstimate


class AbsSurvivalFunction(ABC):
    @abstractmethod
    def __call__(self, dt: float) -> float:
        """Return P(object survives past dt seconds from anchor)."""
        ...


class AbsSurvivalFunctionGenerator(ABC):
    @abstractmethod
    def build(self, estimate: PersistenceParameterEstimate) -> AbsSurvivalFunction: ...
