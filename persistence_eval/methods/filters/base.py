from __future__ import annotations

from abc import ABC, abstractmethod

from persistence_eval.core.domain import BinaryObservationEvent
from persistence_eval.core.protocols import SurvivalFunction


class AbsPersistenceFilter(ABC):
    @abstractmethod
    def predict(
        self,
        survival_fn: SurvivalFunction,
        prior_observations: list[BinaryObservationEvent],
        query_time: float,
    ) -> float:
        """Return predicted probability that the object exists at anchor_time + query_time."""
        ...
