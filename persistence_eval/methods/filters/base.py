from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from persistence_eval.core.domain import BinaryObservationEvent
from persistence_eval.methods.survival.base import AbsSurvivalFunction


class AbsPersistenceFilter(ABC):
    @abstractmethod
    def predict(
        self,
        survival_fn: AbsSurvivalFunction,
        prior_observations: list[BinaryObservationEvent],
        query_times: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return predicted probability that the object exists at anchor_time + query_times[i]
        for each element in query_times (shape (Q,) → shape (Q,)).
        """
        ...
