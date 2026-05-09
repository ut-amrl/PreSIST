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
        anchor_time: float,
        query_times: torch.Tensor,
        observations: list[BinaryObservationEvent] | None = None,
    ) -> torch.Tensor:
        """
        Return predicted probability that the object exists at anchor_time + query_times[i]
        for each element in query_times (shape (Q,) → shape (Q,)).

        anchor_time pins the survival function's reference point: S is evaluated at
        dt = abs_time − anchor_time. observations are absolute-time evidence the
        filter folds in (may be pre- or post-anchor); when None or empty, the filter
        reduces to S(query_times). The caller is responsible for causal masking —
        every observation in the list is consumed for every query, so the episode
        provider must group queries that share an observation prefix before
        invoking the filter.
        """
        ...
