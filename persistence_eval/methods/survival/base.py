from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from persistence_eval.core.domain import PersistenceParameterEstimate


class AbsSurvivalFunction(nn.Module, ABC):
    @abstractmethod
    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        """Return P(object survives past dt seconds from anchor)."""
        ...

    @abstractmethod
    def log_survival(self, dt: torch.Tensor) -> torch.Tensor:
        """Return log(P(object survives past dt seconds from anchor))."""
        ...


class AbsSurvivalFunctionGenerator(ABC):
    @abstractmethod
    def build(self, estimate: PersistenceParameterEstimate) -> AbsSurvivalFunction: ...
