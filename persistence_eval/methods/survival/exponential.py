from __future__ import annotations

import math

import torch
import torch.nn as nn

from persistence_eval.core.domain import PersistenceParameterEstimate
from persistence_eval.core.exceptions import IncompatibleEstimateError
from persistence_eval.methods.survival.base import AbsSurvivalFunction, AbsSurvivalFunctionGenerator


class ExponentialSurvivalFunction(AbsSurvivalFunction):
    """P(T > dt) = exp(-dt / lam), where lam is the mean lifetime."""

    def __init__(self, lam: torch.Tensor) -> None:
        super().__init__()
        self._log_lam = nn.Parameter(torch.log(lam))

    def log_survival(self, dt: torch.Tensor) -> torch.Tensor:
        # log(S(dt)) = -dt / lam
        # Parameterize in log-space so lam remains positive during training.
        lam = torch.exp(self._log_lam)
        return torch.where(dt <= 0, torch.zeros_like(dt), -dt / lam)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.where(dt <= 0, torch.ones_like(dt), torch.exp(self.log_survival(dt)))


class ExponentialSurvivalFunctionGenerator(AbsSurvivalFunctionGenerator):
    """
    Derives the exponential scale parameter lam from either:
      - 'mean': lam = mean
      - 't50':  lam = t50 / ln(2)  (since the median of Exp(lam) is lam * ln(2))
    Checks 'mean' first, then 't50'. Raises IncompatibleEstimateError if neither is present.

    time_scale_factor: multiply the extracted time value before fitting (e.g. 3600.0 to convert hours → seconds).
    """

    def __init__(self, time_scale_factor: float = 1.0, name: str | None = None) -> None:
        self._time_scale_factor = float(time_scale_factor)

    def build(self, estimate: PersistenceParameterEstimate) -> ExponentialSurvivalFunction:
        if "mean" in estimate.params:
            lam = torch.tensor(estimate.params["mean"] * self._time_scale_factor, dtype=torch.float64)
        elif "t50" in estimate.params:
            lam = torch.tensor(estimate.params["t50"] * self._time_scale_factor / math.log(2), dtype=torch.float64)
        else:
            raise IncompatibleEstimateError(
                "ExponentialSurvivalFunctionGenerator requires 'mean' or 't50' in estimate.params; "
                f"got keys: {list(estimate.params)}"
            )
        return ExponentialSurvivalFunction(lam=lam)
