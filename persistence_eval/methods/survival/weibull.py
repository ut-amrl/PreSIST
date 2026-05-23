from __future__ import annotations

import math

import torch
import torch.nn as nn

from persistence_eval.core.domain import PersistenceParameterEstimate
from persistence_eval.core.exceptions import IncompatibleEstimateError
from persistence_eval.methods.survival.base import AbsSurvivalFunction, AbsSurvivalFunctionGenerator


class WeibullSurvivalFunction(AbsSurvivalFunction):
    """P(T > dt) = exp(-(dt / lam) ^ k)"""

    def __init__(self, k: torch.Tensor, lam: torch.Tensor) -> None:
        super().__init__()
        self._log_k = nn.Parameter(torch.log(k))
        self._log_lam = nn.Parameter(torch.log(lam))

    def log_survival(self, dt: torch.Tensor) -> torch.Tensor:
        # log(S(dt)) = -(dt/lam)^k = -exp(k * log(dt/lam))
        # Parameterize in log-space so k and lam remain positive during training.
        k = torch.exp(self._log_k)
        log_dt_over_lam = torch.log(dt.clamp(min=1e-38)) - self._log_lam
        log_S = -torch.exp(k * log_dt_over_lam)
        return torch.where(dt <= 0, torch.zeros_like(dt), log_S)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.where(dt <= 0, torch.ones_like(dt), torch.exp(self.log_survival(dt)))


def weibull_median_from_quantiles(
    t_low: float,
    t_high: float,
    p_low: float = 0.25,
    p_high: float = 0.75,
) -> float:
    """Median disappearance time for a Weibull fit through two failure quantiles.

    Defaults are p25/p75. Returns the time at which S(t) = 0.5, i.e. half of
    objects have disappeared. Times are in whatever unit t_low/t_high are given in.
    """
    if not (0.0 < p_low < p_high < 1.0):
        raise ValueError(f"need 0 < p_low < p_high < 1; got {p_low}, {p_high}")
    if not (0.0 < t_low < t_high):
        raise ValueError(f"need 0 < t_low < t_high; got {t_low}, {t_high}")
    log_tlow, log_thigh = math.log(t_low), math.log(t_high)
    a_low = math.log(-math.log1p(-p_low))
    a_high = math.log(-math.log1p(-p_high))
    k = (a_high - a_low) / (log_thigh - log_tlow)
    log_lam = log_tlow - a_low / k
    # median: S(t)=0.5 -> t = lam * (ln 2)^(1/k)
    return math.exp(log_lam) * math.log(2.0) ** (1.0 / k)


class WeibullSurvivalFunctionGenerator(AbsSurvivalFunctionGenerator):
    """
    Derives Weibull (k, lam) from two percentile estimates via log-log fit.
    low_key and high_key name the params required in PersistenceParameterEstimate,
    e.g. low_key="t50", high_key="t90" or low_key="t25", high_key="t75".
    """

    def __init__(
        self,
        low_key: str = "t50",
        high_key: str = "t90",
        time_scale_factor: float = 1.0,
        name: str | None = None,
    ) -> None:
        self._low_key = low_key
        self._high_key = high_key
        self._time_scale_factor = float(time_scale_factor)

    def build(self, estimate: PersistenceParameterEstimate) -> WeibullSurvivalFunction:
        for key in (self._low_key, self._high_key):
            if key not in estimate.params:
                raise IncompatibleEstimateError(
                    f"WeibullSurvivalFunctionGenerator requires '{key}' in estimate.params; "
                    f"got keys: {list(estimate.params)}"
                )
        t_low = estimate.params[self._low_key] * self._time_scale_factor
        t_high = estimate.params[self._high_key] * self._time_scale_factor

        p_low = self._percentile_from_key(self._low_key)
        p_high = self._percentile_from_key(self._high_key)

        log_tlow = torch.log(torch.tensor(t_low))
        log_thigh = torch.log(torch.tensor(t_high))

        doubleLogPLow = torch.log(-torch.log1p(-torch.tensor(p_low)))
        doubleLogPHigh = torch.log(-torch.log1p(-torch.tensor(p_high)))

        k = (doubleLogPHigh - doubleLogPLow) / (log_thigh - log_tlow)
        logLambda = log_tlow - doubleLogPLow / k
        lambdaVal = torch.exp(logLambda)

        return WeibullSurvivalFunction(k=k, lam=lambdaVal)

    @staticmethod
    def _percentile_from_key(key: str) -> float:
        """Parse failure quantile from key name, e.g. 't50' → 0.50, 't90' → 0.90."""
        digits = "".join(c for c in key if c.isdigit())
        if not digits:
            raise ValueError(f"Cannot parse percentile from key '{key}'")
        return int(digits) / 100.0
