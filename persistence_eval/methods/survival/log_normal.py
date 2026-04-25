from __future__ import annotations

import torch
import torch.nn as nn

from persistence_eval.core.domain import PersistenceParameterEstimate
from persistence_eval.core.exceptions import IncompatibleEstimateError
from persistence_eval.methods.survival.base import AbsSurvivalFunction, AbsSurvivalFunctionGenerator


class LogNormalSurvivalFunction(AbsSurvivalFunction):
    """P(T > dt) = Phi(-(log(dt) - mu) / sigma), where Phi is the standard normal CDF."""

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        super().__init__()
        self._mu = nn.Parameter(mu)
        self._log_sigma = nn.Parameter(torch.log(sigma))

    def log_survival(self, dt: torch.Tensor) -> torch.Tensor:
        # log S(t) = log(Phi(-z)), where z = (log(t) - mu) / sigma
        # Parameterize sigma in log-space so it remains positive during training.
        sigma = torch.exp(self._log_sigma)
        z = (torch.log(dt.clamp(min=1e-38)) - self._mu) / sigma
        log_S = torch.special.log_ndtr(-z)
        return torch.where(dt <= 0, torch.zeros_like(dt), log_S)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.where(dt <= 0, torch.ones_like(dt), torch.exp(self.log_survival(dt)))


class LogNormalSurvivalFunctionGenerator(AbsSurvivalFunctionGenerator):
    """
    Derives log-normal (mu, sigma) from two percentile estimates via normal quantile fit.
    low_key and high_key name the params required in PersistenceParameterEstimate,
    e.g. low_key="t50", high_key="t90" or low_key="t25", high_key="t75".

    The fit solves:
      z_q = Phi_inv(q)  (where q is the failure quantile, e.g. 0.90 for t90)
      sigma = (log(t_high) - log(t_low)) / (z_high - z_low)
      mu    = log(t_low) - sigma * z_low
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

    def build(self, estimate: PersistenceParameterEstimate) -> LogNormalSurvivalFunction:
        for key in (self._low_key, self._high_key):
            if key not in estimate.params:
                raise IncompatibleEstimateError(
                    f"LogNormalSurvivalFunctionGenerator requires '{key}' in estimate.params; "
                    f"got keys: {list(estimate.params)}"
                )
        t_low = estimate.params[self._low_key] * self._time_scale_factor
        t_high = estimate.params[self._high_key] * self._time_scale_factor

        p_low = self._percentile_from_key(self._low_key)
        p_high = self._percentile_from_key(self._high_key)

        z_low = torch.special.ndtri(torch.tensor(p_low, dtype=torch.float64))
        z_high = torch.special.ndtri(torch.tensor(p_high, dtype=torch.float64))

        log_t_low = torch.log(torch.tensor(t_low, dtype=torch.float64))
        log_t_high = torch.log(torch.tensor(t_high, dtype=torch.float64))

        sigma = (log_t_high - log_t_low) / (z_high - z_low)
        mu = log_t_low - sigma * z_low

        return LogNormalSurvivalFunction(mu=mu, sigma=sigma)

    @staticmethod
    def _percentile_from_key(key: str) -> float:
        """Parse failure quantile from key name, e.g. 't50' → 0.50, 't90' → 0.90."""
        digits = "".join(c for c in key if c.isdigit())
        if not digits:
            raise ValueError(f"Cannot parse percentile from key '{key}'")
        return int(digits) / 100.0
