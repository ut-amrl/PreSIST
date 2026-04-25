from __future__ import annotations

import scipy.special
import torch

from persistence_eval.core.domain import PersistenceParameterEstimate
from persistence_eval.core.exceptions import IncompatibleEstimateError
from persistence_eval.methods.survival.base import AbsSurvivalFunction, AbsSurvivalFunctionGenerator


class GeneralPurposeSurvivalFunction(AbsSurvivalFunction):
    """
    Survival function derived from a log-uniform prior over exponential rates in [lambda_l, lambda_u]:

      S(t) = (E_1(t * lambda_l) - E_1(t * lambda_u)) / log(lambda_u / lambda_l)

    where E_1 is the exponential integral. lambda_l < lambda_u must hold.
    See: https://github.com/david-m-rosen/Persistence-Filter
    """

    def __init__(self, lambda_l: torch.Tensor, lambda_u: torch.Tensor) -> None:
        super().__init__()
        if not (lambda_l > 0):
            raise ValueError(f"lambda_l must be positive, got {lambda_l.item()}")
        if not (lambda_u > lambda_l):
            raise ValueError(
                f"lambda_u ({lambda_u.item()}) must be strictly greater than lambda_l ({lambda_l.item()})"
            )
        self.register_buffer("_lambda_l", lambda_l)
        self.register_buffer("_lambda_u", lambda_u)
        self.register_buffer("_log_denom", torch.log(torch.log(lambda_u / lambda_l)))

    def log_survival(self, dt: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(dt)
        mask = dt > 0
        if not mask.any():
            return out

        dt_pos = dt[mask]

        # E_1 is decreasing, so E_1(t*lambda_l) > E_1(t*lambda_u) since lambda_l < lambda_u
        e1_l = torch.tensor(
            scipy.special.exp1((dt_pos * self._lambda_l).detach().cpu().numpy()),
            dtype=dt.dtype, device=dt.device,
        )
        e1_u = torch.tensor(
            scipy.special.exp1((dt_pos * self._lambda_u).detach().cpu().numpy()),
            dtype=dt.dtype, device=dt.device,
        )

        log_e1_l = torch.log(e1_l)
        log_e1_u = torch.log(e1_u)

        # log(E_1(t*lambda_l) - E_1(t*lambda_u)) computed stably via log1p
        log_numerator = log_e1_l + torch.log1p(-torch.exp(log_e1_u - log_e1_l))
        out[mask] = log_numerator - self._log_denom
        return out

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        out = torch.ones_like(dt)
        mask = dt > 0
        if mask.any():
            out[mask] = torch.exp(self.log_survival(dt)[mask])
        return out


class GeneralPurposeSurvivalFunctionGenerator(AbsSurvivalFunctionGenerator):
    """Builds a GeneralPurposeSurvivalFunction from 'lambda_l' and 'lambda_u' in the estimate."""

    def build(self, estimate: PersistenceParameterEstimate) -> GeneralPurposeSurvivalFunction:
        for key in ("lambda_l", "lambda_u"):
            if key not in estimate.params:
                raise IncompatibleEstimateError(
                    f"GeneralPurposeSurvivalFunctionGenerator requires '{key}' in estimate.params; "
                    f"got keys: {list(estimate.params)}"
                )
        lambda_l = estimate.params["lambda_l"]
        lambda_u = estimate.params["lambda_u"]
        if lambda_l >= lambda_u:
            raise IncompatibleEstimateError(
                f"lambda_l ({lambda_l}) must be strictly less than lambda_u ({lambda_u})"
            )
        return GeneralPurposeSurvivalFunction(
            lambda_l=torch.tensor(lambda_l, dtype=torch.float64),
            lambda_u=torch.tensor(lambda_u, dtype=torch.float64),
        )
