"""Lognormal survival generator keyed on raw (logmu, std).

The existing log_normal.py generator derives (mu, sigma) from two percentile
keys. The MLE-based single-lognormal trainer emits raw (logmu, std) directly,
so this generator reads them as-is. Names match upstream Perpetua's
`priors.log_survival_lognormal(t, logmu, std)` signature.
"""

from __future__ import annotations

import torch

from persistence_eval.core.domain import PersistenceParameterEstimate
from persistence_eval.core.exceptions import IncompatibleEstimateError
from persistence_eval.methods.survival.base import AbsSurvivalFunctionGenerator
from persistence_eval.methods.survival.log_normal import LogNormalSurvivalFunction


class LogNormalFromMuSigmaGenerator(AbsSurvivalFunctionGenerator):
    """Reads raw lognormal params (logmu, std) from estimate.params and builds
    a LogNormalSurvivalFunction without percentile inversion.

    time_scale_factor: pre-multiply the implied scale before fitting (e.g. 3600.0
    if logmu was fit on hour-scaled durations and the framework uses seconds).
    Applied as `mu_scaled = logmu + log(time_scale_factor)` so the survival
    function is evaluated on second-scaled inputs.
    """

    LOGMU_KEY = "logmu"
    STD_KEY = "std"

    def __init__(self, time_scale_factor: float = 1.0, name: str | None = None) -> None:
        self._time_scale_factor = float(time_scale_factor)

    def build(self, estimate: PersistenceParameterEstimate) -> LogNormalSurvivalFunction:
        for key in (self.LOGMU_KEY, self.STD_KEY):
            if key not in estimate.params:
                raise IncompatibleEstimateError(
                    f"LogNormalFromMuSigmaGenerator requires '{key}' in estimate.params; "
                    f"got keys: {list(estimate.params)}"
                )
        logmu = float(estimate.params[self.LOGMU_KEY])
        std = float(estimate.params[self.STD_KEY])
        if std <= 0:
            raise IncompatibleEstimateError(
                f"LogNormalFromMuSigmaGenerator: std must be > 0; got {std}"
            )
        if self._time_scale_factor != 1.0:
            import math
            logmu = logmu + math.log(self._time_scale_factor)
        mu_t = torch.tensor(logmu, dtype=torch.float64)
        sigma_t = torch.tensor(std, dtype=torch.float64)
        return LogNormalSurvivalFunction(mu=mu_t, sigma=sigma_t)
