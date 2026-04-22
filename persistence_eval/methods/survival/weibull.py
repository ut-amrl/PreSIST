from __future__ import annotations

import math

from persistence_eval.core.domain import PersistenceParameterEstimate
from persistence_eval.core.exceptions import IncompatibleEstimateError
from persistence_eval.methods.survival.base import AbsSurvivalFunction, AbsSurvivalFunctionGenerator


class WeibullSurvivalFunction(AbsSurvivalFunction):
    """P(T > dt) = exp(-(dt / lam) ^ k)"""

    def __init__(self, k: float, lam: float) -> None:
        self._k = k
        self._lam = lam

    def __call__(self, dt: float) -> float:
        if dt <= 0:
            return 1.0
        return math.exp(-((dt / self._lam) ** self._k))


class WeibullSurvivalFunctionGenerator(AbsSurvivalFunctionGenerator):
    """
    Derives Weibull (k, lam) from two percentile estimates via log-log fit.
    low_key and high_key name the params required in PersistenceParameterEstimate,
    e.g. low_key="t50", high_key="t90" or low_key="t25", high_key="t75".
    """

    def __init__(self, low_key: str = "t50", high_key: str = "t90") -> None:
        self._low_key = low_key
        self._high_key = high_key

    def build(self, estimate: PersistenceParameterEstimate) -> WeibullSurvivalFunction:
        for key in (self._low_key, self._high_key):
            if key not in estimate.params:
                raise IncompatibleEstimateError(
                    f"WeibullSurvivalFunctionGenerator requires '{key}' in estimate.params; "
                    f"got keys: {list(estimate.params)}"
                )
        t_low = estimate.params[self._low_key]
        t_high = estimate.params[self._high_key]

        # Derive survival probabilities from key names (e.g. "t50" → 0.50 remaining)
        p_low = self._percentile_from_key(self._low_key)
        p_high = self._percentile_from_key(self._high_key)

        # log-log fit: log(-log(S(t))) = k * log(t) - k * log(lam)
        # S(t) = exp(-(t/lam)^k) → -log(S) = (t/lam)^k → log(-log(S)) = k*log(t/lam)
        ll_low = math.log(-math.log(p_low))
        ll_high = math.log(-math.log(p_high))
        k = (ll_high - ll_low) / (math.log(t_high) - math.log(t_low))
        lam = t_low * math.exp(-ll_low / k)
        return WeibullSurvivalFunction(k=k, lam=lam)

    @staticmethod
    def _percentile_from_key(key: str) -> float:
        """Parse survival probability from key name, e.g. 't50' → 0.50, 't90' → 0.10."""
        digits = "".join(c for c in key if c.isdigit())
        if not digits:
            raise ValueError(f"Cannot parse percentile from key '{key}'")
        pct = int(digits) / 100.0
        # tXX means XX% of objects have disappeared → XX% survival remaining
        return 1.0 - pct
