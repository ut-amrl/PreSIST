from __future__ import annotations

import math

from persistence_eval.core.domain import BinaryObservationEvent
from persistence_eval.core.protocols import SurvivalFunction
from persistence_eval.methods.filters.base import AbsPersistenceFilter

_NEG_INF = float("-inf")


def _safe_log(x: float) -> float:
    return math.log(x) if x > 0.0 else _NEG_INF


def _logsum(lx: float, ly: float) -> float:
    """Numerically stable log(exp(lx) + exp(ly))."""
    if lx == _NEG_INF:
        return ly
    if ly == _NEG_INF:
        return lx
    if lx < ly:
        lx, ly = ly, lx
    return lx + math.log1p(math.exp(ly - lx))


def _logdiff(lx: float, ly: float) -> float:
    """Numerically stable log(exp(lx) - exp(ly)); requires lx >= ly."""
    if ly == _NEG_INF:
        return lx
    diff = ly - lx
    if diff >= 0.0:
        return _NEG_INF
    return lx + math.log1p(-math.exp(diff))


class RosenPersistenceFilter(AbsPersistenceFilter):
    """
    Bayesian persistence filter per Rosen et al., "Towards Lifelong Feature-Based
    Mapping in Semi-Static Environments," ICRA 2016.

    Computes P(X_t = 1 | Y_{1:N}) by marginalising over all times at which the
    feature could have disappeared:

        P(alive at t | Y_{1:N}) = L(Y_{1:N}) * S(t) / p(Y_{1:N})

    where L is the accumulated detector likelihood assuming continuous survival,
    and p(Y_{1:N}) sums over all possible disappearance epochs weighted by the
    survival-time prior S.

    The survival function is anchored at the anchor observation (dt = 0).
    Prior observations are processed chronologically; their timestamps are
    interpreted relative to the last prior observation, which is used as a
    proxy for the anchor when computing survival-function intervals.  For
    standard survival functions that return 1.0 for dt ≤ 0 (object certain to
    exist before anchor), survival mass in pre-anchor intervals is zero, so the
    filter correctly reduces to S(query_time) — observations only alter the
    result if the survival function assigns non-trivial mass to historical
    intervals.
    """

    def __init__(self, false_detection_prob: float, missed_detection_prob: float) -> None:
        self._p_false = false_detection_prob
        self._p_miss = missed_detection_prob

    def predict(
        self,
        survival_fn: SurvivalFunction,
        prior_observations: list[BinaryObservationEvent],
        query_time: float,
    ) -> float:
        if not prior_observations:
            return survival_fn(query_time)

        log_p_det_alive = _safe_log(1.0 - self._p_miss)   # log P(detect | alive)
        log_p_miss_alive = _safe_log(self._p_miss)         # log P(miss   | alive)
        log_p_det_dead = _safe_log(self._p_false)          # log P(detect | dead)
        log_p_miss_dead = _safe_log(1.0 - self._p_false)   # log P(miss   | dead)

        obs = sorted(prior_observations, key=lambda o: o.timestamp)

        # Use the last observation timestamp as the anchor proxy (dt = 0).
        # All earlier observations have dt < 0, so survival_fn returns 1.0 for
        # them; the last observation is at dt = 0, also giving S = 1.0.
        anchor_proxy = obs[-1].timestamp

        def log_S(abs_time: float) -> float:
            s = survival_fn(abs_time - anchor_proxy)
            return _safe_log(s)

        # --- Rosen filter state ---
        log_likelihood = 0.0       # log p(Y_{1:k} | alive through all obs)
        log_lower_sum = _NEG_INF   # log L(Y_{1:k}): sum over epochs where feature died
        last_log_S = _NEG_INF      # log S at previous observation time

        for idx, o in enumerate(obs):
            cur_log_S = log_S(o.timestamp)
            log_obs_alive = log_p_det_alive if o.is_positive else log_p_miss_alive
            log_obs_dead = log_p_det_dead if o.is_positive else log_p_miss_dead

            if idx == 0:
                # Mass of survival prior that has expired before first observation:
                # log(1 - S(t_1)).  With S(t_1) = 1 this is log(0) = -inf.
                log_mass_before_first = _logdiff(_safe_log(1.0), cur_log_S)
                log_lower_sum = log_obs_dead + log_mass_before_first
            else:
                # Mass in interval (t_{k-1}, t_k]: log(S(t_{k-1}) - S(t_k))
                log_interval_mass = _logdiff(last_log_S, cur_log_S)
                log_lower_sum = _logsum(
                    log_lower_sum,
                    log_likelihood + log_interval_mass,
                ) + log_obs_dead

            log_likelihood += log_obs_alive
            last_log_S = cur_log_S

        # Evidence: P(dies after t_N) + sum of P(dies in each past interval)
        log_evidence = _logsum(log_lower_sum, log_likelihood + last_log_S)

        # Posterior at query_time: P(alive at query) ∝ L * S(query_time)
        log_query_S = _safe_log(survival_fn(query_time))
        log_posterior = log_likelihood + log_query_S - log_evidence

        return math.exp(log_posterior)
