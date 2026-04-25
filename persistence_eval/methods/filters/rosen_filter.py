from __future__ import annotations

import math

import torch

from persistence_eval.core.domain import BinaryObservationEvent
from persistence_eval.methods.filters.base import AbsPersistenceFilter
from persistence_eval.methods.survival.base import AbsSurvivalFunction

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

    predict() accepts a batch of query_times as a tensor. The prior-observation
    loop runs once per episode (independent of query_times), and the final
    posterior is computed for all query_times in a single tensor operation.
    """

    def __init__(self, false_detection_prob: float, missed_detection_prob: float) -> None:
        self._p_false = false_detection_prob
        self._p_miss = missed_detection_prob

    def predict(
        self,
        survival_fn: AbsSurvivalFunction,
        prior_observations: list[BinaryObservationEvent],
        query_times: torch.Tensor,
    ) -> torch.Tensor:
        if not prior_observations:
            return survival_fn(query_times)

        log_p_det_alive = _safe_log(1.0 - self._p_miss)
        log_p_miss_alive = _safe_log(self._p_miss)
        log_p_det_dead = _safe_log(self._p_false)
        log_p_miss_dead = _safe_log(1.0 - self._p_false)

        obs = sorted(prior_observations, key=lambda o: o.timestamp)
        anchor_proxy = obs[-1].timestamp

        def _log_S_scalar(abs_time: float) -> float:
            dt = torch.tensor(abs_time - anchor_proxy, dtype=query_times.dtype)
            return survival_fn.log_survival(dt.unsqueeze(0)).item()

        # --- Rosen filter state (scalars, observation loop runs once per episode) ---
        log_likelihood = 0.0
        log_lower_sum = _NEG_INF
        last_log_S = _NEG_INF

        for idx, o in enumerate(obs):
            cur_log_S = _log_S_scalar(o.timestamp)
            log_obs_alive = log_p_det_alive if o.is_positive else log_p_miss_alive
            log_obs_dead = log_p_det_dead if o.is_positive else log_p_miss_dead

            if idx == 0:
                log_mass_before_first = _logdiff(_safe_log(1.0), cur_log_S)
                log_lower_sum = log_obs_dead + log_mass_before_first
            else:
                log_interval_mass = _logdiff(last_log_S, cur_log_S)
                log_lower_sum = _logsum(
                    log_lower_sum,
                    log_likelihood + log_interval_mass,
                ) + log_obs_dead

            log_likelihood += log_obs_alive
            last_log_S = cur_log_S

        log_evidence = _logsum(log_lower_sum, log_likelihood + last_log_S)

        # Vectorized final step: one log_survival call for all query_times at once.
        log_query_S = survival_fn.log_survival(query_times)   # shape (Q,)
        log_posteriors = log_likelihood + log_query_S - log_evidence
        return torch.exp(log_posteriors).clamp(0.0, 1.0)
