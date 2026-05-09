"""Mixture persistence filter — thin wrapper around upstream Perpetua.

Wraps src.filters.MixturePersistenceFilters.{init,update,predict} from the
vendored upstream package. Translates our (MixtureSurvivalFunction, anchor_time,
query times, observations) into upstream's (FilterState, observation sequence,
query array), runs the filter, and returns a torch tensor.

Returns the marginal posterior across all components at each query time:

    P(alive at t | obs) = Σ_k τ_k(N) · P(alive at t | k, obs)

where τ_k(N) is the per-component posterior weight after the N observations
are folded in (`state.log_tau`). With no observations τ_k = π_k and the result
collapses to the mixture prior survival.

The survival function is anchored at `anchor_time` (passed in), not at any
observation timestamp. Caller is responsible for causal masking — every
observation in the list is consumed for every query.

The filter accepts only MixtureSurvivalFunction; passing a single-component
function raises IncompatibleEstimateError.
"""

from __future__ import annotations

import torch

from persistence_eval.core.domain import BinaryObservationEvent
from persistence_eval.core.exceptions import IncompatibleEstimateError
from persistence_eval.methods.filters.base import AbsPersistenceFilter
from persistence_eval.methods.filters.perpetua_filter_adapter import (
    _jax_modules,
    jnp_to_torch,
)
from persistence_eval.methods.survival.base import AbsSurvivalFunction
from persistence_eval.methods.survival.mixture import MixtureSurvivalFunction


class MixturePersistenceFilter(AbsPersistenceFilter):
    """Wraps upstream MixturePersistenceFilters."""

    def __init__(
        self,
        false_detection_prob: float = 0.05,
        missed_detection_prob: float = 0.1,
        name: str | None = None,
    ) -> None:
        self.p_f = float(false_detection_prob)
        self.p_m = float(missed_detection_prob)

    def predict(
        self,
        survival_fn: AbsSurvivalFunction,
        anchor_time: float,
        query_times: torch.Tensor,
        observations: list[BinaryObservationEvent] | None = None,
    ) -> torch.Tensor:
        if not isinstance(survival_fn, MixtureSurvivalFunction):
            raise IncompatibleEstimateError(
                "MixturePersistenceFilter requires a MixtureSurvivalFunction; "
                f"got {type(survival_fn).__name__}"
            )

        mods = _jax_modules()
        jnp = mods["jnp"]
        mix_filter = mods["mixture_filter"]

        log_s, params, pi = survival_fn.build_jax_state()

        anchor_time = float(anchor_time)
        # Anchor the upstream filter's internal time origin at artifact.anchor_time
        # so survival is evaluated at dt = abs_time - anchor_time consistently for
        # both observation updates and query lookups.
        state = mix_filter.init(log_s, params, pi, initialization_time=anchor_time)

        # Drive observations in chronological order.
        for obs in sorted(observations or [], key=lambda o: o.timestamp):
            state = mix_filter.update(
                state,
                bool(obs.is_positive),
                float(obs.timestamp),
                self.p_m,
                self.p_f,
            )

        abs_query_times = anchor_time + query_times.to(torch.float64).cpu().numpy()
        q_jnp = jnp.asarray(abs_query_times, dtype=jnp.float32)

        # Upstream returns (best_component_posterior, all_component_posteriors).
        #   best:        shape (1, Q)
        #   all (per_k): shape (K, Q)
        # Marginalise over components using the posterior τ_k(N).
        _best, per_k = mix_filter.predict(state, q_jnp)
        log_tau = state.log_tau  # (K,)
        tau = jnp.exp(log_tau)
        tau = tau / jnp.sum(tau)
        marginal = jnp.sum(per_k * tau[:, None], axis=0)
        return jnp_to_torch(marginal, dtype=torch.float64)
