"""Mixture persistence filter — thin wrapper around upstream Perpetua.

Wraps src.filters.MixturePersistenceFilters.{init,update,predict} from the
vendored upstream package. Translates our (MixtureSurvivalFunction, anchor_time,
query times, observations) into upstream's (FilterState, observation sequence,
query array), runs the filter, and returns a torch tensor.

Returns the **argmax-component** posterior per Perpetua eq. (14):

    k* = argmax_k p(C_k | Y_{1:N})
    return p(X_t = 1 | C_{k*}, Y_{1:N})

The paper explicitly rejects posterior-weighted marginalisation here — "due to
potential destructive interference between mixture components, we instead use
the component with the largest posterior weight."

Times handed to upstream are re-anchored to `t = 0` at `anchor_time` to keep
JAX float32 arithmetic in a regime that resolves sub-second dt; passing raw
Unix timestamps (~1.35e9) would collapse sub-128 s horizons to identical
float32 values.

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
        # Re-anchor everything to t=0 before crossing into JAX. Upstream is
        # float32 by default; raw Unix timestamps (~1.35e9) snap to ~128 s ulps
        # and observation/query dt's quantise to that grid. With init_time=0
        # and rel_t = float(obs.timestamp) - anchor_time computed in float64
        # before the cast, max realistic dt is ~3e6 s where float32 ulp ≈ 0.25 s.
        state = mix_filter.init(log_s, params, pi, initialization_time=0.0)

        # Drive observations in chronological order, in track-relative seconds.
        for obs in sorted(observations or [], key=lambda o: o.timestamp):
            rel_t = float(obs.timestamp) - anchor_time
            state = mix_filter.update(
                state,
                bool(obs.is_positive),
                rel_t,
                self.p_m,
                self.p_f,
            )

        # query_times are already dt from anchor (AbsPersistenceFilter contract).
        q_jnp = jnp.asarray(query_times.to(torch.float64).cpu().numpy(), dtype=jnp.float32)

        # Upstream returns (best_component_posterior, all_component_posteriors).
        #   best:        shape (1, Q)  — argmax_k p(C_k | Y_{1:N}), per eq. (14)
        #   per_k:       shape (K, Q)
        best, _per_k = mix_filter.predict(state, q_jnp)
        return jnp_to_torch(best[0], dtype=torch.float64)
