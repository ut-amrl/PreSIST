"""Mixture filter that uses observations only to pick the argmax component.

Same wiring as `MixturePersistenceFilter` (upstream JAX mixture state, re-anchored
to t=0, observations folded in to evolve `log_tau`), but the FINAL PREDICTION
drops the observation-conditioned likelihood correction. The returned value is
just `S_{k*}(query_dt)` — the prior survival under the chosen component — with
`k* = argmax_k log_tau_k` after the observation stream has been processed.

Compared to `MixturePersistenceFilter`:
  - Both: use observations to evolve component posterior weights `log_tau_k`.
  - Both: select `k* = argmax_k log_tau_k` at predict time (paper eq. 14).
  - Standard filter additionally folds the observation-conditioned likelihood
    factor (Rosen's per-component update) into the per-query prediction:
        p(X_t = 1 | C_k*, Y_{1:N})
            = exp(log_likelihood + log_S_{k*}(t) - log_conditional_evidence_{k*})
  - This filter drops the likelihood/evidence terms and returns:
        p(X_t = 1 | C_k*)  =  S_{k*}(query_dt)
    i.e., the prior survival from the picked component, no observation-driven
    update of the per-query probability.

Use case: ablation. Isolates "observations help me pick the right mixture
component" from "observations sharpen the probability given that component."
Especially useful for diagnosing the lognormal MAE regression where the
argmax-k commit + Rosen update combine to produce confident-but-wrong
predictions: this filter keeps the component-selection benefit but does not
let observations amplify confidence in (or against) the picked component.

The filter accepts only `MixtureSurvivalFunction` — same as
`MixturePersistenceFilter`.
"""

from __future__ import annotations

import math

import torch

from persistence_eval.core.domain import BinaryObservationEvent
from persistence_eval.core.exceptions import IncompatibleEstimateError
from persistence_eval.methods.filters.base import AbsPersistenceFilter
from persistence_eval.methods.filters.perpetua_filter_adapter import _jax_modules
from persistence_eval.methods.survival.base import AbsSurvivalFunction
from persistence_eval.methods.survival.mixture import MixtureSurvivalFunction


class MixturePriorPickFilter(AbsPersistenceFilter):
    """Mixture filter — observations select k*, but prediction is the prior S_{k*}(dt)."""

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
                "MixturePriorPickFilter requires a MixtureSurvivalFunction; "
                f"got {type(survival_fn).__name__}"
            )

        mods = _jax_modules()
        jnp = mods["jnp"]
        mix_filter = mods["mixture_filter"]

        log_s, params, pi = survival_fn.build_jax_state()
        anchor_time = float(anchor_time)

        # Re-anchor to t=0 (same float32-safe pattern as MixturePersistenceFilter).
        state = mix_filter.init(log_s, params, pi, initialization_time=0.0)
        for obs in sorted(observations or [], key=lambda o: o.timestamp):
            rel_t = float(obs.timestamp) - anchor_time
            state = mix_filter.update(
                state, bool(obs.is_positive), rel_t, self.p_m, self.p_f
            )

        # Post-observation argmax-k from log_tau (which has absorbed the obs likelihood).
        k_star = int(jnp.argmax(state.log_tau))

        # Compute S_{k*}(query_dt) DIRECTLY from the picked component's params —
        # NO observation-conditioned likelihood factor. query_times are dt from
        # anchor per the AbsPersistenceFilter contract.
        dt = query_times.to(torch.float64)
        if survival_fn.component_kind == "exponential":
            lam = survival_fn.component_param("lambda_")[k_star].to(torch.float64)
            log_S_k = -dt.clamp(min=0.0) * lam
        else:  # lognormal
            logmu = survival_fn.component_param("logmu")[k_star].to(torch.float64)
            std = survival_fn.component_param("std")[k_star].to(torch.float64)
            t_safe = dt.clamp(min=1e-38)
            z = (torch.log(t_safe) - logmu) / std
            log_S_k = torch.special.log_ndtr(-z)
        log_S_k = torch.where(dt <= 0, torch.zeros_like(dt), log_S_k)
        return torch.exp(log_S_k).clamp(0.0, 1.0)
