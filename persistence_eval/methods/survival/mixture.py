"""Mixture survival functions.

Holds K-component (pi, params) bundles for the two distribution families used
in the Perpetua wrappers. Implements `AbsSurvivalFunction` (so any filter that
only consumes log_survival(t) keeps working), and additionally exposes a
`build_jax_state()` method that hands the filter wrapper everything
`FilterState.create` needs:

    log_s: jax.tree_util.Partial    — upstream priors.log_survival_*
    params: dict[str, jnp.ndarray]  — broadcast-compatible (K,) arrays
    pi: jnp.ndarray                 — (K,)

There is no new mixture math here. log_s is upstream's per-component log
survival; pi is the mixing-weight vector that upstream FilterState carries.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from persistence_eval.core.domain import PersistenceParameterEstimate
from persistence_eval.core.exceptions import IncompatibleEstimateError
from persistence_eval.methods.survival.base import (
    AbsSurvivalFunction,
    AbsSurvivalFunctionGenerator,
)


class MixtureSurvivalFunction(AbsSurvivalFunction):
    """K-component mixture survival.

    component_kind: "exponential" or "lognormal".
    component_params: dict of (K,) tensors keyed by upstream parameter names.
        - exponential: {"lambda_": (K,)}
        - lognormal:   {"logmu": (K,), "std": (K,)}
    pi: (K,) tensor, sums to 1.

    Storing parameters in their natural domain makes torch evaluation simple;
    the jax bundle for the filter wrapper is materialised lazily.
    """

    def __init__(
        self,
        *,
        component_kind: str,
        component_params: dict[str, torch.Tensor],
        pi: torch.Tensor,
    ) -> None:
        super().__init__()
        if component_kind not in {"exponential", "lognormal"}:
            raise ValueError(component_kind)
        self.component_kind = component_kind
        # nn.Buffer-style storage so the tensors travel with the module.
        for name, tensor in component_params.items():
            self.register_buffer(f"_p_{name}", tensor.to(torch.float64))
        self.register_buffer("_pi", pi.to(torch.float64))
        self._param_names = tuple(component_params.keys())

    @property
    def n_components(self) -> int:
        return int(self._pi.shape[0])

    @property
    def pi(self) -> torch.Tensor:
        return self._pi

    def component_param(self, name: str) -> torch.Tensor:
        return getattr(self, f"_p_{name}")

    def log_survival(self, dt: torch.Tensor) -> torch.Tensor:
        """log S(t) = logsumexp_k(log pi_k + log S_k(t))."""
        dt = dt.to(torch.float64)
        # Compute per-component log-survival; broadcast (..., K).
        if self.component_kind == "exponential":
            lam = self.component_param("lambda_")  # (K,)
            # S_k(t) = exp(-lam_k * t), valid for t > 0; S(0)=1, S(t<0)=1 by convention.
            log_S_k = -dt.unsqueeze(-1) * lam
        else:  # lognormal
            logmu = self.component_param("logmu")  # (K,)
            std = self.component_param("std")      # (K,)
            # log S(t) = log Phi(-(log t - mu)/sigma); use log_ndtr for stability.
            t_safe = dt.clamp(min=1e-38)
            z = (torch.log(t_safe).unsqueeze(-1) - logmu) / std
            log_S_k = torch.special.log_ndtr(-z)
        log_pi = torch.log(self._pi)
        log_S = torch.logsumexp(log_pi + log_S_k, dim=-1)
        return torch.where(dt <= 0, torch.zeros_like(dt), log_S)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.where(dt <= 0, torch.ones_like(dt), torch.exp(self.log_survival(dt)))

    # -- JAX hand-off for the upstream filter wrapper -----------------------

    def build_jax_state(self):
        """Return (log_s_partial, params_dict, pi_arr) for FilterState.create.

        Imports JAX lazily.
        """
        from persistence_eval.methods.filters.perpetua_filter_adapter import (
            _jax_modules,
            torch_to_jnp,
        )

        mods = _jax_modules()
        Partial = mods["Partial"]
        priors = mods["priors"]

        if self.component_kind == "exponential":
            log_s = Partial(priors.log_survival_exponential)
            params = {"lambda_": torch_to_jnp(self.component_param("lambda_"))}
        else:
            log_s = Partial(priors.log_survival_lognormal)
            params = {
                "logmu": torch_to_jnp(self.component_param("logmu")),
                "std": torch_to_jnp(self.component_param("std")),
            }
        pi = torch_to_jnp(self._pi)
        return log_s, params, pi


def _require_keys(estimate: PersistenceParameterEstimate, keys: Sequence[str]) -> None:
    for k in keys:
        if k not in estimate.params:
            raise IncompatibleEstimateError(
                f"Mixture survival generator requires '{k}' in estimate.params; "
                f"got keys: {list(estimate.params)}"
            )


def _to_list(v) -> list[float]:
    if isinstance(v, list):
        return [float(x) for x in v]
    raise IncompatibleEstimateError(
        f"Expected list of floats; got {type(v).__name__}: {v!r}"
    )


class MixtureExponentialGenerator(AbsSurvivalFunctionGenerator):
    """Reads {"weights": [..K..], "lambdas": [..K..]} and builds a mixture.

    time_scale_factor: pre-multiplies each `lambda_` (rate units 1/time)
    inversely if the trainer fit on a different time unit. Default 1.0
    (the trainers in this repo emit seconds-scaled rates).
    """

    WEIGHTS_KEY = "weights"
    LAMBDAS_KEY = "lambdas"

    def __init__(self, time_scale_factor: float = 1.0, name: str | None = None) -> None:
        self._time_scale_factor = float(time_scale_factor)

    def build(self, estimate: PersistenceParameterEstimate) -> MixtureSurvivalFunction:
        _require_keys(estimate, [self.WEIGHTS_KEY, self.LAMBDAS_KEY])
        pi = _to_list(estimate.params[self.WEIGHTS_KEY])
        lambdas = _to_list(estimate.params[self.LAMBDAS_KEY])
        if len(pi) != len(lambdas):
            raise IncompatibleEstimateError(
                f"weights and lambdas must have equal length; got {len(pi)} vs {len(lambdas)}"
            )
        s = sum(pi)
        if s <= 0:
            raise IncompatibleEstimateError(f"weights must sum > 0; got {pi!r}")
        pi_norm = [w / s for w in pi]
        # Rate scales inversely with time: if trainer used hours and we use seconds,
        # divide rates by (seconds-per-hour). We expose a unit factor on time;
        # rates -> rates * (1/factor) keeps S(t) unchanged when t is rescaled by factor.
        scaled_lambdas = [l / self._time_scale_factor for l in lambdas]
        return MixtureSurvivalFunction(
            component_kind="exponential",
            component_params={"lambda_": torch.tensor(scaled_lambdas, dtype=torch.float64)},
            pi=torch.tensor(pi_norm, dtype=torch.float64),
        )


class MixtureLogNormalGenerator(AbsSurvivalFunctionGenerator):
    """Reads {"weights": [..K..], "logmus": [..K..], "stds": [..K..]} and builds a mixture."""

    WEIGHTS_KEY = "weights"
    LOGMUS_KEY = "logmus"
    STDS_KEY = "stds"

    def __init__(self, time_scale_factor: float = 1.0, name: str | None = None) -> None:
        self._time_scale_factor = float(time_scale_factor)

    def build(self, estimate: PersistenceParameterEstimate) -> MixtureSurvivalFunction:
        _require_keys(estimate, [self.WEIGHTS_KEY, self.LOGMUS_KEY, self.STDS_KEY])
        pi = _to_list(estimate.params[self.WEIGHTS_KEY])
        logmus = _to_list(estimate.params[self.LOGMUS_KEY])
        stds = _to_list(estimate.params[self.STDS_KEY])
        if not (len(pi) == len(logmus) == len(stds)):
            raise IncompatibleEstimateError(
                f"weights/logmus/stds must have equal length; got {len(pi)}/{len(logmus)}/{len(stds)}"
            )
        s = sum(pi)
        if s <= 0:
            raise IncompatibleEstimateError(f"weights must sum > 0; got {pi!r}")
        pi_norm = [w / s for w in pi]
        # Time rescale: if trainer fit on hours and framework uses seconds,
        # add log(factor) to logmu so S(t_seconds) matches S(t_hours/factor).
        if self._time_scale_factor != 1.0:
            shift = math.log(self._time_scale_factor)
            logmus = [m + shift for m in logmus]
        return MixtureSurvivalFunction(
            component_kind="lognormal",
            component_params={
                "logmu": torch.tensor(logmus, dtype=torch.float64),
                "std": torch.tensor(stds, dtype=torch.float64),
            },
            pi=torch.tensor(pi_norm, dtype=torch.float64),
        )
