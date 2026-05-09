"""Pure-torch tests for MixtureSurvivalFunction (no JAX needed).

Validates:
  - K=1 mixture matches a single-component baseline.
  - K>=2 mixture is monotone non-increasing in t and bounded in [0, 1].
  - Generators reject malformed params via IncompatibleEstimateError.
"""

from __future__ import annotations

import math

import pytest
import torch

from persistence_eval.core.domain import PersistenceParameterEstimate
from persistence_eval.core.exceptions import IncompatibleEstimateError
from persistence_eval.methods.survival.exponential import (
    ExponentialSurvivalFunction,
)
from persistence_eval.methods.survival.log_normal import LogNormalSurvivalFunction
from persistence_eval.methods.survival.mixture import (
    MixtureExponentialGenerator,
    MixtureLogNormalGenerator,
    MixtureSurvivalFunction,
)


def test_mixture_exponential_k1_matches_single():
    rate = 0.02   # mean 50s
    times = torch.tensor([1.0, 10.0, 50.0, 200.0], dtype=torch.float64)

    mix = MixtureExponentialGenerator().build(
        PersistenceParameterEstimate(params={"weights": [1.0], "lambdas": [rate]})
    )
    single = ExponentialSurvivalFunction(lam=torch.tensor(1.0 / rate, dtype=torch.float64))

    assert torch.allclose(mix(times), single(times), atol=1e-9)


def test_mixture_lognormal_k1_matches_single():
    mu, sigma = 4.0, 0.5
    times = torch.tensor([1.0, 10.0, 100.0, 1000.0], dtype=torch.float64)

    mix = MixtureLogNormalGenerator().build(
        PersistenceParameterEstimate(
            params={"weights": [1.0], "logmus": [mu], "stds": [sigma]}
        )
    )
    single = LogNormalSurvivalFunction(
        mu=torch.tensor(mu, dtype=torch.float64),
        sigma=torch.tensor(sigma, dtype=torch.float64),
    )

    assert torch.allclose(mix(times), single(times), atol=1e-9)


def test_mixture_exp_two_component_monotone_and_bounded():
    mix = MixtureExponentialGenerator().build(
        PersistenceParameterEstimate(
            params={"weights": [0.3, 0.7], "lambdas": [0.005, 0.05]}
        )
    )
    times = torch.linspace(0.1, 1000.0, 200, dtype=torch.float64)
    s = mix(times)
    assert torch.all(s >= 0.0) and torch.all(s <= 1.0)
    diffs = s[1:] - s[:-1]
    assert torch.all(diffs <= 1e-12), "mixture survival must be non-increasing"


def test_mixture_normalises_unnormalised_weights():
    """Generator should normalise weights that don't sum to 1."""
    g = MixtureExponentialGenerator()
    e1 = PersistenceParameterEstimate(params={"weights": [1.0, 1.0], "lambdas": [0.01, 0.1]})
    e2 = PersistenceParameterEstimate(params={"weights": [0.5, 0.5], "lambdas": [0.01, 0.1]})
    s1 = g.build(e1)(torch.tensor([10.0], dtype=torch.float64))
    s2 = g.build(e2)(torch.tensor([10.0], dtype=torch.float64))
    assert torch.allclose(s1, s2, atol=1e-12)


def test_mismatched_lengths_rejected():
    with pytest.raises(IncompatibleEstimateError):
        MixtureExponentialGenerator().build(
            PersistenceParameterEstimate(params={"weights": [0.5, 0.5], "lambdas": [0.01]})
        )
    with pytest.raises(IncompatibleEstimateError):
        MixtureLogNormalGenerator().build(
            PersistenceParameterEstimate(
                params={"weights": [1.0], "logmus": [4.0], "stds": [0.5, 0.5]}
            )
        )


def test_missing_keys_rejected():
    with pytest.raises(IncompatibleEstimateError):
        MixtureExponentialGenerator().build(
            PersistenceParameterEstimate(params={"weights": [1.0]})  # missing lambdas
        )
