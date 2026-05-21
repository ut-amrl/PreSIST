"""JAX-dependent tests for MixturePersistenceFilter.

Skipped when jax/upstream perpetua-code aren't importable. Validates the
anchor-only form (Σ_k π_k · S_k(query_dt) relative to artifact.anchor_time)
and the observation-bearing form used by the scaled-horizon alternate eval
(post-anchor obs are folded in chronologically; survival origin stays at
anchor_time).
"""

from __future__ import annotations

import os

import pytest
import torch

# Force JAX onto CPU regardless of system default (matches the production wiring).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

jax = pytest.importorskip("jax", reason="jax not installed; skip mixture-filter tests")
upstream = pytest.importorskip(
    "src.filters.MixturePersistenceFilters",
    reason="vendored perpetua-code not importable",
)

from persistence_eval.core.domain import BinaryObservationEvent
from persistence_eval.methods.filters.mixture_persistence_filter import (
    MixturePersistenceFilter,
)
from persistence_eval.methods.survival.mixture import (
    MixtureExponentialGenerator,
)
from persistence_eval.core.domain import PersistenceParameterEstimate


def _make_obs(times_pos_pairs):
    return [BinaryObservationEvent(timestamp=t, is_positive=p) for t, p in times_pos_pairs]


def test_k1_anchor_only_reduces_to_S():
    """No observations ⇒ posterior equals prior survival S(query_dt)."""
    rate = 0.02   # mean 50s
    query = torch.tensor([5.0, 30.0, 60.0], dtype=torch.float64)
    expected_S = torch.exp(-rate * query).numpy()

    mix_filter = MixturePersistenceFilter()
    mix_surv = MixtureExponentialGenerator().build(
        PersistenceParameterEstimate(params={"weights": [1.0], "lambdas": [rate]})
    )
    mix_pred = mix_filter.predict(mix_surv, anchor_time=0.0, query_times=query).numpy()

    # Upstream uses float32, so allow ~1e-4 slack.
    assert mix_pred.shape == expected_S.shape
    assert ((mix_pred >= -1e-6) & (mix_pred <= 1.0 + 1e-6)).all()
    assert (abs(mix_pred - expected_S) < 1e-4).all(), (mix_pred, expected_S)


def test_k1_anchor_only_invariant_to_anchor_time_value():
    """Posterior depends only on dt, not on absolute anchor_time."""
    rate = 0.02
    query = torch.tensor([5.0, 30.0, 60.0], dtype=torch.float64)

    mix_filter = MixturePersistenceFilter()
    mix_surv = MixtureExponentialGenerator().build(
        PersistenceParameterEstimate(params={"weights": [1.0], "lambdas": [rate]})
    )
    pred_at_zero = mix_filter.predict(mix_surv, anchor_time=0.0, query_times=query).numpy()
    pred_at_million = mix_filter.predict(
        mix_surv, anchor_time=1_000_000.0, query_times=query
    ).numpy()

    assert (abs(pred_at_zero - pred_at_million) < 1e-4).all()


def test_two_component_anchor_only_bounded_and_non_increasing():
    query = torch.tensor([60.0, 100.0, 200.0, 500.0], dtype=torch.float64)

    mix_filter = MixturePersistenceFilter()
    mix_surv = MixtureExponentialGenerator().build(
        PersistenceParameterEstimate(
            params={"weights": [0.5, 0.5], "lambdas": [0.005, 0.05]}
        )
    )
    pred = mix_filter.predict(mix_surv, anchor_time=0.0, query_times=query).numpy()
    assert ((pred >= -1e-6) & (pred <= 1.0 + 1e-6)).all()
    diffs = pred[1:] - pred[:-1]
    assert (diffs <= 1e-3).all(), pred


def test_post_anchor_negative_obs_pulls_posterior_below_prior():
    """A post-anchor negative obs at t = anchor + 30 should drive the posterior
    at later queries strictly below the prior survival."""
    rate = 0.02
    anchor = 100.0  # nonzero anchor to confirm anchor_time plumbing
    obs = _make_obs([(anchor + 30.0, False)])
    query = torch.tensor([35.0, 60.0, 90.0], dtype=torch.float64)

    mix_filter = MixturePersistenceFilter()
    mix_surv = MixtureExponentialGenerator().build(
        PersistenceParameterEstimate(params={"weights": [1.0], "lambdas": [rate]})
    )
    posterior = mix_filter.predict(
        mix_surv, anchor_time=anchor, query_times=query, observations=obs
    ).numpy()

    assert ((posterior >= -1e-6) & (posterior <= 1.0 + 1e-6)).all()
    prior_S = torch.exp(-rate * query).numpy()
    assert (posterior < prior_S).all(), (posterior, prior_S)


def test_observation_bearing_invariant_to_absolute_anchor_time():
    """Shifting anchor_time and obs.timestamp together by the same offset must
    leave the posterior unchanged — the filter operates on dt = abs - anchor."""
    rate = 0.02
    query = torch.tensor([5.0, 35.0, 60.0], dtype=torch.float64)
    mix_filter = MixturePersistenceFilter()
    mix_surv = MixtureExponentialGenerator().build(
        PersistenceParameterEstimate(params={"weights": [1.0], "lambdas": [rate]})
    )

    obs_at_zero = _make_obs([(20.0, True), (40.0, False)])
    obs_at_million = _make_obs([(1_000_020.0, True), (1_000_040.0, False)])

    p0 = mix_filter.predict(
        mix_surv, anchor_time=0.0, query_times=query, observations=obs_at_zero
    ).numpy()
    p1 = mix_filter.predict(
        mix_surv, anchor_time=1_000_000.0, query_times=query, observations=obs_at_million
    ).numpy()

    assert (abs(p0 - p1) < 1e-4).all(), (p0, p1)


def test_unix_epoch_anchor_sub_128s_horizons_are_resolved():
    """Regression test for the JAX float32 anchor bug.

    With anchor_time ≈ 1.35e9 (Unix epoch in 2012) and the old code path,
    `q_jnp = jnp.asarray(anchor_time + dt, dtype=jnp.float32)` snapped each
    query to the nearest ~128 s ulp and then subtracted a same-grid anchor —
    sub-128 s horizons collapsed to dt = 0 and the posterior returned S(0) = 1
    for all of them. After re-anchoring to t=0 before crossing into JAX,
    sub-second horizons resolve and the posterior is strictly decreasing.
    """
    rate = 0.02
    anchor = 1_350_000_000.0  # Unix-epoch-scale, where float32 ulp ≈ 128 s
    obs = _make_obs([(anchor + 5.0, True), (anchor + 50.0, False)])
    # Sub-128 s horizons that would have collapsed pre-fix.
    query = torch.tensor([0.0, 1.0, 10.0, 60.0], dtype=torch.float64)

    mix_filter = MixturePersistenceFilter()
    mix_surv = MixtureExponentialGenerator().build(
        PersistenceParameterEstimate(params={"weights": [1.0], "lambdas": [rate]})
    )
    pred = mix_filter.predict(
        mix_surv, anchor_time=anchor, query_times=query, observations=obs
    ).numpy()

    # All four predictions must be distinct (pre-fix they were all equal to S(0)=1).
    uniq = set(round(float(p), 6) for p in pred)
    assert len(uniq) == query.numel(), (pred, uniq)
    # Monotone non-increasing in the query horizon.
    diffs = pred[1:] - pred[:-1]
    assert (diffs <= 1e-6).all(), pred
