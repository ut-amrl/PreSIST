"""Tests for RosenPersistenceFilter under the Phase-2/3 anchor + obs API."""

from __future__ import annotations

import math

import torch

from persistence_eval.core.domain import BinaryObservationEvent
from persistence_eval.methods.filters.rosen_filter import RosenPersistenceFilter
from persistence_eval.methods.survival.base import AbsSurvivalFunction


class _ExpSurvival(AbsSurvivalFunction):
    """S(dt) = exp(-rate * dt) for dt > 0; 1.0 for dt ≤ 0."""

    def __init__(self, rate: float) -> None:
        super().__init__()
        self._rate = float(rate)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.where(dt <= 0, torch.ones_like(dt), torch.exp(-self._rate * dt))

    def log_survival(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.where(
            dt <= 0,
            torch.zeros_like(dt),
            -self._rate * dt,
        )


def _make_obs(times_pos):
    return [BinaryObservationEvent(timestamp=t, is_positive=p) for t, p in times_pos]


def test_anchor_only_matches_S():
    """Empty obs list ⇒ posterior == survival_fn(query_times)."""
    f = RosenPersistenceFilter(false_detection_prob=0.05, missed_detection_prob=0.1)
    surv = _ExpSurvival(rate=0.02)
    query = torch.tensor([5.0, 30.0, 60.0], dtype=torch.float64)

    out = f.predict(surv, anchor_time=0.0, query_times=query)
    expected = surv(query)
    assert torch.allclose(out, expected)


def test_pre_anchor_obs_are_information_free():
    """Pre-anchor positive observations under monotonic survival add no
    information to the posterior — output equals the prior survival."""
    f = RosenPersistenceFilter(false_detection_prob=0.05, missed_detection_prob=0.1)
    surv = _ExpSurvival(rate=0.02)
    query = torch.tensor([5.0, 30.0, 60.0], dtype=torch.float64)

    pre_anchor_obs = _make_obs([(-100.0, True), (-50.0, True), (-10.0, True)])
    out = f.predict(surv, anchor_time=0.0, query_times=query, observations=pre_anchor_obs)
    expected = surv(query)
    assert torch.allclose(out, expected, atol=1e-12)


def test_post_anchor_negative_obs_pulls_posterior_below_prior():
    f = RosenPersistenceFilter(false_detection_prob=0.05, missed_detection_prob=0.1)
    surv = _ExpSurvival(rate=0.02)
    anchor = 100.0
    obs = _make_obs([(anchor + 30.0, False)])
    query = torch.tensor([35.0, 60.0, 90.0], dtype=torch.float64)

    out = f.predict(surv, anchor_time=anchor, query_times=query, observations=obs).numpy()
    prior = surv(query).numpy()
    assert ((out >= 0.0) & (out <= 1.0)).all()
    assert (out < prior).all(), (out, prior)


def test_invariant_to_absolute_anchor_time():
    """Posterior depends only on dt = obs/query - anchor, not on absolute times."""
    f = RosenPersistenceFilter(false_detection_prob=0.05, missed_detection_prob=0.1)
    surv = _ExpSurvival(rate=0.02)
    query = torch.tensor([5.0, 35.0, 60.0], dtype=torch.float64)

    obs0 = _make_obs([(20.0, True), (40.0, False)])
    obs1 = _make_obs([(1_000_020.0, True), (1_000_040.0, False)])

    p0 = f.predict(surv, anchor_time=0.0, query_times=query, observations=obs0)
    p1 = f.predict(surv, anchor_time=1_000_000.0, query_times=query, observations=obs1)
    assert torch.allclose(p0, p1, atol=1e-12)


def test_chronological_ordering_independent_of_input_order():
    f = RosenPersistenceFilter(false_detection_prob=0.05, missed_detection_prob=0.1)
    surv = _ExpSurvival(rate=0.02)
    query = torch.tensor([60.0], dtype=torch.float64)

    obs_in_order = _make_obs([(10.0, True), (20.0, True), (30.0, False)])
    obs_shuffled = _make_obs([(30.0, False), (10.0, True), (20.0, True)])

    p_in = f.predict(surv, anchor_time=0.0, query_times=query, observations=obs_in_order)
    p_sh = f.predict(surv, anchor_time=0.0, query_times=query, observations=obs_shuffled)
    assert torch.allclose(p_in, p_sh)


def test_single_post_anchor_positive_obs_zero_noise_returns_prior_extrapolated_from_obs():
    """With p_miss=p_false=0 and a single post-anchor positive obs at t_obs,
    the posterior at any query t > t_obs is S(t-anchor) / S(t_obs-anchor).
    This is a sanity check that the math anchors the survival origin
    correctly through the observation update."""
    f = RosenPersistenceFilter(false_detection_prob=1e-12, missed_detection_prob=1e-12)
    surv = _ExpSurvival(rate=0.05)
    anchor = 0.0
    t_obs = 10.0
    obs = _make_obs([(t_obs, True)])
    query = torch.tensor([20.0, 40.0], dtype=torch.float64)

    out = f.predict(surv, anchor_time=anchor, query_times=query, observations=obs).numpy()
    expected = (
        surv(query).numpy()
        / surv(torch.tensor([t_obs - anchor], dtype=torch.float64)).item()
    )
    assert all(math.isclose(o, e, rel_tol=1e-6) for o, e in zip(out, expected))
