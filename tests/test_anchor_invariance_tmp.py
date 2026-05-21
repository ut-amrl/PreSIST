"""Anchor-time invariance for both Rosen and Mixture filters under realistic obs streams."""
from __future__ import annotations
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import pytest
import torch

from persistence_eval.core.domain import BinaryObservationEvent, PersistenceParameterEstimate
from persistence_eval.methods.filters.rosen_filter import RosenPersistenceFilter
from persistence_eval.methods.filters.mixture_persistence_filter import MixturePersistenceFilter
from persistence_eval.methods.survival.mixture import MixtureExponentialGenerator
from persistence_eval.methods.survival.exponential import ExponentialSurvivalFunction


def _obs(times, alives):
    return [BinaryObservationEvent(timestamp=t, is_positive=a) for t, a in zip(times, alives)]


def _check(filter_inst, survival_fn, base_times, alives, offset, queries):
    p0 = filter_inst.predict(
        survival_fn, anchor_time=0.0, query_times=queries,
        observations=_obs(base_times, alives),
    ).detach().numpy()
    p1 = filter_inst.predict(
        survival_fn, anchor_time=offset, query_times=queries,
        observations=_obs([t + offset for t in base_times], alives),
    ).detach().numpy()
    return float(abs(p0 - p1).max())


@pytest.fixture
def queries():
    return torch.tensor([60.0, 600.0, 3600.0, 86400.0, 604800.0], dtype=torch.float64)


@pytest.fixture
def single_exp():
    # lam = mean lifetime = 50000 s ≈ 14 h (so rate ≈ 2e-5 /s).
    lam = torch.tensor(50000.0, dtype=torch.float64)
    return ExponentialSurvivalFunction(lam=lam)


@pytest.fixture
def mix_surv():
    return MixtureExponentialGenerator().build(
        PersistenceParameterEstimate(params={"weights": [0.4, 0.6], "lambdas": [2e-5, 5e-3]})
    )


OFFSET = 1.35e9  # PKLot-scale Unix epoch
SCENARIOS = [
    ("no obs",                      [], []),
    ("3 obs (2+ 1-)",               [300., 600., 1500.], [True, True, False]),
    ("5 positives",                 [60., 120., 180., 240., 300.], [True]*5),
    ("5-min cadence mid-window-",   [0., 300., 600., 900., 1200., 1500.], [True, True, True, True, False, False]),
]


@pytest.mark.parametrize("label,times,alives", SCENARIOS)
def test_rosen_invariant_single_exp(queries, single_exp, label, times, alives):
    rosen = RosenPersistenceFilter(false_detection_prob=0.05, missed_detection_prob=0.1)
    diff = _check(rosen, single_exp, times, alives, OFFSET, queries)
    assert diff < 1e-3, f"Rosen single-exp anchor invariance broke for {label}: max|Δ|={diff:.3e}"


@pytest.mark.parametrize("label,times,alives", SCENARIOS)
def test_rosen_invariant_mixture_exp(queries, mix_surv, label, times, alives):
    rosen = RosenPersistenceFilter(false_detection_prob=0.05, missed_detection_prob=0.1)
    diff = _check(rosen, mix_surv, times, alives, OFFSET, queries)
    assert diff < 1e-3, f"Rosen mixture-exp anchor invariance broke for {label}: max|Δ|={diff:.3e}"


@pytest.mark.parametrize("label,times,alives", SCENARIOS)
def test_mixture_filter_invariant(queries, mix_surv, label, times, alives):
    mix = MixturePersistenceFilter(false_detection_prob=0.05, missed_detection_prob=0.1)
    diff = _check(mix, mix_surv, times, alives, OFFSET, queries)
    assert diff < 1e-3, f"MixtureFilter anchor invariance broke for {label}: max|Δ|={diff:.3e}"
