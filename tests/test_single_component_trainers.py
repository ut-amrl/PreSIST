"""Tests for SingleExponentialTrainer and SingleLogNormalTrainer.

Builds an in-memory TrainingDataset from synthetic raw_tracks and verifies:
  - closed-form fits recover planted parameters within tolerance
  - per-class and per-location grouping produce different bucket sets
  - empty/degenerate buckets fall back to the global pooled fit
  - subsample_fraction reduces the number of tracks used
"""

from __future__ import annotations

import math
import random

import pytest

from persistence_eval.methods.training.base import TrainingDataset
from persistence_eval.methods.training.fit_single_exponential import (
    SingleExponentialTrainer,
)
from persistence_eval.methods.training.fit_single_lognormal import (
    SingleLogNormalTrainer,
)


_US_TO_S = 1_000_000.0


def _make_track(track_id: str, duration_s: float, *, space_id: str | None) -> dict:
    """Build a minimal raw track dict with first/last_observed_ts span = duration_s."""
    first = 1_000_000_000  # seconds since epoch is irrelevant; use any base
    last = first + int(duration_s * _US_TO_S)
    track: dict = {
        "track_id": track_id,
        "observations": [
            {"state": "Positive", "timestamp": first, "bbox": [0, 0, 10, 10]},
            {"state": "Positive", "timestamp": last, "bbox": [0, 0, 10, 10]},
        ],
        "boundaries": {
            "first_observed_ts": first,
            "last_observed_ts": last,
            "last_absent_before_ts": None,
            "first_absent_after_ts": None,
        },
    }
    if space_id is not None:
        track["space_id"] = space_id
    return track


def test_single_exponential_recovers_rate():
    rng = random.Random(0)
    rate = 0.01  # mean = 100s
    durations = [rng.expovariate(rate) for _ in range(2000)]
    raw = [
        _make_track(f"t{i}", d, space_id="A") for i, d in enumerate(durations)
    ]
    ds = TrainingDataset(raw_tracks=raw, cell_size_px=64)
    out = SingleExponentialTrainer(grouping="location").train(ds)

    fitted_rate = out["A"]["params"]["lambda_"]
    fitted_mean = out["A"]["params"]["mean"]
    assert math.isclose(fitted_mean, 1 / rate, rel_tol=0.10), fitted_mean
    assert math.isclose(fitted_rate, rate, rel_tol=0.10), fitted_rate


def test_single_lognormal_recovers_mu_sigma():
    rng = random.Random(0)
    mu, sigma = 4.0, 0.5  # logmu=4, std=0.5
    durations = [math.exp(rng.gauss(mu, sigma)) for _ in range(2000)]
    raw = [_make_track(f"t{i}", d, space_id="A") for i, d in enumerate(durations)]
    ds = TrainingDataset(raw_tracks=raw, cell_size_px=64)
    out = SingleLogNormalTrainer(grouping="location").train(ds)

    p = out["A"]["params"]
    assert math.isclose(p["logmu"], mu, abs_tol=0.05), p
    assert math.isclose(p["std"], sigma, abs_tol=0.05), p


def test_per_class_vs_per_location_buckets_differ():
    raw = [
        _make_track("t1", 100.0, space_id="A"),
        _make_track("t2", 200.0, space_id="B"),
        _make_track("t3", 300.0, space_id="A"),
    ]
    ds = TrainingDataset(raw_tracks=raw, cell_size_px=64)

    by_class = SingleExponentialTrainer(grouping="class").train(ds)
    by_loc = SingleExponentialTrainer(grouping="location").train(ds)

    assert set(by_class) == {"__global__", "0"}
    assert set(by_loc) == {"__global__", "A", "B"}


def test_lognormal_fallback_for_degenerate_bucket():
    raw = [
        _make_track("t1", 50.0, space_id="A"),
        _make_track("t2", 100.0, space_id="A"),
        # B has only one sample → degenerate variance estimator → fallback.
        _make_track("t3", 200.0, space_id="B"),
    ]
    ds = TrainingDataset(raw_tracks=raw, cell_size_px=64)
    out = SingleLogNormalTrainer(grouping="location").train(ds)
    assert out["B"]["metadata"].get("fallback_to_global") is True
    assert out["A"]["metadata"].get("fallback_to_global") is None or False


def test_subsample_fraction_limits_tracks():
    raw = [_make_track(f"t{i}", 10.0 + i, space_id="A") for i in range(100)]
    ds = TrainingDataset(
        raw_tracks=raw, cell_size_px=64, subsample_fraction=0.1, subsample_seed=0
    )
    assert ds.n_tracks() == 10
    out = SingleExponentialTrainer(grouping="location").train(ds)
    assert out["A"]["metadata"]["n_samples"] == 10


def test_subsample_seed_determinism():
    raw = [_make_track(f"t{i}", 10.0 + i, space_id="A") for i in range(100)]
    a = TrainingDataset(raw_tracks=raw, cell_size_px=64, subsample_fraction=0.2, subsample_seed=7)
    b = TrainingDataset(raw_tracks=raw, cell_size_px=64, subsample_fraction=0.2, subsample_seed=7)
    c = TrainingDataset(raw_tracks=raw, cell_size_px=64, subsample_fraction=0.2, subsample_seed=8)
    a_ids = [t.track_id for t in a.iter_tracks()]
    b_ids = [t.track_id for t in b.iter_tracks()]
    c_ids = [t.track_id for t in c.iter_tracks()]
    assert a_ids == b_ids
    assert a_ids != c_ids
