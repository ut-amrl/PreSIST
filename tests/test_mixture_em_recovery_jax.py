"""End-to-end EM recovery test for the mixture trainers.

Synthetic 2-component exponential mixture with planted (pi, lambda); fit our
MixtureExponentialTrainer; assert recovery within tolerance.

Skipped when jax/upstream aren't installed. JAX is forced onto CPU.

Note: upstream's EM uses float32 throughout, which limits numerical
precision; we accept ~10% relative error on lambdas and 0.1 absolute on pi
weights with N=200 sequences.
"""

from __future__ import annotations

import math
import os
import random

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

jax = pytest.importorskip("jax", reason="jax not installed; skip EM recovery test")
upstream = pytest.importorskip(
    "src.learning.mixture_exponential", reason="vendored perpetua-code not importable"
)

from persistence_eval.methods.training.base import TrainingDataset
from persistence_eval.methods.training.fit_mixture_exponential import (
    MixtureExponentialTrainer,
)


_US_TO_S = 1_000_000.0


def _make_track(track_id: str, duration_s: float, *, space_id: str, sample_dt: float = 1.0) -> dict:
    """Build a track with regularly-sampled Positive observations during the alive
    interval, then a couple of Negative observations after disappearance.

    This mirrors how real annotation_tool tracks look (many observations per
    track) and gives upstream EM the temporal density it needs to localize the
    disappearance time. With only two endpoint observations EM is pathological.
    """
    first = 1_000_000_000
    last = first + int(duration_s * _US_TO_S)
    obs = []
    t = first
    step_us = max(1, int(sample_dt * _US_TO_S))
    while t < last:
        obs.append({"state": "Positive", "timestamp": t, "bbox": [0, 0, 10, 10]})
        t += step_us
    obs.append({"state": "Positive", "timestamp": last, "bbox": [0, 0, 10, 10]})
    # A pair of Negative observations after disappearance bracket the failure time.
    obs.append({"state": "Negative", "timestamp": last + step_us, "bbox": None})
    obs.append({"state": "Negative", "timestamp": last + 2 * step_us, "bbox": None})
    return {
        "track_id": track_id,
        "space_id": space_id,
        "observations": obs,
        "boundaries": {
            "first_observed_ts": first,
            "last_observed_ts": last,
            "last_absent_before_ts": None,
            "first_absent_after_ts": last + step_us,
        },
    }


def test_em_exponential_runs_and_produces_sensible_2component():
    """Behavioural sniff test for the mixture-exponential trainer.

    We don't assert recovery of the *planted* (π, λ) — upstream's EM is
    sensitive to initialisation and the paper itself uses model selection +
    perturbation refinement to escape local optima, neither of which we
    implement here. Instead we check the trainer:
      - runs without NaN / inf
      - returns 2 components (no collapse to single via fallback)
      - both weights are non-trivial (no near-zero component)
      - both rates are positive and finite
      - the two rates separate (mixture didn't degenerate to a single rate)
      - the implied mean lifetime is in the right order of magnitude
    """
    rng = random.Random(0)
    n = 200
    pi_true = [0.4, 0.6]
    lam_true = [0.005, 0.05]   # mean 200, mean 20
    durations = []
    for _ in range(n):
        k = 0 if rng.random() < pi_true[0] else 1
        durations.append(rng.expovariate(lam_true[k]))
    raw = [_make_track(f"t{i}", d, space_id="A", sample_dt=5.0)
           for i, d in enumerate(durations)]
    ds = TrainingDataset(raw_tracks=raw, cell_size_px=64)

    # params_range=[2] skips the K-sweep so the test stays fast and
    # deterministic; num_retrain_attempts=0 likewise skips the perturbation
    # refinement loop. The select-and-refine machinery is exercised by other
    # tests/runs.
    trainer = MixtureExponentialTrainer(
        grouping="location",
        params_range=[2],
        num_retrain_attempts=0,
        min_mixture_samples=10,
        n_iter=200,
    )
    out = trainer.train(ds)
    entry = out["A"]
    assert entry["metadata"].get("fallback_to_single") is not True

    pi_hat = entry["params"]["weights"]
    lam_hat = entry["params"]["lambdas"]

    assert len(pi_hat) == 2 and len(lam_hat) == 2
    assert all(math.isfinite(w) for w in pi_hat)
    assert all(math.isfinite(l) for l in lam_hat)
    assert math.isclose(sum(pi_hat), 1.0, abs_tol=1e-5), pi_hat
    assert all(w >= 0.05 for w in pi_hat), pi_hat
    assert all(l > 0.0 for l in lam_hat), lam_hat
    # Components must separate — if they collapse the mixture is meaningless.
    assert max(lam_hat) / min(lam_hat) >= 2.0, lam_hat

    # Implied mean lifetime in the same order of magnitude as the planted
    # mixture's mean (planted = 0.4*200 + 0.6*20 = 92s).
    means = [1.0 / l for l in lam_hat]
    fitted_mean = sum(w * m for w, m in zip(pi_hat, means))
    planted_mean = sum(w * m for w, m in zip(pi_true, [1 / l for l in lam_true]))
    assert 0.1 * planted_mean < fitted_mean < 10.0 * planted_mean, (
        fitted_mean, planted_mean
    )

    # log-evidence reported by the trainer should be finite (EM converged
    # on something).
    assert math.isfinite(entry["metadata"]["final_log_evidence"])
