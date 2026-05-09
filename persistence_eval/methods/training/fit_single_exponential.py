"""Single-exponential MLE trainer.

Fits one exponential rate parameter `lambda_` per bucket from per-track
durations:  λ̂ = N / Σ d_i  (no censoring; the plan assumes fully-observed
tracks). `lambda_` matches upstream Perpetua's parameter naming so the same
key feeds priors.log_survival_exponential without renaming.

Bucket-size guard: buckets with zero positive duration fall back to the
pooled "__global__" fit; surfaced via metadata["fallback_to_global"].
"""

from __future__ import annotations

import math

from persistence_eval.methods.training.base import (
    AbsTrainer,
    ParamEntry,
    TrainingDataset,
    TrainingTrack,
    bucket_tracks,
)


GLOBAL_FALLBACK_KEY = "__global__"


def _fit_lambda(durations: list[float]) -> float | None:
    """Return MLE rate λ̂ = N / Σ d_i, or None if no positive duration."""
    total = sum(d for d in durations if d > 0.0)
    n = sum(1 for d in durations if d > 0.0)
    if total <= 0.0 or n == 0:
        return None
    return n / total


def _entry(
    lambda_: float,
    n_samples: int,
    fallback: bool = False,
    extra_metadata: dict | None = None,
) -> ParamEntry:
    metadata = {
        "n_samples": n_samples,
        "model_type": "single_exponential",
    }
    if fallback:
        metadata["fallback_to_global"] = True
    if extra_metadata:
        metadata.update(extra_metadata)
    # Emit both `lambda_` (rate; upstream Perpetua naming) and `mean` (=1/lambda_;
    # what our existing ExponentialSurvivalFunctionGenerator expects). Keeping
    # both lets the same JSON feed either path without retraining.
    return {
        "params": {
            "lambda_": float(lambda_),
            "mean": float(1.0 / lambda_),
        },
        "confidence": min(1.0, n_samples / 30.0),
        "metadata": metadata,
    }


class SingleExponentialTrainer:
    """grouping ∈ {"class", "location"}."""

    def __init__(self, grouping: str = "location", name: str = "single_exponential") -> None:
        if grouping not in {"class", "location"}:
            raise ValueError(grouping)
        self.grouping = grouping
        self.name = name

    def train(self, dataset: TrainingDataset) -> dict[str, ParamEntry]:
        tracks = list(dataset.iter_tracks())
        global_lambda = _fit_lambda([t.duration_s for t in tracks])
        if global_lambda is None:
            raise RuntimeError(
                "SingleExponentialTrainer: no positive durations in training set."
            )

        out: dict[str, ParamEntry] = {
            GLOBAL_FALLBACK_KEY: _entry(
                global_lambda,
                n_samples=sum(1 for t in tracks if t.duration_s > 0.0),
                extra_metadata={"global_pool": True},
            )
        }

        for key, bucket in bucket_tracks(tracks, self.grouping).items():
            durations = [t.duration_s for t in bucket]
            lam = _fit_lambda(durations)
            if lam is None or math.isnan(lam) or math.isinf(lam):
                out[key] = _entry(
                    global_lambda,
                    n_samples=len(bucket),
                    fallback=True,
                )
            else:
                out[key] = _entry(lam, n_samples=len(bucket))
        return out
