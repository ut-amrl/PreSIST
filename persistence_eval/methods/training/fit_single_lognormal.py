"""Single-lognormal MLE trainer.

Closed-form MLE on y_i = log(d_i):
  μ̂   = mean(y),  σ̂²  = var(y, ddof=0).

Param names `logmu`/`std` match upstream Perpetua's `priors.log_survival_lognormal`
signature so the same keys feed downstream without renaming.

Numerical floor: if a bucket has too few samples or all-equal durations
(σ̂ = 0), we fall back to the pooled "__global__" fit.
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
_MIN_SIGMA = 1e-3   # numerical floor to avoid σ=0 in survival evaluation


def _fit_mu_sigma(durations: list[float]) -> tuple[float, float] | None:
    """Closed-form MLE of (logmu, std) on log-durations. Returns None if degenerate."""
    pos = [d for d in durations if d > 0.0]
    if len(pos) < 2:
        return None
    logs = [math.log(d) for d in pos]
    n = len(logs)
    mu = sum(logs) / n
    var = sum((y - mu) ** 2 for y in logs) / n
    sigma = math.sqrt(var)
    if not math.isfinite(mu) or not math.isfinite(sigma):
        return None
    sigma = max(sigma, _MIN_SIGMA)
    return mu, sigma


def _entry(
    logmu: float,
    std: float,
    n_samples: int,
    fallback: bool = False,
    sigma_floored: bool = False,
    extra_metadata: dict | None = None,
) -> ParamEntry:
    metadata = {
        "n_samples": n_samples,
        "model_type": "single_lognormal",
    }
    if fallback:
        metadata["fallback_to_global"] = True
    if sigma_floored:
        metadata["sigma_floored"] = True
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "params": {"logmu": float(logmu), "std": float(std)},
        "confidence": min(1.0, n_samples / 30.0),
        "metadata": metadata,
    }


class SingleLogNormalTrainer:
    def __init__(self, grouping: str = "location", name: str = "single_lognormal") -> None:
        if grouping not in {"class", "location"}:
            raise ValueError(grouping)
        self.grouping = grouping
        self.name = name

    def train(self, dataset: TrainingDataset) -> dict[str, ParamEntry]:
        tracks = list(dataset.iter_tracks())
        global_fit = _fit_mu_sigma([t.duration_s for t in tracks])
        if global_fit is None:
            raise RuntimeError(
                "SingleLogNormalTrainer: not enough positive durations in training set."
            )
        g_mu, g_sigma = global_fit
        out: dict[str, ParamEntry] = {
            GLOBAL_FALLBACK_KEY: _entry(
                g_mu, g_sigma,
                n_samples=sum(1 for t in tracks if t.duration_s > 0.0),
                sigma_floored=(g_sigma == _MIN_SIGMA),
                extra_metadata={"global_pool": True},
            )
        }

        for key, bucket in bucket_tracks(tracks, self.grouping).items():
            fit = _fit_mu_sigma([t.duration_s for t in bucket])
            if fit is None:
                out[key] = _entry(
                    g_mu, g_sigma, n_samples=len(bucket), fallback=True,
                )
            else:
                mu, sigma = fit
                out[key] = _entry(
                    mu, sigma,
                    n_samples=len(bucket),
                    sigma_floored=(sigma == _MIN_SIGMA),
                )
        return out
