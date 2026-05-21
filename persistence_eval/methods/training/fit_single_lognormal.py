"""Single-lognormal trainer.

Two fit methods, selected by the `fit_method` constructor arg:

  * ``"naive_mle"`` (default) — closed-form MLE on log(d_i):
      ``μ̂ = mean(y),  σ̂² = var(y, ddof=0)``  where ``y = log(d)``.
    Fast, dependency-free, but biased when many tracks are right-censored in
    the observation window.

  * ``"em_k1"`` — upstream-faithful K=1 fit. Delegates to
    :class:`MixtureLogNormalTrainer` with ``params_range=[1]`` and
    ``num_retrain_attempts=0``, then re-shapes its mixture output back to the
    single-component schema below. Censoring is handled implicitly via the
    persistence-filter forward pass in upstream ``em_step``. Requires
    upstream perpetua + jax (run in ``spf_perpetua_container``).

Both methods emit the same param schema:

  ``{"logmu": μ, "std": σ}`` per bucket plus a ``"__global__"`` entry.

Param names `logmu`/`std` match upstream Perpetua's
``priors.log_survival_lognormal`` signature so the same keys feed downstream
without renaming.

Numerical floor: if a bucket has too few samples or all-equal durations
(σ̂ = 0), we fall back to the pooled "__global__" fit (naive-MLE path only).
"""

from __future__ import annotations

import math

from persistence_eval.methods.training.base import (
    ParamEntry,
    TrainingDataset,
    bucket_tracks,
)


GLOBAL_FALLBACK_KEY = "__global__"
_MIN_SIGMA = 1e-3   # numerical floor to avoid σ=0 in survival evaluation
_VALID_FIT_METHODS = {"naive_mle", "em_k1"}


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


def _translate_mixture_entry(mix_entry: ParamEntry) -> ParamEntry:
    """Re-shape a ``MixtureLogNormalTrainer`` entry (K=1) into single-trainer schema."""
    logmu = float(mix_entry["params"]["logmus"][0])
    std = float(mix_entry["params"]["stds"][0])
    metadata = dict(mix_entry.get("metadata", {}))
    metadata["model_type"] = "single_lognormal"
    metadata["fit_method"] = "em_k1"
    return {
        "params": {"logmu": logmu, "std": std},
        "confidence": mix_entry.get("confidence"),
        "metadata": metadata,
    }


class SingleLogNormalTrainer:
    """grouping ∈ {"class", "location"}.

    `fit_method`:
        "naive_mle" (default) — closed-form `(μ̂, σ̂)` on `log(d)`. No censoring.
        "em_k1"               — K=1 EM via the upstream-faithful mixture trainer.
    """

    def __init__(
        self,
        grouping: str = "location",
        name: str = "single_lognormal",
        fit_method: str = "naive_mle",
        p_m: float = 0.1,
        p_f: float = 0.05,
        n_iter: int = 200,
        delta: float = 1e-6,
        min_em_samples: int = 20,
    ) -> None:
        if grouping not in {"class", "location"}:
            raise ValueError(grouping)
        if fit_method not in _VALID_FIT_METHODS:
            raise ValueError(
                f"fit_method must be one of {_VALID_FIT_METHODS}; got {fit_method!r}"
            )
        self.grouping = grouping
        self.name = name
        self.fit_method = fit_method
        self.p_m = float(p_m)
        self.p_f = float(p_f)
        self.n_iter = int(n_iter)
        self.delta = float(delta)
        self.min_em_samples = int(min_em_samples)

    def train(self, dataset: TrainingDataset) -> dict[str, ParamEntry]:
        if self.fit_method == "em_k1":
            return self._train_em_k1(dataset)
        return self._train_naive_mle(dataset)

    # ------------------------------------------------------------------
    def _train_naive_mle(self, dataset: TrainingDataset) -> dict[str, ParamEntry]:
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
                extra_metadata={"global_pool": True, "fit_method": "naive_mle"},
            )
        }

        for key, bucket in bucket_tracks(tracks, self.grouping).items():
            fit = _fit_mu_sigma([t.duration_s for t in bucket])
            if fit is None:
                out[key] = _entry(
                    g_mu, g_sigma, n_samples=len(bucket), fallback=True,
                    extra_metadata={"fit_method": "naive_mle"},
                )
            else:
                mu, sigma = fit
                out[key] = _entry(
                    mu, sigma,
                    n_samples=len(bucket),
                    sigma_floored=(sigma == _MIN_SIGMA),
                    extra_metadata={"fit_method": "naive_mle"},
                )
        return out

    # ------------------------------------------------------------------
    def _train_em_k1(self, dataset: TrainingDataset) -> dict[str, ParamEntry]:
        # Import lazily so the naive-MLE path doesn't pull jax in.
        from persistence_eval.methods.training.fit_mixture_lognormal import (
            MixtureLogNormalTrainer,
        )

        mix = MixtureLogNormalTrainer(
            grouping=self.grouping,
            params_range=[1],
            num_retrain_attempts=0,
            p_m=self.p_m,
            p_f=self.p_f,
            n_iter=self.n_iter,
            delta=self.delta,
            min_mixture_samples=self.min_em_samples,
            name=self.name,
        )
        mix_out = mix.train(dataset)
        return {k: _translate_mixture_entry(v) for k, v in mix_out.items()}
