"""Single-exponential trainer.

Two fit methods, selected by the `fit_method` constructor arg:

  * ``"naive_mle"`` (default) — closed-form MLE on observed durations:
      ``λ̂ = N / Σ d_i``  (no censoring; assumes fully-observed tracks).
    Fast, dependency-free, but biased low when many tracks are right-censored
    inside the observation window.

  * ``"em_k1"`` — upstream-faithful K=1 fit. Delegates to
    :class:`MixtureExponentialTrainer` with ``params_range=[1]`` and
    ``num_retrain_attempts=0``, then re-shapes its mixture output back to the
    single-component schema below. Censoring is handled implicitly via the
    persistence-filter forward pass in upstream ``em_step``. Requires
    upstream perpetua + jax (run in ``spf_perpetua_container``).

Both methods emit the same param schema so downstream consumers don't care
which path was taken:

  ``{"lambda_": λ, "mean": 1/λ}`` per bucket plus a ``"__global__"`` entry.

`lambda_` matches upstream Perpetua's parameter naming so the same key feeds
``priors.log_survival_exponential`` without renaming.
"""

from __future__ import annotations

import math

from persistence_eval.methods.training.base import (
    ParamEntry,
    TrainingDataset,
    bucket_tracks,
)


GLOBAL_FALLBACK_KEY = "__global__"
_VALID_FIT_METHODS = {"naive_mle", "em_k1"}


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


def _translate_mixture_entry(mix_entry: ParamEntry) -> ParamEntry:
    """Re-shape a ``MixtureExponentialTrainer`` entry (K=1) into single-trainer schema."""
    lam = float(mix_entry["params"]["lambdas"][0])
    metadata = dict(mix_entry.get("metadata", {}))
    metadata["model_type"] = "single_exponential"
    metadata["fit_method"] = "em_k1"
    return {
        "params": {"lambda_": lam, "mean": 1.0 / lam},
        "confidence": mix_entry.get("confidence"),
        "metadata": metadata,
    }


class SingleExponentialTrainer:
    """grouping ∈ {"class", "location"}.

    `fit_method`:
        "naive_mle" (default) — closed-form `λ̂ = N / Σ d_i`. No censoring.
        "em_k1"               — K=1 EM via the upstream-faithful mixture trainer.

    EM-only params (`p_m`, `p_f`, `n_iter`, `delta`, `min_em_samples`) are
    ignored when `fit_method="naive_mle"`. They mirror the defaults of
    `MixtureExponentialTrainer`.
    """

    def __init__(
        self,
        grouping: str = "location",
        name: str = "single_exponential",
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
        global_lambda = _fit_lambda([t.duration_s for t in tracks])
        if global_lambda is None:
            raise RuntimeError(
                "SingleExponentialTrainer: no positive durations in training set."
            )

        out: dict[str, ParamEntry] = {
            GLOBAL_FALLBACK_KEY: _entry(
                global_lambda,
                n_samples=sum(1 for t in tracks if t.duration_s > 0.0),
                extra_metadata={"global_pool": True, "fit_method": "naive_mle"},
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
                    extra_metadata={"fit_method": "naive_mle"},
                )
            else:
                out[key] = _entry(
                    lam, n_samples=len(bucket),
                    extra_metadata={"fit_method": "naive_mle"},
                )
        return out

    # ------------------------------------------------------------------
    def _train_em_k1(self, dataset: TrainingDataset) -> dict[str, ParamEntry]:
        # Import lazily so the naive-MLE path doesn't pull jax in.
        from persistence_eval.methods.training.fit_mixture_exponential import (
            MixtureExponentialTrainer,
        )

        mix = MixtureExponentialTrainer(
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
