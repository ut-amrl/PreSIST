"""Mixture-exponential trainer — drives upstream em_step with model selection.

Wraps `src.learning.mixture_exponential.em_step` (vendored at src/perpetua-code/)
inside an upstream-style **select-and-refine** loop, mirroring
`src.learning.mixture_exponential.select_and_refine_model`:

  1. **K-sweep** — fit a mixture for each K in `params_range` (default {2, 3, 4}),
     initialised at the (k+1)/(K+1)-th quantiles of observed durations
     (matches upstream's `lambda_init = 1/quantiles`).
  2. **AIC selection** — pick the K with smallest AIC = -2·log_evidence + 2·(2K).
  3. **Pruning** — drop components with weight < `prune_threshold` and
     renormalise; reset best-AIC so the refinement step has freedom to
     re-evaluate.
  4. **Refinement** — `num_retrain_attempts` perturbed restarts of the chosen
     K (gaussian noise on quantile-derived `λ` and uniform-prior `π`); keep the
     run with the highest log-evidence.

Buckets with too few sequences fall back to a single-component fit, tagged
with `metadata["fallback_to_single"] = True`. Buckets that fail every
selection-and-refine attempt fall back to the global pooled fit.

The output param entry is consumed by `MixtureExponentialGenerator`:
    {"weights": [..K..], "lambdas": [..K..]}.
Metadata records `n_components_fitted`, `aic`, `bic`, `iters_run`,
`n_pruned_components`, and `n_refinement_improvements` for diagnostics.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Iterable

from tqdm.auto import tqdm

from persistence_eval.methods.training.base import (
    AbsTrainer,
    ParamEntry,
    TrainingDataset,
    TrainingTrack,
    bucket_tracks,
)


GLOBAL_FALLBACK_KEY = "__global__"

# Number of free parameters per mixture component for AIC. Exponential mixture:
# one rate λ_k and one weight π_k per component (with the K weights summing to
# 1, but upstream's convention uses 2K — keep parity).
_PARAMS_PER_COMPONENT_EXP = 2


# ---------------------------------------------------------------------------
# Helpers (initialisation + per-track sequence shaping)
# ---------------------------------------------------------------------------

def _quantile_lambdas(durations: list[float], K: int) -> list[float]:
    """λ_k = 1 / (k+1)/(K+1)-th duration quantile. Matches upstream init."""
    pos = sorted(d for d in durations if d > 0.0)
    if not pos:
        return [1.0] * K
    out = []
    for k in range(K):
        q = (k + 1) / (K + 1)
        idx = max(0, min(len(pos) - 1, int(q * len(pos))))
        d = pos[idx]
        out.append(1.0 / max(d, 1e-9))
    return out


def _track_to_seq(track: TrainingTrack) -> tuple[list[float], list[int]]:
    """Track-relative observation times. See module docstring rationale.

    Upstream's EM creates a FilterState with `initialization_time=0.0` and
    evaluates `S(t - init_time)`; passing seconds-since-epoch (~1.35e9 for
    PKLot) makes `exp(-λ·t)` underflow and breaks responsibilities entirely.

    Leading non-positive observations are dropped: the persistence model
    assumes alive at t=0, so a leading Absent (e.g., PKLot's
    `last_absent_before_ts`) is pre-emergence evidence the persistence
    likelihood can't represent and would only explain via the
    missed-detection prob. Anchor at the first Positive.
    """
    raw_times = list(track.observation_times)
    raw_pos = list(track.is_positive)
    first_pos = next((i for i, p in enumerate(raw_pos) if p), None)
    if first_pos is None:
        return [], []
    raw_times = raw_times[first_pos:]
    raw_pos = raw_pos[first_pos:]
    t0 = raw_times[0]
    times = [t - t0 for t in raw_times]
    y_bool = [1 if b else 0 for b in raw_pos]
    return times, y_bool


def _entry_single_fallback(durations: list[float], n_samples: int) -> ParamEntry | None:
    pos = [d for d in durations if d > 0.0]
    if not pos:
        return None
    mean = sum(pos) / len(pos)
    return {
        "params": {"weights": [1.0], "lambdas": [1.0 / mean]},
        "confidence": min(1.0, n_samples / 30.0),
        "metadata": {
            "n_samples": n_samples,
            "model_type": "mixture_exponential",
            "n_components_fitted": 1,
            "fallback_to_single": True,
        },
    }


def _perturb(values: list[float], scale: float, rng: random.Random) -> list[float]:
    """Multiplicative gaussian perturbation, clipped positive — matches upstream's
    `perturb()` helper which uses `clip(params + N(0, |p|·scale), min=1e-3)`."""
    return [max(1e-3, p + rng.gauss(0, abs(p) * scale)) for p in values]


# ---------------------------------------------------------------------------
class MixtureExponentialTrainer:
    """grouping ∈ {"class", "location"}.

    Model-selection params (mirroring upstream `select_and_refine_model`):
      params_range:           list of K values to try (default [2, 3, 4]).
      num_retrain_attempts:   perturbation restarts after pruning (default 5).
      prune_threshold:        drop components with π_k below this (default 1e-2).
      perturbation_scale_lambda / _pi: gaussian sigma for the refinement
        restarts (default 0.01 / 0.1 — upstream defaults).

    Per-K EM params:
      n_iter:                 max iterations per EM run (default 200).
      delta:                  absolute-tolerance convergence threshold on
                              log-evidence (default 1e-6, matches upstream
                              `fit()`).

    Observation model:
      p_m / p_f:              missed-detection / false-alarm probabilities,
                              identical to the matching MixturePersistenceFilter.

    Bucket-level:
      min_mixture_samples:    buckets below this fall back to a single-component
                              fit (no EM run).
    """

    def __init__(
        self,
        grouping: str = "location",
        params_range: list[int] | None = None,
        num_retrain_attempts: int = 5,
        prune_threshold: float = 1e-2,
        perturbation_scale_lambda: float = 0.01,
        perturbation_scale_pi: float = 0.1,
        p_m: float = 0.1,
        p_f: float = 0.05,
        min_mixture_samples: int = 20,
        delta: float = 1e-6,
        n_iter: int = 200,
        rng_seed: int = 42,
        name: str = "mixture_exponential",
    ) -> None:
        if grouping not in {"class", "location"}:
            raise ValueError(grouping)
        self.grouping = grouping
        self.name = name
        self.params_range = list(params_range) if params_range is not None else [2, 3, 4]
        if not self.params_range:
            raise ValueError("params_range must contain at least one K value")
        if any(K < 1 for K in self.params_range):
            raise ValueError(f"params_range entries must be >= 1; got {self.params_range}")
        self.num_retrain_attempts = int(num_retrain_attempts)
        self.prune_threshold = float(prune_threshold)
        self.perturbation_scale_lambda = float(perturbation_scale_lambda)
        self.perturbation_scale_pi = float(perturbation_scale_pi)
        self.p_m = float(p_m)
        self.p_f = float(p_f)
        self.min_mixture_samples = int(min_mixture_samples)
        self.delta = float(delta)
        self.n_iter = int(n_iter)
        self.rng_seed = int(rng_seed)

    # ------------------------------------------------------------------
    def _run_em(
        self,
        tracks: list[TrainingTrack],
        K: int,
        lambda_init: list[float],
        pi_init: list[float],
        progress_desc: str | None = None,
    ) -> tuple[list[float], list[float], float, int] | None:
        """One EM run with the given init. Returns (lambdas, pis, log_evidence,
        iters_run) or None on failure."""
        from persistence_eval.methods.filters.perpetua_filter_adapter import _jax_modules

        mods = _jax_modules()
        jnp = mods["jnp"]
        Partial = mods["Partial"]
        priors = mods["priors"]
        from src.learning.mixture_exponential import em_step
        from src.learning.learning_utils import create_filter, compute_marginals

        if not tracks:
            return None

        obs_times = [jnp.asarray(_track_to_seq(t)[0], dtype=jnp.float32) for t in tracks]
        y_bool = [jnp.asarray(_track_to_seq(t)[1], dtype=jnp.int32) for t in tracks]

        lambda_hat = jnp.asarray(lambda_init, dtype=jnp.float32)
        pi_hat = jnp.asarray(pi_init, dtype=jnp.float32)
        log_s = Partial(priors.log_survival_exponential)

        loss_prev = -float(jnp.inf)
        best_lambda, best_pi, best_loss = lambda_hat, pi_hat, loss_prev
        iters_run = 0

        iter_pbar = (
            tqdm(range(self.n_iter), desc=progress_desc, leave=False, unit="iter")
            if progress_desc is not None
            else None
        )
        iterator = iter_pbar if iter_pbar is not None else range(self.n_iter)

        for m in iterator:
            state = create_filter(
                "persistence", log_s, {"lambda_": lambda_hat}, pi_hat,
                initialization_time=0.0,
            )
            pi_bar, lambda_bar = em_step(
                "persistence", len(tracks), K, obs_times, y_bool, state,
                pi_hat, lambda_hat, self.p_m, self.p_f,
            )
            state2 = create_filter(
                "persistence", log_s, {"lambda_": lambda_bar}, pi_bar,
                initialization_time=0.0,
            )
            loss_t = 0.0
            for c in range(len(tracks)):
                log_evidence, _ = compute_marginals(
                    "persistence", state2, y_bool[c], obs_times[c], self.p_m, self.p_f
                )
                loss_t += float(log_evidence)
            iters_run = m + 1
            lambda_hat, pi_hat = lambda_bar, pi_bar
            if loss_t > best_loss and not math.isnan(loss_t):
                best_lambda, best_pi, best_loss = lambda_hat, pi_hat, loss_t
            if iter_pbar is not None:
                iter_pbar.set_postfix(K=K, log_evidence=f"{loss_t:.1f}")
            if math.isnan(loss_t):
                break
            if math.isfinite(loss_prev) and abs(loss_t - loss_prev) < self.delta:
                loss_prev = loss_t
                break
            loss_prev = loss_t

        if iter_pbar is not None:
            iter_pbar.close()

        if not math.isfinite(best_loss):
            return None
        return (
            [float(x) for x in best_lambda],
            [float(x) for x in best_pi],
            float(best_loss),
            iters_run,
        )

    # ------------------------------------------------------------------
    def _select_and_refine(
        self,
        tracks: list[TrainingTrack],
        durations: list[float],
        progress_desc: str | None = None,
    ) -> dict | None:
        """Upstream-style K sweep + AIC selection + pruning + perturbation refine.

        Returns a dict {lambdas, pis, K, aic, log_evidence, iters_run,
        n_pruned, n_refinement_improvements} or None on total failure.
        """
        best: dict | None = None
        best_K_for_each: dict[int, dict] = {}

        for K in self.params_range:
            lambda_init = _quantile_lambdas(durations, K)
            pi_init = [1.0 / K] * K
            result = self._run_em(
                tracks, K, lambda_init, pi_init,
                progress_desc=(f"{progress_desc} K={K}" if progress_desc else None),
            )
            if result is None:
                continue
            lambdas, pis, log_ev, iters = result
            aic = -2 * log_ev + 2 * _PARAMS_PER_COMPONENT_EXP * K
            entry = {
                "lambdas": lambdas, "pis": pis, "K": K,
                "log_evidence": log_ev, "aic": aic, "iters_run": iters,
            }
            best_K_for_each[K] = entry
            if best is None or aic < best["aic"]:
                best = entry

        if best is None:
            return None

        n_pruned = 0
        pre_prune: dict | None = None
        if any(w < self.prune_threshold for w in best["pis"]):
            keep = [i for i, w in enumerate(best["pis"]) if w >= self.prune_threshold]
            if not keep:
                # Pathological: all weights were below threshold. Skip pruning.
                pass
            elif len(keep) < len(best["pis"]):
                # Capture the full converged (λ, π) BEFORE we drop low-weight
                # components and renormalize. Surfaced via metadata so post-hoc
                # analyses can see what was discarded.
                pre_prune = {
                    "K": best["K"],
                    "lambdas": list(best["lambdas"]),
                    "pis": list(best["pis"]),
                    "kept_indices": list(keep),
                    "dropped_indices": [
                        i for i in range(len(best["pis"])) if i not in set(keep)
                    ],
                    "log_evidence": best["log_evidence"],
                    "aic": best["aic"],
                    "prune_threshold": self.prune_threshold,
                }
                pruned_lambdas = [best["lambdas"][i] for i in keep]
                pruned_pis = [best["pis"][i] for i in keep]
                s = sum(pruned_pis)
                pruned_pis = [w / s for w in pruned_pis]
                n_pruned = len(best["pis"]) - len(keep)
                best = {
                    "lambdas": pruned_lambdas,
                    "pis": pruned_pis,
                    "K": len(keep),
                    # Per upstream: invalidate AIC/log_evidence so refinement
                    # has unbiased freedom to update best.
                    "log_evidence": -float("inf"),
                    "aic": float("inf"),
                    "iters_run": best["iters_run"],
                }

        # Perturbation refinement around the chosen K.
        K = best["K"]
        n_refinement_improvements = 0
        if K >= 2 and self.num_retrain_attempts > 0:
            rng = random.Random(self.rng_seed)
            base_lambdas = _quantile_lambdas(durations, K)
            base_pis = [1.0 / K] * K
            for _ in range(self.num_retrain_attempts):
                lambda_init = _perturb(base_lambdas, self.perturbation_scale_lambda, rng)
                pi_init = _perturb(base_pis, self.perturbation_scale_pi, rng)
                s = sum(pi_init)
                if s <= 0:
                    continue
                pi_init = [w / s for w in pi_init]
                result = self._run_em(tracks, K, lambda_init, pi_init)
                if result is None:
                    continue
                lambdas, pis, log_ev, iters = result
                if log_ev > best["log_evidence"]:
                    best = {
                        "lambdas": lambdas, "pis": pis, "K": K,
                        "log_evidence": log_ev,
                        "aic": -2 * log_ev + 2 * _PARAMS_PER_COMPONENT_EXP * K,
                        "iters_run": iters,
                    }
                    n_refinement_improvements += 1

        best["n_pruned_components"] = n_pruned
        best["n_refinement_improvements"] = n_refinement_improvements
        if pre_prune is not None:
            best["pre_prune"] = pre_prune
        return best

    # ------------------------------------------------------------------
    def _fit_bucket(
        self,
        tracks: list[TrainingTrack],
        progress_desc: str | None = None,
    ) -> ParamEntry | None:
        """Full select-and-refine over a single bucket. None on total failure."""
        durations = [t.duration_s for t in tracks]
        if not any(d > 0.0 for d in durations):
            return None
        result = self._select_and_refine(tracks, durations, progress_desc)
        if result is None:
            return None

        # Sort components by lambda ascending for canonical output.
        K = result["K"]
        order = sorted(range(K), key=lambda i: result["lambdas"][i])
        lam_sorted = [result["lambdas"][i] for i in order]
        pi_sorted = [result["pis"][i] for i in order]
        s = sum(pi_sorted)
        if s <= 0:
            return None
        pi_sorted = [w / s for w in pi_sorted]

        n_total = len(tracks)
        return {
            "params": {"weights": pi_sorted, "lambdas": lam_sorted},
            "confidence": min(1.0, n_total / 30.0),
            "metadata": {
                "n_samples": n_total,
                "model_type": "mixture_exponential",
                "n_components_fitted": K,
                "iters_run": result["iters_run"],
                "final_log_evidence": result["log_evidence"],
                "aic": result["aic"],
                "n_pruned_components": result["n_pruned_components"],
                "n_refinement_improvements": result["n_refinement_improvements"],
                "params_range": list(self.params_range),
                "p_m": self.p_m,
                "p_f": self.p_f,
                **({"pre_prune": result["pre_prune"]} if "pre_prune" in result else {}),
            },
        }

    # ------------------------------------------------------------------
    def train(self, dataset: TrainingDataset) -> dict[str, ParamEntry]:
        tracks = list(dataset.iter_tracks())
        if not tracks:
            raise RuntimeError("MixtureExponentialTrainer: empty training set.")

        buckets = bucket_tracks(tracks, self.grouping)
        print(
            f"[MixtureExponentialTrainer] {len(tracks)} tracks across {len(buckets)} "
            f"buckets (grouping={self.grouping}); "
            f"min_mixture_samples={self.min_mixture_samples}, "
            f"params_range={self.params_range}, "
            f"refinement_attempts={self.num_retrain_attempts}, n_iter={self.n_iter}",
            flush=True,
        )

        print(
            f"[MixtureExponentialTrainer] fitting global pool ({len(tracks)} tracks)...",
            flush=True,
        )
        t0 = time.perf_counter()
        global_entry = (
            self._fit_bucket(tracks, progress_desc="global EM")
            or _entry_single_fallback([t.duration_s for t in tracks], len(tracks))
        )
        if global_entry is None:
            raise RuntimeError(
                "MixtureExponentialTrainer: no positive durations in training set."
            )
        global_entry["metadata"]["global_pool"] = True
        print(
            f"[MixtureExponentialTrainer] global pool done in {time.perf_counter()-t0:.1f}s "
            f"(K={global_entry['metadata'].get('n_components_fitted', '?')}, "
            f"iters={global_entry['metadata'].get('iters_run', '?')})",
            flush=True,
        )

        out: dict[str, ParamEntry] = {GLOBAL_FALLBACK_KEY: global_entry}

        n_mixture = 0
        n_fallback_too_small = 0   # bucket below min_mixture_samples → use global mixture
        n_fallback_em_failed = 0   # bucket large enough but EM failed → use global mixture
        K_seen: list[int] = []
        bucket_pbar = tqdm(
            buckets.items(),
            total=len(buckets),
            desc="bucket fits",
            unit="bucket",
        )
        for key, bucket in bucket_pbar:
            if len(bucket) < self.min_mixture_samples:
                # Per-bucket naive MLE on 1-2 (possibly censored) observations
                # produces wildly biased λ̂s; borrow strength from the full-pool
                # mixture fit instead.
                out[key] = _copy_with_fallback(global_entry, "too_small")
                n_fallback_too_small += 1
            else:
                entry = self._fit_bucket(bucket)
                if entry is None:
                    out[key] = _copy_with_fallback(global_entry, "em_failed")
                    n_fallback_em_failed += 1
                else:
                    out[key] = entry
                    n_mixture += 1
                    K_seen.append(entry["metadata"]["n_components_fitted"])
            median_K = sorted(K_seen)[len(K_seen) // 2] if K_seen else 0
            bucket_pbar.set_postfix(
                mix=n_mixture,
                fb_small=n_fallback_too_small,
                fb_em=n_fallback_em_failed,
                med_K=median_K,
            )
        bucket_pbar.close()
        if K_seen:
            from collections import Counter
            K_dist = dict(Counter(K_seen))
        else:
            K_dist = {}
        print(
            f"[MixtureExponentialTrainer] done. "
            f"mixture={n_mixture}, fallback_too_small={n_fallback_too_small}, "
            f"fallback_em_failed={n_fallback_em_failed}, K_distribution={K_dist}",
            flush=True,
        )
        return out


def _copy_with_fallback(entry: ParamEntry, kind: str) -> ParamEntry:
    new_meta = dict(entry.get("metadata", {}))
    new_meta["fallback_to_global"] = True
    new_meta["fallback_kind"] = kind
    return {
        "params": dict(entry["params"]),
        "confidence": entry.get("confidence"),
        "metadata": new_meta,
    }
