"""Mixture-lognormal trainer — drives upstream em_step with model selection.

Mirrors `fit_mixture_exponential.py` (see that module for the full select-and-
refine description). Only the EM primitive and the parameter set differ:
upstream's `mixture_lognormal.em_step` returns `(pi, mu, sigma)` per K, and we
use `3K` free parameters per AIC formula (logmu, std, weight per component).

Outputs param entries consumed by `MixtureLogNormalGenerator`:
    {"weights": [..K..], "logmus": [..K..], "stds": [..K..]}.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter

from tqdm.auto import tqdm

from persistence_eval.methods.training.base import (
    AbsTrainer,
    ParamEntry,
    TrainingDataset,
    TrainingTrack,
    bucket_tracks,
)


GLOBAL_FALLBACK_KEY = "__global__"
_MIN_SIGMA = 1e-3
_PARAMS_PER_COMPONENT_LOGNORMAL = 3   # logmu + std + weight per component


def _quantile_inits_lognormal(
    durations: list[float], K: int
) -> tuple[list[float], list[float]]:
    """logmu_k = log(quantile_k); std_k = 1.0. Matches upstream's init."""
    pos = sorted(d for d in durations if d > 0.0)
    if not pos:
        return ([0.0] * K, [1.0] * K)
    logmus = []
    for k in range(K):
        q = (k + 1) / (K + 1)
        idx = max(0, min(len(pos) - 1, int(q * len(pos))))
        d = pos[idx]
        logmus.append(math.log(max(d, 1e-9)))
    stds = [1.0] * K
    return logmus, stds


def _track_to_seq(track: TrainingTrack) -> tuple[list[float], list[int]]:
    """Track-relative observation times. See fit_mixture_exponential for rationale."""
    raw = list(track.observation_times)
    if not raw:
        return [], []
    t0 = min(raw)
    times = [t - t0 for t in raw]
    y_bool = [1 if b else 0 for b in track.is_positive]
    return times, y_bool


def _entry_single_fallback(
    durations: list[float], n_samples: int
) -> ParamEntry | None:
    pos = [d for d in durations if d > 0.0]
    if len(pos) < 2:
        return None
    logs = [math.log(d) for d in pos]
    n = len(logs)
    mu = sum(logs) / n
    var = sum((y - mu) ** 2 for y in logs) / n
    sigma = max(math.sqrt(var), _MIN_SIGMA)
    return {
        "params": {"weights": [1.0], "logmus": [mu], "stds": [sigma]},
        "confidence": min(1.0, n_samples / 30.0),
        "metadata": {
            "n_samples": n_samples,
            "model_type": "mixture_lognormal",
            "n_components_fitted": 1,
            "fallback_to_single": True,
            "sigma_floored": (sigma == _MIN_SIGMA),
        },
    }


def _perturb(values: list[float], scale: float, rng: random.Random) -> list[float]:
    return [max(1e-3, p + rng.gauss(0, abs(p) * scale)) for p in values]


# ---------------------------------------------------------------------------
class MixtureLogNormalTrainer:
    """Same model-selection regime as MixtureExponentialTrainer; see that
    docstring. The lognormal-specific perturbation scales apply to logmu and
    sigma respectively (kept symmetric with upstream's defaults)."""

    def __init__(
        self,
        grouping: str = "location",
        params_range: list[int] | None = None,
        num_retrain_attempts: int = 5,
        prune_threshold: float = 1e-2,
        perturbation_scale_logmu: float = 0.01,
        perturbation_scale_std: float = 0.01,
        perturbation_scale_pi: float = 0.1,
        p_m: float = 0.1,
        p_f: float = 0.05,
        min_mixture_samples: int = 20,
        delta_log_evidence_rel: float = 1e-5,
        min_iters: int = 5,
        n_iter: int = 200,
        rng_seed: int = 42,
        name: str = "mixture_lognormal",
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
        self.perturbation_scale_logmu = float(perturbation_scale_logmu)
        self.perturbation_scale_std = float(perturbation_scale_std)
        self.perturbation_scale_pi = float(perturbation_scale_pi)
        self.p_m = float(p_m)
        self.p_f = float(p_f)
        self.min_mixture_samples = int(min_mixture_samples)
        self.delta_log_evidence_rel = float(delta_log_evidence_rel)
        self.min_iters = int(min_iters)
        self.n_iter = int(n_iter)
        self.rng_seed = int(rng_seed)

    # ------------------------------------------------------------------
    def _run_em(
        self,
        tracks: list[TrainingTrack],
        K: int,
        logmu_init: list[float],
        std_init: list[float],
        pi_init: list[float],
        progress_desc: str | None = None,
    ) -> tuple[list[float], list[float], list[float], float, int] | None:
        """One EM run with the given init. Returns (logmus, stds, pis, log_evidence,
        iters_run) or None on failure."""
        from persistence_eval.methods.filters.perpetua_filter_adapter import _jax_modules

        mods = _jax_modules()
        jnp = mods["jnp"]
        Partial = mods["Partial"]
        priors = mods["priors"]
        from src.learning.mixture_lognormal import em_step
        from src.learning.learning_utils import create_filter, compute_marginals

        if not tracks:
            return None

        obs_times = [jnp.asarray(_track_to_seq(t)[0], dtype=jnp.float32) for t in tracks]
        y_bool = [jnp.asarray(_track_to_seq(t)[1], dtype=jnp.int32) for t in tracks]

        mu_hat = jnp.asarray(logmu_init, dtype=jnp.float32)
        sigma_hat = jnp.asarray(std_init, dtype=jnp.float32)
        pi_hat = jnp.asarray(pi_init, dtype=jnp.float32)
        log_s = Partial(priors.log_survival_lognormal)

        loss_prev = -float(jnp.inf)
        best_mu, best_sigma, best_pi, best_loss = mu_hat, sigma_hat, pi_hat, loss_prev
        iters_run = 0

        iter_pbar = (
            tqdm(range(self.n_iter), desc=progress_desc, leave=False, unit="iter")
            if progress_desc is not None
            else None
        )
        iterator = iter_pbar if iter_pbar is not None else range(self.n_iter)

        for m in iterator:
            state = create_filter(
                "persistence", log_s,
                {"logmu": mu_hat, "std": sigma_hat},
                pi_hat,
                initialization_time=0.0,
            )
            pi_bar, mu_bar, sigma_bar = em_step(
                "persistence", len(tracks), K,
                obs_times, y_bool, state,
                pi_hat, mu_hat, sigma_hat,
                self.p_m, self.p_f,
            )
            state2 = create_filter(
                "persistence", log_s,
                {"logmu": mu_bar, "std": sigma_bar},
                pi_bar,
                initialization_time=0.0,
            )
            loss_t = 0.0
            for c in range(len(tracks)):
                log_evidence, _ = compute_marginals(
                    "persistence", state2, y_bool[c], obs_times[c], self.p_m, self.p_f
                )
                loss_t += float(log_evidence)
            iters_run = m + 1
            mu_hat, sigma_hat, pi_hat = mu_bar, sigma_bar, pi_bar
            if loss_t > best_loss and not math.isnan(loss_t):
                best_mu, best_sigma, best_pi, best_loss = mu_hat, sigma_hat, pi_hat, loss_t
            if iter_pbar is not None:
                iter_pbar.set_postfix(K=K, log_evidence=f"{loss_t:.1f}")
            if math.isnan(loss_t):
                break
            if iters_run >= self.min_iters and math.isfinite(loss_prev):
                rel = abs(loss_t - loss_prev) / max(abs(loss_prev), 1.0)
                if rel < self.delta_log_evidence_rel:
                    loss_prev = loss_t
                    break
            loss_prev = loss_t

        if iter_pbar is not None:
            iter_pbar.close()

        if not math.isfinite(best_loss):
            return None
        return (
            [float(x) for x in best_mu],
            [max(_MIN_SIGMA, float(x)) for x in best_sigma],
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
        best: dict | None = None

        for K in self.params_range:
            logmu_init, std_init = _quantile_inits_lognormal(durations, K)
            pi_init = [1.0 / K] * K
            result = self._run_em(
                tracks, K, logmu_init, std_init, pi_init,
                progress_desc=(f"{progress_desc} K={K}" if progress_desc else None),
            )
            if result is None:
                continue
            logmus, stds, pis, log_ev, iters = result
            aic = -2 * log_ev + 2 * _PARAMS_PER_COMPONENT_LOGNORMAL * K
            entry = {
                "logmus": logmus, "stds": stds, "pis": pis, "K": K,
                "log_evidence": log_ev, "aic": aic, "iters_run": iters,
            }
            if best is None or aic < best["aic"]:
                best = entry

        if best is None:
            return None

        n_pruned = 0
        if any(w < self.prune_threshold for w in best["pis"]):
            keep = [i for i, w in enumerate(best["pis"]) if w >= self.prune_threshold]
            if keep and len(keep) < len(best["pis"]):
                pruned_logmus = [best["logmus"][i] for i in keep]
                pruned_stds = [best["stds"][i] for i in keep]
                pruned_pis = [best["pis"][i] for i in keep]
                s = sum(pruned_pis)
                pruned_pis = [w / s for w in pruned_pis]
                n_pruned = len(best["pis"]) - len(keep)
                best = {
                    "logmus": pruned_logmus, "stds": pruned_stds,
                    "pis": pruned_pis, "K": len(keep),
                    "log_evidence": -float("inf"), "aic": float("inf"),
                    "iters_run": best["iters_run"],
                }

        K = best["K"]
        n_refinement_improvements = 0
        if K >= 2 and self.num_retrain_attempts > 0:
            rng = random.Random(self.rng_seed)
            base_logmus, base_stds = _quantile_inits_lognormal(durations, K)
            base_pis = [1.0 / K] * K
            for _ in range(self.num_retrain_attempts):
                logmu_init = _perturb(base_logmus, self.perturbation_scale_logmu, rng)
                std_init = _perturb(base_stds, self.perturbation_scale_std, rng)
                pi_init = _perturb(base_pis, self.perturbation_scale_pi, rng)
                s = sum(pi_init)
                if s <= 0:
                    continue
                pi_init = [w / s for w in pi_init]
                result = self._run_em(tracks, K, logmu_init, std_init, pi_init)
                if result is None:
                    continue
                logmus, stds, pis, log_ev, iters = result
                if log_ev > best["log_evidence"]:
                    best = {
                        "logmus": logmus, "stds": stds, "pis": pis, "K": K,
                        "log_evidence": log_ev,
                        "aic": -2 * log_ev + 2 * _PARAMS_PER_COMPONENT_LOGNORMAL * K,
                        "iters_run": iters,
                    }
                    n_refinement_improvements += 1

        best["n_pruned_components"] = n_pruned
        best["n_refinement_improvements"] = n_refinement_improvements
        return best

    # ------------------------------------------------------------------
    def _fit_bucket(
        self,
        tracks: list[TrainingTrack],
        progress_desc: str | None = None,
    ) -> ParamEntry | None:
        durations = [t.duration_s for t in tracks]
        if not any(d > 0.0 for d in durations):
            return None
        result = self._select_and_refine(tracks, durations, progress_desc)
        if result is None:
            return None

        K = result["K"]
        order = sorted(range(K), key=lambda i: result["logmus"][i])
        logmus_sorted = [result["logmus"][i] for i in order]
        stds_sorted = [result["stds"][i] for i in order]
        pi_sorted = [result["pis"][i] for i in order]
        s = sum(pi_sorted)
        if s <= 0:
            return None
        pi_sorted = [w / s for w in pi_sorted]

        n_total = len(tracks)
        return {
            "params": {"weights": pi_sorted, "logmus": logmus_sorted, "stds": stds_sorted},
            "confidence": min(1.0, n_total / 30.0),
            "metadata": {
                "n_samples": n_total,
                "model_type": "mixture_lognormal",
                "n_components_fitted": K,
                "iters_run": result["iters_run"],
                "final_log_evidence": result["log_evidence"],
                "aic": result["aic"],
                "n_pruned_components": result["n_pruned_components"],
                "n_refinement_improvements": result["n_refinement_improvements"],
                "params_range": list(self.params_range),
                "p_m": self.p_m,
                "p_f": self.p_f,
            },
        }

    # ------------------------------------------------------------------
    def train(self, dataset: TrainingDataset) -> dict[str, ParamEntry]:
        tracks = list(dataset.iter_tracks())
        if not tracks:
            raise RuntimeError("MixtureLogNormalTrainer: empty training set.")

        buckets = bucket_tracks(tracks, self.grouping)
        print(
            f"[MixtureLogNormalTrainer] {len(tracks)} tracks across {len(buckets)} "
            f"buckets (grouping={self.grouping}); "
            f"min_mixture_samples={self.min_mixture_samples}, "
            f"params_range={self.params_range}, "
            f"refinement_attempts={self.num_retrain_attempts}, n_iter={self.n_iter}",
            flush=True,
        )

        print(
            f"[MixtureLogNormalTrainer] fitting global pool ({len(tracks)} tracks)...",
            flush=True,
        )
        t0 = time.perf_counter()
        global_entry = (
            self._fit_bucket(tracks, progress_desc="global EM")
            or _entry_single_fallback([t.duration_s for t in tracks], len(tracks))
        )
        if global_entry is None:
            raise RuntimeError(
                "MixtureLogNormalTrainer: not enough positive durations in training set."
            )
        global_entry["metadata"]["global_pool"] = True
        print(
            f"[MixtureLogNormalTrainer] global pool done in {time.perf_counter()-t0:.1f}s "
            f"(K={global_entry['metadata'].get('n_components_fitted', '?')}, "
            f"iters={global_entry['metadata'].get('iters_run', '?')})",
            flush=True,
        )
        out: dict[str, ParamEntry] = {GLOBAL_FALLBACK_KEY: global_entry}

        n_mixture = 0
        n_fallback_single = 0
        n_fallback_global = 0
        K_seen: list[int] = []
        bucket_pbar = tqdm(
            buckets.items(),
            total=len(buckets),
            desc="bucket fits",
            unit="bucket",
        )
        for key, bucket in bucket_pbar:
            if len(bucket) < self.min_mixture_samples:
                fb = _entry_single_fallback([t.duration_s for t in bucket], len(bucket))
                out[key] = fb if fb is not None else _copy_with_fallback(global_entry)
                n_fallback_single += 1
            else:
                entry = self._fit_bucket(bucket)
                if entry is None:
                    out[key] = _copy_with_fallback(global_entry)
                    n_fallback_global += 1
                else:
                    out[key] = entry
                    n_mixture += 1
                    K_seen.append(entry["metadata"]["n_components_fitted"])
            median_K = sorted(K_seen)[len(K_seen) // 2] if K_seen else 0
            bucket_pbar.set_postfix(
                mix=n_mixture,
                fb_single=n_fallback_single,
                fb_global=n_fallback_global,
                med_K=median_K,
            )
        bucket_pbar.close()
        K_dist = dict(Counter(K_seen)) if K_seen else {}
        print(
            f"[MixtureLogNormalTrainer] done. "
            f"mixture={n_mixture}, fallback_single={n_fallback_single}, "
            f"fallback_global={n_fallback_global}, K_distribution={K_dist}",
            flush=True,
        )
        return out


def _copy_with_fallback(entry: ParamEntry) -> ParamEntry:
    new_meta = dict(entry.get("metadata", {}))
    new_meta["fallback_to_global"] = True
    return {
        "params": dict(entry["params"]),
        "confidence": entry.get("confidence"),
        "metadata": new_meta,
    }
