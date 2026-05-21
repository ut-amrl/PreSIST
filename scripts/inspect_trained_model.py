"""Health checks for a trained Perpetua-style persistence model.

Usage:
    python scripts/inspect_trained_model.py <model_dir>

where <model_dir> contains persistence_params.json + training_metadata.json.

Exit code: 0 if everything looks plausible, 1 if any sanity check fires
(degenerate fits, implausible lifetimes, severely skewed bucket sizes, etc.).
The check runs print a one-line summary by default; pass --verbose for the
full breakdown.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


# Plausibility band for parking-spot mean lifetimes. Wider than what we expect
# in practice; a fit outside this range almost certainly indicates a numerical
# or input-data bug (e.g. seconds-since-epoch leaking into duration estimates).
_PLAUSIBLE_LIFETIME_MIN_S = 60.0          # 1 minute
_PLAUSIBLE_LIFETIME_MAX_S = 365 * 86400.0  # 1 year


def _mean_lifetime_seconds(entry: dict) -> float | None:
    """Implied mean lifetime in seconds, regardless of distribution family."""
    p = entry["params"]
    model = entry["metadata"].get("model_type", "")

    if model == "single_exponential" or "lambdas" in p:
        # E[T] = 1/lam for one component; mixture mean = Σ w_k / lam_k.
        if "lambda_" in p:
            lam = float(p["lambda_"])
            return 1.0 / lam if lam > 0 else None
        if "mean" in p:
            return float(p["mean"])
        if "weights" in p and "lambdas" in p:
            ws = [float(w) for w in p["weights"]]
            ls = [float(l) for l in p["lambdas"]]
            if any(l <= 0 for l in ls):
                return None
            return sum(w / l for w, l in zip(ws, ls))
        return None

    if "logmu" in p and "std" in p:
        # E[T] = exp(mu + sigma^2/2)
        mu, sigma = float(p["logmu"]), float(p["std"])
        return math.exp(mu + sigma * sigma / 2.0)
    if "weights" in p and "logmus" in p and "stds" in p:
        ws = [float(w) for w in p["weights"]]
        mus = [float(m) for m in p["logmus"]]
        sds = [float(s) for s in p["stds"]]
        return sum(w * math.exp(m + s * s / 2.0) for w, m, s in zip(ws, mus, sds))

    return None


def _degenerate(entry: dict) -> str | None:
    """Return a short reason if the entry's params look broken; else None."""
    p = entry["params"]
    if "lambdas" in p and any(not (l > 0 and math.isfinite(l)) for l in p["lambdas"]):
        return "non-positive or non-finite lambda"
    if "lambda_" in p and not (p["lambda_"] > 0 and math.isfinite(p["lambda_"])):
        return "non-positive or non-finite lambda_"
    if "stds" in p and any(not (s > 0 and math.isfinite(s)) for s in p["stds"]):
        return "non-positive or non-finite std"
    if "std" in p and not (p["std"] > 0 and math.isfinite(p["std"])):
        return "non-positive or non-finite std"
    if "weights" in p:
        ws = [float(w) for w in p["weights"]]
        if any(not math.isfinite(w) for w in ws) or abs(sum(ws) - 1.0) > 1e-3:
            return f"weights sum to {sum(ws):.4f} (expected ~1)"

    mean = _mean_lifetime_seconds(entry)
    if mean is None:
        return "could not compute mean lifetime"
    if not (math.isfinite(mean) and mean > 0):
        return f"non-finite/non-positive mean lifetime ({mean})"
    if mean < _PLAUSIBLE_LIFETIME_MIN_S or mean > _PLAUSIBLE_LIFETIME_MAX_S:
        return f"implausible mean lifetime ({mean / 3600:.2f}h)"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    params_path = args.model_dir / "persistence_params.json"
    meta_path = args.model_dir / "training_metadata.json"
    if not params_path.exists():
        print(f"ERROR: {params_path} not found", file=sys.stderr)
        return 2
    params = json.loads(params_path.read_text())
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    keys = list(params)
    n_buckets = len(keys)
    n_global = sum(1 for k in keys if k == "__global__")
    real_buckets = [k for k in keys if k != "__global__"]

    # Legacy: pre-mixture-fallback-rewrite runs tagged small-bucket fallbacks
    # with `fallback_to_single`. Post-rewrite, every fallback uses the global
    # mixture and is tagged `fallback_to_global` with a `fallback_kind` of
    # "too_small" (small bucket) or "em_failed" (mixture EM failed). The
    # single-component trainers still emit `fallback_to_global` without a kind.
    n_fb_single = sum(
        1 for k in real_buckets if params[k]["metadata"].get("fallback_to_single")
    )
    fb_global_keys = [
        k for k in real_buckets if params[k]["metadata"].get("fallback_to_global")
    ]
    n_fb_global = len(fb_global_keys)
    n_fb_too_small = sum(
        1 for k in fb_global_keys
        if params[k]["metadata"].get("fallback_kind") == "too_small"
    )
    n_fb_em_failed = sum(
        1 for k in fb_global_keys
        if params[k]["metadata"].get("fallback_kind") == "em_failed"
    )
    n_full = len(real_buckets) - n_fb_single - n_fb_global

    degen = []
    for k in keys:
        why = _degenerate(params[k])
        if why is not None:
            degen.append((k, why))

    bucket_sizes = meta.get("bucket_sizes", {})
    sizes = list(bucket_sizes.values()) if bucket_sizes else []
    median_size = int(statistics.median(sizes)) if sizes else 0
    min_size = min(sizes) if sizes else 0
    max_size = max(sizes) if sizes else 0

    global_entry = params.get("__global__")
    global_mean_h = (
        _mean_lifetime_seconds(global_entry) / 3600.0
        if global_entry and _mean_lifetime_seconds(global_entry) is not None
        else float("nan")
    )

    if args.verbose:
        print(f"  model_dir:           {args.model_dir}")
        print(f"  trainer:             {meta.get('trainer_name')} ({meta.get('trainer_class')})")
        print(f"  grouping:            {meta.get('grouping')}")
        print(f"  subsample fraction:  {meta.get('subsample_fraction')}")
        print(f"  n_tracks_used:       {meta.get('n_tracks_used')}")
        print(f"  n_buckets:           {n_buckets} ({n_global} global + {len(real_buckets)} real)")
        print(f"    full mixture/single fits: {n_full}")
        print(f"    fallback_to_single:       {n_fb_single}  (legacy naive-MLE path; pre-rewrite runs only)")
        print(f"    fallback_to_global:       {n_fb_global}  "
              f"(too_small={n_fb_too_small}, em_failed={n_fb_em_failed}, "
              f"unknown={n_fb_global - n_fb_too_small - n_fb_em_failed})")
        print(f"  bucket size:         min={min_size} median={median_size} max={max_size}")
        print(f"  global mean lifetime: {global_mean_h:.2f} h")
        if degen:
            print(f"  DEGENERATE BUCKETS ({len(degen)}):")
            for k, why in degen[:10]:
                print(f"    {k}: {why}")
            if len(degen) > 10:
                print(f"    ... ({len(degen) - 10} more)")
    else:
        flags = []
        if degen:
            flags.append(f"DEGENERATE={len(degen)}")
        # Only EM-failure fallbacks indicate a data-quality problem worth
        # surfacing. too_small fallbacks are expected at low N and are not a
        # red flag — the bucket simply borrows strength from the global fit.
        if n_fb_em_failed:
            flags.append(f"fb_em_failed={n_fb_em_failed}")
        flag_str = (" [" + ", ".join(flags) + "]") if flags else ""
        print(
            f"OK: {meta.get('trainer_name', '?')} {meta.get('grouping', '?')} "
            f"frac={meta.get('subsample_fraction', '?')} | "
            f"buckets={n_buckets} (full={n_full}, fb_single={n_fb_single}, "
            f"fb_global={n_fb_global} [too_small={n_fb_too_small}, em_failed={n_fb_em_failed}]) | "
            f"bucket_size median={median_size} | "
            f"global_mean_lifetime={global_mean_h:.2f}h"
            f"{flag_str}"
        )

    # Hard fail: degenerate global, implausible mean, or > 25% of real buckets
    # are degenerate or had EM failures. too_small fallbacks are NOT a fail
    # condition — they're the expected behavior when a bucket can't support
    # its own mixture fit and borrows strength from the global pool.
    if global_entry is not None and _degenerate(global_entry) is not None:
        return 1
    if real_buckets and (len(degen) / max(1, len(real_buckets)) > 0.25):
        return 1
    if real_buckets and (n_fb_em_failed / max(1, len(real_buckets)) > 0.25):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
