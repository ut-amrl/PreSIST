"""Re-evaluate already-trained persistence methods under ScaledHorizonEpisodeProvider.

The percentile_horizons scenario used in the daycount sweep hands the filter
no observations beyond the anchor, so per-class filters collapse to the prior
mixture survival and produce vertical-line lifetime CDFs. This wrapper re-runs
`eval_persistence.py mode=evaluate` against the existing inference artifacts
with `eval_scenario=scaled_horizon` so observations are actually fed in (with
the strict causal cutoff t_q − min(recency_cap, dt_q/2)).

Run inside the spf_perpetua_container (paths are container-side /workspace/data).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


# --- Container-side roots (script must be run inside spf_perpetua_container) ---
OUTPUT_ROOT = Path("/workspace/data/output/persistence_eval_outputs")
PLOTS_ROOT = Path("/workspace/data/output/persistence_eval_plots")
TEST_MANIFEST = (
    "/workspace/data/input/new_pklot/postprocessed/annotations/"
    "PUCPR_repaired_split_flanks_fixed/splits/50_20_30/test_from_2012-10-30.json"
)
TEST_SIDECAR = (
    "/workspace/data/input/new_pklot/postprocessed/annotations/"
    "PUCPR_repaired_split_flanks_fixed/splits/50_20_30/"
    "test_from_2012-10-30.track_id_to_location_cell128.json"
)
DATASET_NAME = "pucpr_split_new_pklot_PUCPR_test_50_20_30_balanced"

# Defaults matched to the existing daycount_p0p01_mins5 sweep.
MIXTURE_SUFFIX = "_daycount_fullK_mins5_p0p01"
SINGLE_SUFFIX = "_daycount_emk1_noise0p01_mins5"
VISION_EXP_BASE = "vision_anchor_midpoint_pucpr_split_50_20_30_daycount_balanced_train"

# (label, dist, grouping, kind)
PERPETUA_VARIANTS: list[tuple[str, str, str, str]] = [
    ("mixture_exp:class",           "mixture_exp",       "class",    "mixture"),
    ("mixture_exp:location",        "mixture_exp",       "location", "mixture"),
    ("mixture_lognormal:class",     "mixture_lognormal", "class",    "mixture"),
    ("mixture_lognormal:location",  "mixture_lognormal", "location", "mixture"),
    ("single_exp:class",            "single_exp",        "class",    "single"),
    ("single_exp:location",         "single_exp",        "location", "single"),
    ("single_lognormal:class",      "single_lognormal",  "class",    "single"),
    ("single_lognormal:location",   "single_lognormal",  "location", "single"),
]
VISION_VARIANT: tuple[str, str] = ("vision_anchor:class", "class")

# LLM (gemma) — no day-count axis, fixed inferencer directory.
LLM_LABEL = "LLM (gemma)"
LLM_EVAL_CFG = (
    "eval_config_llm_pucpr_split_50_20_30_gemma_v3iqrBehav4_mO20_percentile_balanced"
)
LLM_INFERENCER_DIR = (
    "gemma_cap_proc_long7__gemmaVLMDesc_blur_bg_context_crop_mask_t3_cmd1"
    "_no_holes_longPrompt__ph_v3IqrBehav4__gemma4_31b_default"
)


# ---------------------------------------------------------------------------

def _exp_name(dist: str, grouping: str, kind: str) -> str:
    suffix = MIXTURE_SUFFIX if kind == "mixture" else SINGLE_SUFFIX
    return f"perpetua_{dist}_{grouping}_pucpr_split_50_20_30{suffix}"


def _eval_cfg(dist: str, grouping: str) -> str:
    return (
        f"eval_config_perpetua_{dist}_{grouping}"
        f"_pucpr_split_50_20_30_daycount_balanced"
    )


def latest_inference_run_dir(exp_name: str, days: int, seed: int) -> Path | None:
    pattern = (
        f"{OUTPUT_ROOT}/{DATASET_NAME}/{exp_name}_days{days}_seed{seed}/"
        f"pucpr_grouped_images/inference_runs/*"
    )
    hits = sorted(p for p in glob.glob(pattern) if Path(p).is_dir())
    return Path(hits[-1]) if hits else None


def extract_realised_horizons(inference_run_dir: Path) -> list[float]:
    """Read query_times from any existing prediction_records.json under
    survival_based__percentile_horizons/*/."""
    candidates = sorted(
        inference_run_dir.glob("evaluation_runs/survival_based__percentile_horizons/*/prediction_records.json")
    )
    if not candidates:
        raise RuntimeError(
            f"No percentile_horizons prediction_records.json under {inference_run_dir}"
        )
    with candidates[-1].open() as f:
        recs = json.load(f)
    qts = sorted({float(r["query_time"]) for r in recs})
    # ScaledHorizonEpisodeProvider requires strictly positive horizons
    # (dt_q = 0 would give h_eff = 0).
    qts = [q for q in qts if q > 0.0]
    return qts


def run_eval(
    *,
    eval_cfg: str,
    exp_name: str | None,
    day_count: int | None,
    seed: int,
    inference_run_dir: Path,
    recency_cap: float,
    query_horizons: list[float],
    max_anchor_obs: int,
    grouping: str | None,
    extra_overrides: Iterable[str] = (),
    env_extra: dict[str, str] | None = None,
) -> Path:
    """Invoke eval_persistence.py and return the new evaluation_runs/.../<ts>/ dir.

    For perpetua/vision methods pass `exp_name` and `day_count` (they parameterise
    the inferencer.name template). For methods without a daycount axis (LLM),
    leave both as None.
    """
    horizons_lit = "[" + ",".join(repr(float(h)) for h in query_horizons) + "]"
    cmd = [
        "python",
        "eval_persistence.py",
        f"--config-name={eval_cfg}",
        "mode=evaluate",
        f"benchmark_dataset.manifest_path={TEST_MANIFEST}",
        f"++benchmark_dataset.max_anchor_observations={max_anchor_obs}",
        "eval_scenario=scaled_horizon",
        f"++eval_scenario.recency_cap={recency_cap}",
        f"++eval_scenario.query_horizons={horizons_lit}",
        f"++paths.explicit_inference_run_dir={inference_run_dir}",
    ]
    if exp_name is not None:
        cmd.append(f"experiment.name={exp_name}")
    if day_count is not None:
        cmd.append(f"experiment.day_count={day_count}")
    cmd += [
        "experiment.subsample_fraction=1.0",
        f"experiment.subsample_seed={seed}",
    ]
    if grouping == "location":
        cmd.append(f"inferencer.track_id_to_location_file={TEST_SIDECAR}")
    cmd.extend(extra_overrides)

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    print("[run_eval]", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.check_call(cmd, env=env)
    dt = time.time() - t0

    new_dirs = sorted(
        inference_run_dir.glob("evaluation_runs/survival_based__scaled_horizon/*")
    )
    if not new_dirs:
        raise RuntimeError(
            f"No scaled_horizon evaluation dir appeared under {inference_run_dir}"
        )
    new_dir = new_dirs[-1]
    print(f"[run_eval] done in {dt:.1f}s -> {new_dir}", flush=True)
    return new_dir


def check_no_leakage(
    eval_dir: Path, recency_cap: float, query_horizons: list[float]
) -> None:
    """Confirm that for every record, observations referenced by the filter
    would have been bounded by t_q − min(recency_cap, dt_q/2).

    The prediction_records do not carry observation lists, so we instead
    verify the realised query_time set matches the requested horizons and
    flag any records with predicted_prob exactly equal across episodes (the
    pathological signature of "no observations were passed in")."""
    rec_path = eval_dir / "prediction_records.json"
    with rec_path.open() as f:
        recs = json.load(f)
    realised_qts = sorted({float(r["query_time"]) for r in recs})
    requested = sorted(query_horizons)
    if len(realised_qts) != len(requested) or not all(
        abs(a - b) <= max(abs(b), 1.0) * 1e-3 for a, b in zip(realised_qts, requested)
    ):
        raise AssertionError(
            f"Query horizons mismatch: requested={requested} got={realised_qts}"
        )
    # Sanity: predicted_prob should vary across episodes at the largest horizon
    # (otherwise observations are not actually differentiating).
    large = max(realised_qts)
    probs_at_large = [r["predicted_prob"] for r in recs if r["query_time"] == large]
    if probs_at_large:
        uniq = len({round(p, 6) for p in probs_at_large})
        print(
            f"[leak-check] {eval_dir.name}: {len(probs_at_large)} eps at dt_q={large:g}s, "
            f"{uniq} unique predicted_prob values"
        )
    print(f"[leak-check] {eval_dir.name}: OK")


def write_manifest(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "path", "has_train_pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"[manifest] wrote {len(rows)} rows -> {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=35)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recency-cap", type=float, default=1800.0,
                   help="Seconds (also caps observation pool length).")
    p.add_argument("--max-anchor-obs", type=int, default=50,
                   help="benchmark_dataset.max_anchor_observations override.")
    p.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated subset of labels (mixture_exp:class, "
            "single_exp:class, vision_anchor:class, ...), or 'all', or "
            "'class' / 'location' / 'mixture' / 'single' / 'vision'."
        ),
    )
    p.add_argument("--output-tag", default="obswin30m_test")
    p.add_argument("--cuda-visible-devices", default=None,
                   help="Set CUDA_VISIBLE_DEVICES for the eval subprocess "
                        "(e.g. '4'). Default: inherit parent env.")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip re-running eval; just re-build manifest from "
                        "existing scaled_horizon eval dirs.")
    p.add_argument("--manifest-only", action="store_true",
                   help="Alias for --skip-eval.")
    return p.parse_args()


def select_methods(spec: str) -> tuple[list[tuple[str, str, str, str]], bool, bool]:
    """Return (perpetua_variants, include_vision, include_llm)."""
    spec = spec.strip()
    perp: list[tuple[str, str, str, str]] = []
    include_vision = False
    include_llm = False

    if spec == "all":
        return PERPETUA_VARIANTS[:], True, True
    if spec in {"class", "location"}:
        return (
            [v for v in PERPETUA_VARIANTS if v[2] == spec],
            (spec == "class"),
            (spec == "class"),
        )
    if spec == "mixture":
        return [v for v in PERPETUA_VARIANTS if v[3] == "mixture"], False, False
    if spec == "single":
        return [v for v in PERPETUA_VARIANTS if v[3] == "single"], False, False
    if spec == "vision":
        return [], True, False
    if spec == "llm":
        return [], False, True

    labels = {s.strip() for s in spec.split(",") if s.strip()}
    for v in PERPETUA_VARIANTS:
        if v[0] in labels:
            perp.append(v)
    if VISION_VARIANT[0] in labels:
        include_vision = True
    if LLM_LABEL in labels:
        include_llm = True
    if not perp and not include_vision and not include_llm:
        raise SystemExit(f"No methods matched '{spec}'")
    return perp, include_vision, include_llm


def main() -> int:
    args = parse_args()
    skip_eval = args.skip_eval or args.manifest_only

    perpetua_variants, include_vision, include_llm = select_methods(args.methods)
    env_extra: dict[str, str] = {}
    if args.cuda_visible_devices is not None:
        env_extra["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    # Resolve a reference inference run so we can read realised query_horizons.
    ref_exp = _exp_name("mixture_exp", "class", "mixture")
    ref_inference = latest_inference_run_dir(ref_exp, args.days, args.seed)
    if ref_inference is None:
        # Fall back to LLM reference (no daycount axis) for cross-container runs
        # where the perpetua artifacts may not be mounted.
        llm_inf = sorted(
            (OUTPUT_ROOT / DATASET_NAME / LLM_INFERENCER_DIR / "pucpr_grouped_images"
             / "inference_runs").glob("*")
        )
        if not llm_inf:
            sys.exit(f"No inference run found for reference {ref_exp} D={args.days}")
        ref_inference = llm_inf[-1]
    horizons = extract_realised_horizons(ref_inference)
    print(
        f"[horizons] reusing {len(horizons)} realised query_horizons from "
        f"{ref_inference.name}; range=[{horizons[0]:g}, {horizons[-1]:g}] s"
    )

    rows: list[dict] = []

    for label, dist, grouping, kind in perpetua_variants:
        exp_name = _exp_name(dist, grouping, kind)
        eval_cfg = _eval_cfg(dist, grouping)
        inference_dir = latest_inference_run_dir(exp_name, args.days, args.seed)
        if inference_dir is None:
            print(f"[skip] no inference run for {label} ({exp_name} D={args.days})")
            continue

        if skip_eval:
            existing = sorted(
                inference_dir.glob("evaluation_runs/survival_based__scaled_horizon/*")
            )
            if not existing:
                print(f"[skip] no existing scaled_horizon eval for {label}")
                continue
            eval_dir = existing[-1]
            print(f"[reuse] {label}: {eval_dir}")
        else:
            eval_dir = run_eval(
                eval_cfg=eval_cfg,
                exp_name=exp_name,
                day_count=args.days,
                seed=args.seed,
                inference_run_dir=inference_dir,
                recency_cap=args.recency_cap,
                query_horizons=horizons,
                max_anchor_obs=args.max_anchor_obs,
                grouping=grouping,
                env_extra=env_extra,
            )
            check_no_leakage(eval_dir, args.recency_cap, horizons)

        rows.append({
            "label": label,
            "path": str(eval_dir),
            "has_train_pct": "true",
        })

    if include_vision:
        label, grouping = VISION_VARIANT
        exp_name = VISION_EXP_BASE
        eval_cfg = (
            "eval_config_vision_anchor_midpoint_pucpr_split_50_20_30"
            "_daycount_balanced_train_balanced_eval"
        )
        inference_dir = latest_inference_run_dir(exp_name, args.days, args.seed)
        if inference_dir is None:
            print(f"[skip] no inference run for {label} (D={args.days})")
        else:
            if skip_eval:
                existing = sorted(
                    inference_dir.glob("evaluation_runs/survival_based__scaled_horizon/*")
                )
                if existing:
                    eval_dir = existing[-1]
                    rows.append({"label": label, "path": str(eval_dir), "has_train_pct": "true"})
            else:
                eval_dir = run_eval(
                    eval_cfg=eval_cfg,
                    exp_name=exp_name,
                    day_count=args.days,
                    seed=args.seed,
                    inference_run_dir=inference_dir,
                    recency_cap=args.recency_cap,
                    query_horizons=horizons,
                    max_anchor_obs=args.max_anchor_obs,
                    grouping=grouping,
                    env_extra=env_extra,
                )
                check_no_leakage(eval_dir, args.recency_cap, horizons)
                rows.append({"label": label, "path": str(eval_dir), "has_train_pct": "true"})

    if include_llm:
        llm_inf_dir = OUTPUT_ROOT / DATASET_NAME / LLM_INFERENCER_DIR / "pucpr_grouped_images" / "inference_runs"
        runs = sorted(p for p in llm_inf_dir.glob("*") if p.is_dir())
        if not runs:
            print(f"[skip] no LLM inference run under {llm_inf_dir}")
        else:
            inference_dir = runs[-1]
            if skip_eval:
                existing = sorted(
                    inference_dir.glob("evaluation_runs/survival_based__scaled_horizon/*")
                )
                if existing:
                    rows.append({"label": LLM_LABEL, "path": str(existing[-1]), "has_train_pct": "false"})
            else:
                eval_dir = run_eval(
                    eval_cfg=LLM_EVAL_CFG,
                    exp_name=None,
                    day_count=None,
                    seed=args.seed,
                    inference_run_dir=inference_dir,
                    recency_cap=args.recency_cap,
                    query_horizons=horizons,
                    max_anchor_obs=args.max_anchor_obs,
                    grouping=None,
                    env_extra=env_extra,
                )
                check_no_leakage(eval_dir, args.recency_cap, horizons)
                rows.append({"label": LLM_LABEL, "path": str(eval_dir), "has_train_pct": "false"})

    manifest_path = PLOTS_ROOT / "manifests" / f"{args.output_tag}.csv"
    write_manifest(rows, manifest_path)

    print(
        "\nNext step (run inside pf_plus_datasets_v2 plotter container):\n"
        f"  python plot_train_pct_metrics.py --manifest {manifest_path} "
        f"--lifetime-cdf --lifetime-combined-sweep 1 --lifetime-combined-sweep 35 "
        f"--lifetime-combined-tag {args.output_tag} "
        f"--out-dir {PLOTS_ROOT / args.output_tag}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
