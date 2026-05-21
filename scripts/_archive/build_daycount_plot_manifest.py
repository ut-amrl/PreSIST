"""Build a plot manifest CSV pointing at the latest percentile_horizons
metric_results.json for every cell in the day-count sweep at p=0.01, min=5.

Covers:
  - 8 perpetua variants × 11 D values (curves)
  - vision_anchor:class × N D values that exist (curve)
  - LLM gemma (baseline; no train axis)

Output CSV header: label,path,has_train_pct  (consumed by plot_train_pct_metrics.py).
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path


# Host path (for globbing on the host filesystem) and the in-container
# rewrite of those paths (what gets written into the manifest, since the
# plotter runs inside pf_plus_datasets_v2).
HOST_ROOT = Path("/scratch/aaadkins/data/semantic-persistence-filter/output/persistence_eval_outputs/pucpr_split_new_pklot_PUCPR_test_50_20_30_balanced")
CONTAINER_ROOT = "/workspace/data/output/persistence_eval_outputs/pucpr_split_new_pklot_PUCPR_test_50_20_30_balanced"
EVAL_ROOT = HOST_ROOT
SCENARIO_GLOB = "*/pucpr_grouped_images/inference_runs/*/evaluation_runs/survival_based__percentile_horizons/*"


PERPETUA_DAYS = [1, 2, 3, 4, 5, 7, 10, 14, 21, 28, 35]
VISION_DAYS = [1, 2, 3, 4, 5, 7, 10, 14, 21, 28, 35]   # vision-anchor cells on disk

# Mapping label -> (exp_name template, days list, has_train_pct=True)
PERPETUA_VARIANTS = [
    # (label, dist, grouping, kind)
    ("mixture_exp:class",          "mixture_exp",       "class",    "mixture"),
    ("mixture_exp:location",       "mixture_exp",       "location", "mixture"),
    ("mixture_lognormal:class",    "mixture_lognormal", "class",    "mixture"),
    ("mixture_lognormal:location", "mixture_lognormal", "location", "mixture"),
    ("single_exp:class",           "single_exp",        "class",    "single"),
    ("single_exp:location",        "single_exp",        "location", "single"),
    ("single_lognormal:class",     "single_lognormal",  "class",    "single"),
    ("single_lognormal:location",  "single_lognormal",  "location", "single"),
]


def _to_container(host_path: str) -> str:
    """Rewrite a host-side eval path into the container-side path."""
    return host_path.replace(str(HOST_ROOT), CONTAINER_ROOT, 1)


def latest_metric_dir(exp_name: str, days: int, seed: int = 0) -> str | None:
    """Return the latest container-side TS_EVAL directory containing
    metric_results.json for this exp / days / seed cell. None if no match."""
    pattern = f"{EVAL_ROOT}/{exp_name}_days{days}_seed{seed}/pucpr_grouped_images/" \
              f"inference_runs/*/evaluation_runs/survival_based__percentile_horizons/*"
    hits = sorted(glob.glob(pattern))
    hits = [h for h in hits if (Path(h) / "metric_results.json").is_file()]
    if not hits:
        return None
    return _to_container(hits[-1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True,
                   help="Output manifest CSV path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grouping", choices=["all", "class", "location"], default="all",
                   help="Filter perpetua variants by grouping. Vision-anchor "
                        "(class) and LLM (no grouping) are always included.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict] = []

    # 8 perpetua variants (optionally filtered by grouping)
    for label, dist, grouping, kind in PERPETUA_VARIANTS:
        if args.grouping != "all" and grouping != args.grouping:
            continue
        if kind == "mixture":
            suffix = "_daycount_fullK_mins5_p0p01"
        else:
            suffix = "_daycount_emk1_noise0p01_mins5"
        exp_name = f"perpetua_{dist}_{grouping}_pucpr_split_50_20_30{suffix}"
        for D in PERPETUA_DAYS:
            d = latest_metric_dir(exp_name, D, args.seed)
            if d is None:
                print(f"WARN: missing {label} D={D}  (looked for {exp_name}_days{D}_seed{args.seed})")
                continue
            rows.append({"label": label, "path": str(d), "has_train_pct": "true"})

    # Vision-anchor curve
    vision_exp = "vision_anchor_midpoint_pucpr_split_50_20_30_daycount_balanced_train"
    for D in VISION_DAYS:
        d = latest_metric_dir(vision_exp, D, args.seed)
        if d is None:
            print(f"WARN: missing vision_anchor:class D={D}")
            continue
        rows.append({"label": "vision_anchor:class", "path": str(d), "has_train_pct": "true"})

    # LLM baseline (no day-count axis)
    llm_pattern = f"{EVAL_ROOT}/gemma_*/{SCENARIO_GLOB.split('/', 1)[1]}"
    llm_hits = sorted(p for p in glob.glob(llm_pattern) if (Path(p) / "metric_results.json").is_file())
    if llm_hits:
        rows.append({"label": "LLM (gemma)", "path": _to_container(llm_hits[-1]), "has_train_pct": "false"})
    else:
        print("WARN: no LLM percentile_horizons result found")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "path", "has_train_pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.out}")
    # Summary by label
    by_label: dict[str, int] = {}
    for r in rows:
        by_label[r["label"]] = by_label.get(r["label"], 0) + 1
    for lbl, n in sorted(by_label.items()):
        print(f"  {lbl}: {n} point(s)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
