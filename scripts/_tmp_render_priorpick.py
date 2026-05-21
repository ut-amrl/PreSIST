"""Side-by-side comparison: anchor-only vs standard mixture filter vs prior-pick filter.

Produces a 4-panel plot (one per mixture variant) showing bAcc-vs-D curves for
three configurations:
  1. anchor-only (percentile_horizons, no observations fed to filter)
  2. standard mixture_persistence_filter @ percentile_scaled_horizons (30-min recency)
  3. mixture_prior_pick_filter @ percentile_scaled_horizons (NEW, observations
     used only to pick argmax-k component; prediction is S_{k*}(dt))

Also a separate vision/LLM overlay on each panel so we can see how the
mixtures compare to the non-Perpetua baselines.
"""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/workspace/data/output")
OUT_DIR = DATA / "persistence_eval_plots/priorpick_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = DATA / "persistence_eval_plots/manifests/daycount_p0p01_mins5.csv"

# CSVs — pick noise level here.
NOISE = "matched"   # or "matched"
NOISE_SUFFIX = "" if NOISE == "default" else "_pm0p01_pf0p01"

PCT_PERP = DATA / "persistence_eval_outputs/sweep_metrics_pucpr_split_50_20_30_daycount_p0p01_mins5_all8.csv"
PSH_PERP_STD = DATA / f"persistence_eval_outputs/sweep_metrics_pucpr_split_50_20_30_daycount_p0p01_mins5_all8_percentile_scaled_horizons{NOISE_SUFFIX}.csv"
PSH_PERP_PRIORPICK = DATA / f"persistence_eval_outputs/sweep_metrics_pucpr_split_50_20_30_daycount_p0p01_mins5_all8_percentile_scaled_horizons{NOISE_SUFFIX}_filtermixture_prior_pick_filter.csv"

# vision/LLM at the same scenario for context
PSH_VL = DATA / f"persistence_eval_outputs/sweep_metrics_pucpr_split_50_20_30_daycount_p0p01_mins5_vision_llm_percentile_scaled_horizons{NOISE_SUFFIX}.csv"


def load_perp(path, metric="balanced_accuracy@0.5"):
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                key = (r["variant"], r["grouping"], int(r["days"]))
                out[key] = float(r[metric])
            except (ValueError, KeyError):
                continue
    return out


def load_vl(path, metric="balanced_accuracy@0.5"):
    vision, llm = {}, None
    if not path.exists():
        return vision, llm
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                kind = r["trainer_kind"]
                val = float(r[metric])
                if kind == "vision":
                    vision[int(r["days"])] = val
                elif kind == "llm":
                    llm = val
            except (ValueError, KeyError):
                continue
    return vision, llm


def load_vl_manifest(metric="balanced_accuracy@0.5"):
    vision, llm = {}, None
    if not MANIFEST.exists():
        return vision, llm
    with open(MANIFEST) as f:
        for line in f.readlines()[1:]:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 2:
                continue
            label, path = parts[0], parts[1]
            path = path.replace("/workspace/data/output", str(DATA))
            mr = Path(path) / "metric_results.json"
            if not mr.exists():
                continue
            with open(mr) as g:
                metrics = json.load(g)
            val = next((x["value"] for x in metrics if x["name"] == metric), None)
            if label == "vision_anchor:class":
                m = re.search(r"_days(\d+)_seed", path)
                if m:
                    vision[int(m.group(1))] = val
            elif label == "LLM (gemma)":
                llm = val
    return vision, llm


MIXTURE_VARIANTS = [
    ("mixture_exp:cls",        ("mixture_exp",       "class")),
    ("mixture_exp:loc",        ("mixture_exp",       "location")),
    ("mixture_lognormal:cls",  ("mixture_lognormal", "class")),
    ("mixture_lognormal:loc",  ("mixture_lognormal", "location")),
]
DAYS = [1, 2, 3, 4, 5, 7, 10, 14, 21, 28, 35]


def plot(metric, ylabel, anchor_perp, scaledhz_perp, priorpick_perp, vis_anchor, llm_anchor, vis_scaledhz, llm_scaledhz, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True, sharex=True)
    axes = axes.flatten()
    for ax, (label, (variant, grouping)) in zip(axes, MIXTURE_VARIANTS):
        for source, lookup, color, marker, ls in [
            ("anchor-only",         anchor_perp,    "#7f7f7f", "o", ":"),
            ("standard mix filter", scaledhz_perp,  "#1f77b4", "o", "-"),
            ("prior-pick filter",   priorpick_perp, "#d62728", "s", "-"),
        ]:
            xs, ys = [], []
            for d in DAYS:
                v = lookup.get((variant, grouping, d))
                if v is not None:
                    xs.append(d); ys.append(v)
            if xs:
                ax.plot(xs, ys, color=color, marker=marker, linestyle=ls,
                        markersize=6, linewidth=1.6, label=source)
        # baselines
        xs = sorted(vis_anchor)
        ys = [vis_anchor[d] for d in xs]
        if xs:
            ax.plot(xs, ys, color="black", marker="*", linestyle=":", markersize=9, linewidth=1.4, label="vision (anchor)")
        xs = sorted(vis_scaledhz)
        ys = [vis_scaledhz[d] for d in xs]
        if xs:
            ax.plot(xs, ys, color="black", marker="*", linestyle="-", markersize=9, linewidth=1.4, label="vision (scaled_horizons)")
        if llm_anchor is not None:
            ax.axhline(llm_anchor, color="purple", linestyle=":", linewidth=1.3, label=f"LLM (anchor) = {llm_anchor:.3f}")
        if llm_scaledhz is not None:
            ax.axhline(llm_scaledhz, color="purple", linestyle="-", linewidth=1.3, label=f"LLM (scaled) = {llm_scaledhz:.3f}")
        ax.set_title(label)
        ax.set_xlabel("Training day count D (log scale)")
        if ax in (axes[0], axes[2]):
            ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_xticks(DAYS)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, ncol=1, framealpha=0.85)
    fig.suptitle(f"Mixture filter ablation — {NOISE} noise (mixture variants only)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    anchor_perp     = load_perp(PCT_PERP, "balanced_accuracy@0.5")
    scaledhz_perp   = load_perp(PSH_PERP_STD, "balanced_accuracy@0.5")
    priorpick_perp  = load_perp(PSH_PERP_PRIORPICK, "balanced_accuracy@0.5")
    vis_anchor, llm_anchor   = load_vl_manifest("balanced_accuracy@0.5")
    vis_scaledhz, llm_scaledhz = load_vl(PSH_VL, "balanced_accuracy@0.5")

    print(f"=== Coverage @ noise={NOISE} ===")
    for name, d in [("anchor-only", anchor_perp), ("standard scaledhz", scaledhz_perp), ("prior-pick", priorpick_perp)]:
        n = sum(1 for k in d if k[0].startswith("mixture_"))
        print(f"  {name:22s}: {n}/44 mixture cells")
    print()

    plot("balanced_accuracy@0.5", "balanced accuracy @ 0.5", anchor_perp, scaledhz_perp, priorpick_perp,
         vis_anchor, llm_anchor, vis_scaledhz, llm_scaledhz,
         OUT_DIR / f"bacc_comparison_{NOISE}_noise.png")

    # MAE
    anchor_perp     = load_perp(PCT_PERP, "existence_mae")
    scaledhz_perp   = load_perp(PSH_PERP_STD, "existence_mae")
    priorpick_perp  = load_perp(PSH_PERP_PRIORPICK, "existence_mae")
    vis_anchor, llm_anchor   = load_vl_manifest("existence_mae")
    vis_scaledhz, llm_scaledhz = load_vl(PSH_VL, "existence_mae")
    plot("existence_mae", "existence MAE (lower = better)", anchor_perp, scaledhz_perp, priorpick_perp,
         vis_anchor, llm_anchor, vis_scaledhz, llm_scaledhz,
         OUT_DIR / f"mae_comparison_{NOISE}_noise.png")


if __name__ == "__main__":
    main()
