from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from plot_train_pct_metrics import (
    DEFAULT_OUTPUT_ROOT,
    load_manifest,
    parse_train_pct_from_path,
)

logger = logging.getLogger("plot_disappearance_distributions")

PROB_THRESHOLD = 0.5


def per_episode_disappearance_times(
    records: list[dict],
) -> tuple[list[float], list[float]]:
    """For each episode (one anchor query series), compute:
      true_t  — smallest query_time where ground_truth flips to False
      pred_t  — smallest query_time where predicted_prob crosses below
                PROB_THRESHOLD

    Right-censored episodes (truth never flips / pred never crosses) are capped
    at the largest query_time seen for that episode.
    """
    by_ep: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_ep[r["episode_id"]].append(r)

    true_ts: list[float] = []
    pred_ts: list[float] = []
    for recs in by_ep.values():
        recs.sort(key=lambda r: r["query_time"])
        qts = [r["query_time"] for r in recs]
        cap = qts[-1]

        true_t = cap
        for r in recs:
            if not r["ground_truth"]:
                true_t = r["query_time"]
                break

        pred_t = cap
        for r in recs:
            if r["predicted_prob"] < PROB_THRESHOLD:
                pred_t = r["query_time"]
                break

        true_ts.append(true_t)
        pred_ts.append(pred_t)
    return true_ts, pred_ts


def ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    if not values:
        return [], []
    xs = sorted(values)
    n = len(xs)
    # Step ECDF: y jumps from i/n to (i+1)/n at xs[i].
    step_x: list[float] = [xs[0]]
    step_y: list[float] = [0.0]
    for i, x in enumerate(xs):
        step_x.append(x)
        step_y.append((i + 1) / n)
    return step_x, step_y


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(
        description="Plot ECDFs of true vs predicted disappearance times across methods."
    )
    p.add_argument("--manifest", type=Path, required=True,
                   help="CSV with header 'label,path,has_train_pct' (reused from "
                        "plot_train_pct_metrics).")
    p.add_argument("--fractions", type=str, default=None,
                   help="Space-separated train fractions to include for trained "
                        "rows (e.g., '0.1 0.5 1.0'). Untrained rows "
                        "(has_train_pct=false) are always included.")
    p.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--plot-subdir", type=str, default=None,
                   help="Per-manifest subdir. Default: manifest stem.")
    p.add_argument("--prefix", type=str, default="disappearance")
    p.add_argument("--ext", type=str, default="pdf")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--xlog", action="store_true",
                   help="Use a log scale on the time axis (helps when "
                        "disappearance times span several orders of magnitude).")
    p.add_argument("--time-units", choices=["seconds", "hours", "days"],
                   default="hours",
                   help="X-axis time units (default: hours).")
    args = p.parse_args(argv)

    fraction_filter: set[float] | None = None
    if args.fractions is not None:
        fraction_filter = {float(f) for f in args.fractions.split()}

    subdir = args.plot_subdir or args.manifest.stem
    out_dir = args.output_root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest(args.manifest)

    # curves keyed by full curve label (one entry per row we actually load).
    # Value is (per-episode predicted times, is_baseline) where is_baseline is
    # True for untrained rows (rendered as a dashed line, like the horizontal
    # baselines in plot_train_pct_metrics.py).
    pred_curves: dict[str, tuple[list[float], bool]] = {}
    # truth values from each row — should all match; we keep one canonical set
    # and warn if any row disagrees.
    canonical_truth: list[float] | None = None

    for label, run_path, has_tp in manifest_rows:
        if has_tp:
            try:
                frac = parse_train_pct_from_path(run_path)
            except ValueError as e:
                logger.warning("[%s] %s — skipping", label, e)
                continue
            if fraction_filter is not None and frac not in fraction_filter:
                continue
            curve_label = f"{label} (frac={frac:g})"
        else:
            curve_label = f"{label} (no train)"

        pred_file = run_path / "prediction_records.json"
        if not pred_file.is_file():
            logger.warning("[%s] missing %s — skipping", curve_label, pred_file)
            continue

        records = json.loads(pred_file.read_text())
        true_ts, pred_ts = per_episode_disappearance_times(records)
        if not pred_ts:
            logger.warning("[%s] no episodes — skipping", curve_label)
            continue

        if canonical_truth is None:
            canonical_truth = sorted(true_ts)
        else:
            if sorted(true_ts) != canonical_truth:
                logger.warning(
                    "[%s] true-disappearance distribution differs from earlier "
                    "rows — different test subset?", curve_label
                )

        pred_curves[curve_label] = (pred_ts, not has_tp)
        logger.info("[%s] %d episodes", curve_label, len(pred_ts))

    if canonical_truth is None or not pred_curves:
        logger.error("no usable rows after filtering — nothing to plot")
        return 1

    time_scale = {"seconds": 1.0, "hours": 1.0 / 3600.0, "days": 1.0 / 86400.0}[args.time_units]

    fig, ax = plt.subplots(figsize=(11, 6))
    tx, ty = ecdf(canonical_truth)
    ax.step([x * time_scale for x in tx], ty, where="post",
            linewidth=2.5, color="black", label="ground truth")

    for curve_label, (ts, is_baseline) in pred_curves.items():
        px, py = ecdf(ts)
        ax.step([x * time_scale for x in px], py, where="post",
                alpha=0.85, linestyle="--" if is_baseline else "-",
                label=curve_label)

    ax.set_xlabel(f"Disappearance time ({args.time_units} past anchor)")
    ax.set_ylabel("ECDF")
    ax.set_ylim(0.0, 1.02)
    if args.xlog:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    if args.title:
        ax.set_title(args.title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, borderaxespad=0.0)

    suffix = "_xlog" if args.xlog else ""
    out_path = out_dir / f"{args.prefix}_ecdf_{args.time_units}{suffix}.{args.ext}"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
