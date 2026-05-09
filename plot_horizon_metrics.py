from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger("plot_horizon_metrics")

ALL_METRICS = ["accuracy", "bacc", "precision", "recall", "f1", "mae", "mean_pred",
               "frac_pred_positive", "auroc", "auprc"]
BOUNDED_METRICS = {"accuracy", "bacc", "precision", "recall", "f1", "frac_pred_positive", "auroc", "auprc"}
DEFAULT_OUTPUT_ROOT = "/workspace/data/output/persistence_eval_plots"

# query_time is stored in seconds; multiplier converts seconds -> target unit.
X_UNIT_TO_SECONDS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
}


def load_manifest(path: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "label" not in reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError(f"manifest {path} must have header 'label,path'")
        for r in reader:
            label = r["label"].strip()
            run_path = Path(r["path"].strip()).expanduser()
            if not label or not str(run_path):
                continue
            rows.append((label, run_path))
    if not rows:
        raise ValueError(f"manifest {path} has no rows")
    return rows


def load_records(eval_run_dir: Path, label: str) -> list[dict]:
    pred_file = eval_run_dir / "prediction_records.json"
    if not pred_file.is_file():
        raise FileNotFoundError(f"[{label}] missing {pred_file}")
    return json.loads(pred_file.read_text())


def bucket_by_horizon(records: list[dict]) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[float, list[tuple[float, int]]] = defaultdict(list)
    for r in records:
        gt = r.get("ground_truth")
        if gt is None:
            continue
        qt_key = round(float(r["query_time"]), 6)
        grouped[qt_key].append((float(r["predicted_prob"]), int(bool(gt))))
    out: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for qt, pairs in grouped.items():
        probs = np.fromiter((p for p, _ in pairs), dtype=np.float64, count=len(pairs))
        gts = np.fromiter((g for _, g in pairs), dtype=np.int64, count=len(pairs))
        out[qt] = (probs, gts)
    return out


def compute_metric(name: str, probs: np.ndarray, gt: np.ndarray, threshold: float) -> float | None:
    if probs.size == 0:
        return None
    if name == "mean_pred":
        return float(probs.mean())
    if name == "frac_pred_positive":
        return float((probs >= threshold).mean())
    if name == "mae":
        return float(np.abs(probs - gt).mean())

    if name in {"auroc", "auprc"}:
        if np.unique(gt).size < 2:
            return None
        if name == "auroc":
            return float(roc_auc_score(gt, probs))
        return float(average_precision_score(gt, probs))

    preds = probs >= threshold
    labels = gt.astype(bool)
    tp = float(np.sum(preds & labels))
    tn = float(np.sum(~preds & ~labels))
    fp = float(np.sum(preds & ~labels))
    fn = float(np.sum(~preds & labels))
    n = float(preds.size)

    if name == "accuracy":
        return (tp + tn) / n
    if name == "bacc":
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return (tpr + tnr) / 2.0
    if name == "precision":
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    if name == "recall":
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if name == "f1":
        denom = 2 * tp + fp + fn
        return (2 * tp) / denom if denom > 0 else 0.0
    raise ValueError(f"unknown metric {name}")


def plot_metric(
    metric: str,
    per_label_buckets: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]],
    threshold: float,
    min_count: int,
    title: str | None,
    out_path: Path,
    x_units: str,
    show_base_rate: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    any_data = False
    seconds_per_unit = X_UNIT_TO_SECONDS[x_units]
    for label, buckets in per_label_buckets.items():
        xs: list[float] = []
        ys: list[float] = []
        for horizon in sorted(buckets):
            probs, gts = buckets[horizon]
            if probs.size < min_count:
                logger.warning("[%s] horizon %.6g s: %d < min_count %d, skipping", label, horizon, probs.size, min_count)
                continue
            value = compute_metric(metric, probs, gts, threshold)
            if value is None:
                logger.warning("[%s] horizon %.6g s: metric %s undefined (e.g. single class), skipping", label, horizon, metric)
                continue
            xs.append(horizon / seconds_per_unit)
            ys.append(value)
        if xs:
            ax.plot(xs, ys, marker="o", label=label)
            any_data = True

    ax.set_xlabel(f"Horizon ({x_units})")
    y_label = f"{metric}@{threshold}" if metric in {"accuracy", "bacc", "precision", "recall", "f1"} else metric
    ax.set_ylabel(y_label)
    if metric in BOUNDED_METRICS:
        ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    handles, labels_ = ax.get_legend_handles_labels()

    if show_base_rate:
        horizon_to_gts: dict[float, list[np.ndarray]] = defaultdict(list)
        for buckets in per_label_buckets.values():
            for horizon, (_, gts) in buckets.items():
                horizon_to_gts[horizon].append(gts)
        bx: list[float] = []
        by: list[float] = []
        bn: list[int] = []
        for horizon in sorted(horizon_to_gts):
            merged = np.concatenate(horizon_to_gts[horizon])
            if merged.size < min_count:
                continue
            bx.append(horizon / seconds_per_unit)
            by.append(100.0 * float(merged.mean()))
            bn.append(int(merged.size))
        if bx:
            ax2 = ax.twinx()
            fill = ax2.fill_between(bx, 0.0, by, color="gray", alpha=0.2, label="% present (GT)")
            ax2.set_ylabel("% present (ground truth)")
            ax2.set_ylim(0.0, 100.0)
            handles.append(fill)
            labels_.append("% present (GT)")
            logger.info("base rate per horizon (n): %s", dict(zip(bx, bn)))

    if handles:
        ax.legend(handles, labels_)
    elif not any_data:
        logger.warning("metric %s: no data plotted", metric)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def write_debug_csv(
    per_label_buckets: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]],
    threshold: float,
    out_path: Path,
    metrics: list[str],
) -> None:
    """Per-(label, horizon) diagnostics: confusion-matrix counts, base rate,
    mean prediction, positive prediction rate, and the configured metrics.
    Useful for diagnosing why a metric curve hugs the base rate (e.g., a
    constant 'always present' predictor pins precision to the base rate)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_cols = [
        "label", "horizon_s", "n", "n_positive", "base_rate",
        "mean_pred", "frac_pred_positive", "tp", "tn", "fp", "fn",
    ]
    fieldnames = base_cols + list(metrics)
    rows: list[dict] = []
    for label, buckets in per_label_buckets.items():
        for horizon in sorted(buckets):
            probs, gts = buckets[horizon]
            n = int(probs.size)
            if n == 0:
                continue
            preds = probs >= threshold
            labels = gts.astype(bool)
            tp = int(np.sum(preds & labels))
            tn = int(np.sum(~preds & ~labels))
            fp = int(np.sum(preds & ~labels))
            fn = int(np.sum(~preds & labels))
            row = {
                "label": label,
                "horizon_s": horizon,
                "n": n,
                "n_positive": int(labels.sum()),
                "base_rate": float(labels.mean()),
                "mean_pred": float(probs.mean()),
                "frac_pred_positive": float(preds.mean()),
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            }
            for m in metrics:
                v = compute_metric(m, probs, gts, threshold)
                row[m] = "" if v is None else v
            rows.append(row)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote debug csv %s (%d rows)", out_path, len(rows))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot per-horizon metrics across eval configs.")
    p.add_argument("--manifest", type=Path, required=True, help="CSV with 'label,path' header.")
    p.add_argument("--metric", action="append", choices=ALL_METRICS,
                   help="Metric(s) to plot (repeatable). Default: all.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--plot-subdir", type=str, default=None,
                   help="Per-manifest subdir. Default: manifest filename without extension.")
    p.add_argument("--prefix", type=str, default="")
    p.add_argument("--ext", type=str, default="pdf")
    p.add_argument("--x-units", choices=sorted(X_UNIT_TO_SECONDS.keys()), default="hours",
                   help="Units for the horizon axis (query_time is stored in seconds). Default: hours.")
    p.add_argument("--show-base-rate", action="store_true",
                   help="Add a right y-axis showing %% of ground truth that is present at each horizon.")
    p.add_argument("--debug-csv", type=Path, default=None,
                   help="If set, write a per-(label,horizon) diagnostics CSV here. "
                        "Useful for understanding why a curve hugs the base rate.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    metrics = args.metric or ALL_METRICS
    subdir = args.plot_subdir or args.manifest.stem
    out_dir = args.output_root / subdir
    fname_prefix = f"{args.prefix}_" if args.prefix else ""

    manifest_rows = load_manifest(args.manifest)
    per_label_buckets: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]] = {}
    for label, run_path in manifest_rows:
        records = load_records(run_path, label)
        per_label_buckets[label] = bucket_by_horizon(records)
        logger.info("[%s] loaded %d records, %d horizons", label, len(records), len(per_label_buckets[label]))

    for metric in metrics:
        out_path = out_dir / f"{fname_prefix}{metric}.{args.ext}"
        plot_metric(metric, per_label_buckets, args.threshold, args.min_count, args.title, out_path,
                    args.x_units, args.show_base_rate)

    if args.debug_csv is not None:
        write_debug_csv(per_label_buckets, args.threshold, args.debug_csv, metrics)

    return 0


if __name__ == "__main__":
    sys.exit(main())
