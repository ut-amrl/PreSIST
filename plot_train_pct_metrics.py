from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger("plot_train_pct_metrics")

# Mapping from metric_results.json record name -> CLI metric name.
METRIC_NAME_MAP = {
    "accuracy@0.5": "accuracy",
    "balanced_accuracy@0.5": "bacc",
    "f1@0.5": "f1",
    "precision@0.5": "precision",
    "recall@0.5": "recall",
    "existence_mae": "mae",
    "tp@0.5": "tp",
    "tn@0.5": "tn",
    "fp@0.5": "fp",
    "fn@0.5": "fn",
    "tpr@0.5": "tpr",
    "tnr@0.5": "tnr",
    "fpr@0.5": "fpr",
    "fnr@0.5": "fnr",
    "mean_image_total_latency_ms": "latency_ms",
}
ALL_METRICS = list(METRIC_NAME_MAP.values())
BOUNDED_METRICS = {"accuracy", "bacc", "f1", "precision", "recall", "tpr", "tnr", "fpr", "fnr"}
DEFAULT_OUTPUT_ROOT = "/workspace/data/output/persistence_eval_plots"

# Eval run dir layout:
#   <output_root>/<dataset>/<method>/<benchmark>/inference_runs/<TS_INF>/
#       evaluation_runs/<predictor>__<scenario>/<TS_EVAL>/
# So the <method> segment sits five levels above the <TS_EVAL> directory.
METHOD_SEGMENT_DEPTH = 5

FRAC_RE = re.compile(r"frac(\d+(?:p\d+)?)_seed\d+")


def parse_train_pct_from_path(run_path: Path) -> float:
    parts = run_path.resolve().parts
    if len(parts) <= METHOD_SEGMENT_DEPTH:
        raise ValueError(f"path too shallow to contain method segment: {run_path}")
    method_segment = parts[-(METHOD_SEGMENT_DEPTH + 1)]
    m = FRAC_RE.search(method_segment)
    if m is None:
        raise ValueError(
            f"could not parse train fraction from method segment '{method_segment}' "
            f"(path={run_path}); expected '...frac<X>p<Y>_seed<Z>...'"
        )
    return float(m.group(1).replace("p", "."))


ManifestRow = tuple[str, Path, bool]


def _parse_bool(s: str, *, source: str) -> bool:
    v = s.strip().lower()
    if v in {"true", "t", "yes", "y", "1"}:
        return True
    if v in {"false", "f", "no", "n", "0"}:
        return False
    raise ValueError(f"could not parse boolean '{s}' from {source}")


def load_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        required = {"label", "path", "has_train_pct"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"manifest {path} must have header 'label,path,has_train_pct' "
                f"(got {reader.fieldnames})"
            )
        for r in reader:
            label = r["label"].strip()
            run_path = Path(r["path"].strip()).expanduser()
            if not label or not str(run_path):
                continue
            has_tp = _parse_bool(r["has_train_pct"], source=f"{path} label={label}")
            rows.append((label, run_path, has_tp))
    if not rows:
        raise ValueError(f"manifest {path} has no rows")
    return rows


def load_metric_results(run_dir: Path, label: str) -> dict[str, float]:
    metric_file = run_dir / "metric_results.json"
    if not metric_file.is_file():
        raise FileNotFoundError(f"[{label}] missing {metric_file}")
    records = json.loads(metric_file.read_text())
    out: dict[str, float] = {}
    for rec in records:
        if rec.get("group"):
            continue
        cli_name = METRIC_NAME_MAP.get(rec["name"])
        if cli_name is None:
            continue
        out[cli_name] = float(rec["value"])
    return out


def gather_points(
    rows: list[ManifestRow],
) -> tuple[dict[str, list[tuple[float, dict[str, float]]]], dict[str, dict[str, float]]]:
    curves: dict[str, list[tuple[float, dict[str, float]]]] = defaultdict(list)
    baselines: dict[str, dict[str, float]] = {}
    for label, run_path, has_tp in rows:
        metrics = load_metric_results(run_path, label)
        if has_tp:
            train_pct = parse_train_pct_from_path(run_path)
            curves[label].append((train_pct, metrics))
        else:
            if label in baselines:
                logger.warning("baseline label '%s' specified more than once; overwriting", label)
            baselines[label] = metrics
    for label in curves:
        curves[label].sort(key=lambda kv: kv[0])
    return dict(curves), baselines


def _x_scale(x_units: str) -> float:
    if x_units == "percent":
        return 100.0
    if x_units == "fraction":
        return 1.0
    raise ValueError(f"unknown x_units {x_units}")


def plot_metric(
    metric: str,
    curves: dict[str, list[tuple[float, dict[str, float]]]],
    baselines: dict[str, dict[str, float]],
    title: str | None,
    out_path: Path,
    x_units: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    scale = _x_scale(x_units)
    any_data = False

    all_xs: list[float] = []
    for label, points in curves.items():
        xs: list[float] = []
        ys: list[float] = []
        for train_pct, metrics in points:
            value = metrics.get(metric)
            if value is None:
                logger.warning("[%s] train_pct=%.4g: metric %s missing, skipping", label, train_pct, metric)
                continue
            xs.append(train_pct * scale)
            ys.append(value)
        if xs:
            ax.plot(xs, ys, marker="o", label=label)
            all_xs.extend(xs)
            any_data = True

    if not all_xs:
        # Fallback x-range so baselines still plot meaningfully.
        all_xs = [0.0, 1.0 * scale]

    xmin, xmax = min(all_xs), max(all_xs)
    if xmin == xmax:
        xmax = xmin + 1.0

    for label, metrics in baselines.items():
        value = metrics.get(metric)
        if value is None:
            logger.warning("[%s] (no train) metric %s missing, skipping", label, metric)
            continue
        ax.hlines(value, xmin, xmax, linestyles="--", label=f"{label} (no train)")
        any_data = True

    ax.set_xlabel(f"Training data ({x_units})")
    ax.set_ylabel(metric)
    if metric in BOUNDED_METRICS:
        ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    handles, labels_ = ax.get_legend_handles_labels()
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
    curves: dict[str, list[tuple[float, dict[str, float]]]],
    baselines: dict[str, dict[str, float]],
    out_path: Path,
    metrics: list[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "train_pct", "kind"] + list(metrics)
    rows: list[dict] = []
    for label, points in curves.items():
        for train_pct, m in points:
            row = {"label": label, "train_pct": train_pct, "kind": "trained"}
            for name in metrics:
                row[name] = m.get(name, "")
            rows.append(row)
    for label, m in baselines.items():
        row = {"label": label, "train_pct": "", "kind": "baseline"}
        for name in metrics:
            row[name] = m.get(name, "")
        rows.append(row)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote debug csv %s (%d rows)", out_path, len(rows))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot metrics versus training-data percentage across eval runs.")
    p.add_argument("--manifest", type=Path, required=True,
                   help="CSV with header 'label,path,has_train_pct'.")
    p.add_argument("--metric", action="append", choices=ALL_METRICS,
                   help="Metric(s) to plot (repeatable). Default: all.")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--plot-subdir", type=str, default=None,
                   help="Per-manifest subdir. Default: manifest filename without extension.")
    p.add_argument("--prefix", type=str, default="")
    p.add_argument("--ext", type=str, default="pdf")
    p.add_argument("--x-units", choices=["fraction", "percent"], default="percent")
    p.add_argument("--debug-csv", type=Path, default=None,
                   help="If set, write per-(label, train_pct) diagnostics here.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    metrics = args.metric or ALL_METRICS
    subdir = args.plot_subdir or args.manifest.stem
    out_dir = args.output_root / subdir
    fname_prefix = f"{args.prefix}_" if args.prefix else ""

    manifest_rows = load_manifest(args.manifest)
    curves, baselines = gather_points(manifest_rows)
    for label, points in curves.items():
        logger.info("[%s] %d trained points: %s", label, len(points), [p[0] for p in points])
    for label in baselines:
        logger.info("[%s] baseline (no train)", label)

    for metric in metrics:
        out_path = out_dir / f"{fname_prefix}{metric}.{args.ext}"
        plot_metric(metric, curves, baselines, args.title, out_path, args.x_units)

    if args.debug_csv is not None:
        write_debug_csv(curves, baselines, args.debug_csv, metrics)

    return 0


if __name__ == "__main__":
    sys.exit(main())
