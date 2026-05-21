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
import numpy as np
import matplotlib as mpl
from matplotlib.colors import Normalize

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
# So the <method> segment sits six levels above the <TS_EVAL> directory.
METHOD_SEGMENT_DEPTH = 6

FRAC_RE = re.compile(r"frac(\d+(?:[p.]\d+)?)_seed\d+")
DAYS_RE = re.compile(r"days(\d+)_seed\d+")


def parse_train_pct_from_path(run_path: Path, *, kind: str = "fraction") -> float:
    """Parse the x-axis value from the method-segment of a run path.

    kind="fraction" -> matches 'frac<X>(p|.)<Y>_seed<Z>'; returns float fraction (0..1).
    kind="days"     -> matches 'days<N>_seed<Z>'; returns float day count.
    """
    parts = run_path.resolve().parts
    if len(parts) <= METHOD_SEGMENT_DEPTH:
        raise ValueError(f"path too shallow to contain method segment: {run_path}")
    method_segment = parts[-(METHOD_SEGMENT_DEPTH + 1)]
    if kind == "fraction":
        m = FRAC_RE.search(method_segment)
        if m is None:
            raise ValueError(
                f"could not parse train fraction from method segment '{method_segment}' "
                f"(path={run_path}); expected '...frac<X>(p|.)<Y>_seed<Z>...'"
            )
        return float(m.group(1).replace("p", "."))
    if kind == "days":
        m = DAYS_RE.search(method_segment)
        if m is None:
            raise ValueError(
                f"could not parse day count from method segment '{method_segment}' "
                f"(path={run_path}); expected '...days<N>_seed<Z>...'"
            )
        return float(m.group(1))
    raise ValueError(f"unknown parse kind {kind!r}")


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
    *,
    parse_kind: str = "fraction",
) -> tuple[dict[str, list[tuple[float, dict[str, float]]]], dict[str, dict[str, float]]]:
    curves: dict[str, list[tuple[float, dict[str, float]]]] = defaultdict(list)
    baselines: dict[str, dict[str, float]] = {}
    for label, run_path, has_tp in rows:
        metrics = load_metric_results(run_path, label)
        if has_tp:
            train_pct = parse_train_pct_from_path(run_path, kind=parse_kind)
            curves[label].append((train_pct, metrics))
        else:
            if label in baselines:
                logger.warning("baseline label '%s' specified more than once; overwriting", label)
            baselines[label] = metrics
    for label in curves:
        curves[label].sort(key=lambda kv: kv[0])
    return dict(curves), baselines


def _x_scale(x_units: str, n_total_tracks: int | None) -> float:
    if x_units == "percent":
        return 100.0
    if x_units == "fraction":
        return 1.0
    if x_units == "tracks":
        if n_total_tracks is None:
            raise ValueError("--x-units tracks requires --n-total-tracks")
        return float(n_total_tracks)
    if x_units == "days":
        return 1.0
    raise ValueError(f"unknown x_units {x_units}")


def plot_metric(
    metric: str,
    curves: dict[str, list[tuple[float, dict[str, float]]]],
    baselines: dict[str, dict[str, float]],
    title: str | None,
    out_path: Path,
    x_units: str,
    n_total_tracks: int | None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    scale = _x_scale(x_units, n_total_tracks)
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


def load_prediction_records(run_dir: Path, label: str) -> list[dict]:
    pred_file = run_dir / "prediction_records.json"
    if not pred_file.is_file():
        raise FileNotFoundError(f"[{label}] missing {pred_file}")
    return json.loads(pred_file.read_text())


LIFETIME_CONVENTIONS = ("first_absent", "last_positive", "midpoint")


def _resolve_crossing_time(
    qts: list[float], flags: list[bool], convention: str
) -> tuple[float, bool]:
    """Given query times and a boolean trajectory (True = still alive / above thresh),
    return (lifetime, event_observed). event_observed=False means right-censored.

    flags is True while the subject is "alive" and flips False at the first observed
    disappearance / threshold crossing. If flags never flips False, the subject is
    right-censored at the last query time.
    """
    cross_idx = next((i for i, alive in enumerate(flags) if not alive), None)
    if cross_idx is None:
        return qts[-1], False
    if convention == "first_absent":
        return qts[cross_idx], True
    if convention == "last_positive":
        return (qts[cross_idx - 1] if cross_idx > 0 else 0.0), True
    if convention == "midpoint":
        prev = qts[cross_idx - 1] if cross_idx > 0 else 0.0
        return 0.5 * (prev + qts[cross_idx]), True
    raise ValueError(f"unknown lifetime convention {convention!r}")


def extract_lifetimes(
    records: list[dict], *, threshold: float, convention: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-episode GT/predicted lifetimes (in seconds) with event flags.

    Returns (gt_lifetimes, pred_lifetimes, gt_events, pred_events) where event=True
    means a disappearance / threshold crossing was actually observed within the
    sampled horizons, and event=False means right-censored at the last horizon.
    """
    by_ep: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_ep[r["episode_id"]].append(r)

    gt_life: list[float] = []
    gt_evt: list[bool] = []
    pr_life: list[float] = []
    pr_evt: list[bool] = []
    for ep_records in by_ep.values():
        ep_sorted = sorted(ep_records, key=lambda r: r["query_time"])
        qts = [r["query_time"] for r in ep_sorted]
        if not qts:
            continue
        gt_alive = [bool(r["ground_truth"]) for r in ep_sorted]
        pr_alive = [float(r["predicted_prob"]) > threshold for r in ep_sorted]

        t, e = _resolve_crossing_time(qts, gt_alive, convention)
        gt_life.append(t); gt_evt.append(e)
        t, e = _resolve_crossing_time(qts, pr_alive, convention)
        pr_life.append(t); pr_evt.append(e)

    return (
        np.asarray(gt_life, dtype=float),
        np.asarray(pr_life, dtype=float),
        np.asarray(gt_evt, dtype=bool),
        np.asarray(pr_evt, dtype=bool),
    )


def kaplan_meier_cdf(
    times: np.ndarray, events: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, cdf) suitable for a step plot. cdf = 1 - S_KM(t).

    Censored observations (events == False) leave the at-risk set without producing
    a drop in S, so the CDF flattens after the last observed event and tops out
    below 1.0 when the largest observation is censored.
    """
    if times.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(times)
    times = times[order]
    events = events[order]

    unique_times = np.unique(times)
    n_at_risk = times.size
    s = 1.0
    xs: list[float] = [0.0]
    surv: list[float] = [1.0]
    for t in unique_times:
        mask = times == t
        d = int(np.sum(mask & events))   # events at t
        n = n_at_risk
        if d > 0 and n > 0:
            s *= (n - d) / n
        # Everyone observed at time t (event or censor) leaves the at-risk set after t.
        n_at_risk -= int(np.sum(mask))
        xs.append(float(t))
        surv.append(s)
    xs_arr = np.asarray(xs)
    cdf = 1.0 - np.asarray(surv)
    return xs_arr, cdf


def _time_scale(units: str) -> tuple[float, str]:
    if units == "seconds":
        return 1.0, "seconds"
    if units == "hours":
        return 1.0 / 3600.0, "hours"
    if units == "days":
        return 1.0 / 86400.0, "days"
    raise ValueError(f"unknown lifetime-units {units!r}")


def plot_lifetime_cdfs(
    curves_records: dict[str, list[tuple[float, list[dict]]]],
    baselines_records: dict[str, list[dict]],
    out_dir: Path,
    *,
    threshold: float,
    convention: str,
    lifetime_units: str,
    fname_prefix: str,
    ext: str,
    title: str | None,
    combined_sweeps: list[float] | None = None,
    combined_include: list[str] | None = None,
    combined_tag: str | None = None,
) -> None:
    scale, unit_label = _time_scale(lifetime_units)

    def _plot_one(label: str, items: list[tuple[float | None, list[dict]]], fname: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        gt_life_all: list[np.ndarray] = []
        gt_evt_all: list[np.ndarray] = []
        sweep_vals = [s for s, _ in items if s is not None]
        if sweep_vals:
            norm = Normalize(vmin=min(sweep_vals), vmax=max(sweep_vals))
            cmap = mpl.colormaps["viridis"]
        else:
            norm = None
            cmap = None

        for sweep_val, records in items:
            gt_life, pr_life, gt_evt, pr_evt = extract_lifetimes(
                records, threshold=threshold, convention=convention
            )
            gt_life_all.append(gt_life); gt_evt_all.append(gt_evt)
            xs, ys = kaplan_meier_cdf(pr_life * scale, pr_evt)
            if xs.size == 0:
                continue
            if sweep_val is None:
                color = "tab:red"
                cens_frac = 1.0 - pr_evt.mean() if pr_evt.size else 0.0
                lab = f"predicted (cens={cens_frac:.0%})"
            else:
                color = cmap(norm(sweep_val)) if cmap is not None else None
                cens_frac = 1.0 - pr_evt.mean() if pr_evt.size else 0.0
                lab = f"pred (x={sweep_val:g}, cens={cens_frac:.0%})"
            ax.step(xs, ys, where="post", color=color, alpha=0.85, label=lab)

        if gt_life_all:
            gt_pooled = np.concatenate(gt_life_all)
            gt_evt_pooled = np.concatenate(gt_evt_all)
            xs, ys = kaplan_meier_cdf(gt_pooled * scale, gt_evt_pooled)
            cens_frac = 1.0 - gt_evt_pooled.mean() if gt_evt_pooled.size else 0.0
            ax.step(xs, ys, where="post", color="black", linewidth=2.0,
                    label=f"ground truth (cens={cens_frac:.0%})")

        ax.set_xlabel(f"Lifetime ({unit_label})")
        ax.set_ylabel("CDF  P[lifetime ≤ t]  (1 − Kaplan–Meier)")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_title(title or f"Lifetime CDF — {label}")
        # Cap legend size if the sweep is large.
        handles, labels_ = ax.get_legend_handles_labels()
        if len(handles) > 12:
            ax.legend(handles[:1] + handles[-11:], labels_[:1] + labels_[-11:], fontsize=8)
        else:
            ax.legend(fontsize=8)
        out_path = out_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("wrote %s", out_path)

    include_set = set(combined_include) if combined_include else None

    def _plot_combined(sweep_val: float, fname: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        gt_life_all: list[np.ndarray] = []
        gt_evt_all: list[np.ndarray] = []
        plotted_any = False
        for label, points in curves_records.items():
            if include_set is not None and label not in include_set:
                continue
            match = next(((sv, recs) for sv, recs in points if sv == sweep_val), None)
            if match is None:
                logger.warning("combined sweep=%g: label %s has no run at this value, skipping",
                               sweep_val, label)
                continue
            _, records = match
            gt_life, pr_life, gt_evt, pr_evt = extract_lifetimes(
                records, threshold=threshold, convention=convention
            )
            gt_life_all.append(gt_life); gt_evt_all.append(gt_evt)
            xs, ys = kaplan_meier_cdf(pr_life * scale, pr_evt)
            if xs.size == 0:
                continue
            cens_frac = 1.0 - pr_evt.mean() if pr_evt.size else 0.0
            ax.step(xs, ys, where="post", alpha=0.85,
                    label=f"{label} (cens={cens_frac:.0%})")
            plotted_any = True
        # Baselines (e.g. LLM) overlaid as dashed lines.
        for label, records in baselines_records.items():
            if include_set is not None and label not in include_set:
                continue
            gt_life, pr_life, gt_evt, pr_evt = extract_lifetimes(
                records, threshold=threshold, convention=convention
            )
            gt_life_all.append(gt_life); gt_evt_all.append(gt_evt)
            xs, ys = kaplan_meier_cdf(pr_life * scale, pr_evt)
            if xs.size == 0:
                continue
            cens_frac = 1.0 - pr_evt.mean() if pr_evt.size else 0.0
            ax.step(xs, ys, where="post", linestyle="--", alpha=0.85,
                    label=f"{label} (cens={cens_frac:.0%})")
            plotted_any = True
        if gt_life_all:
            gt_pooled = np.concatenate(gt_life_all)
            gt_evt_pooled = np.concatenate(gt_evt_all)
            xs, ys = kaplan_meier_cdf(gt_pooled * scale, gt_evt_pooled)
            cens_frac = 1.0 - gt_evt_pooled.mean() if gt_evt_pooled.size else 0.0
            ax.step(xs, ys, where="post", color="black", linewidth=2.0,
                    label=f"ground truth (cens={cens_frac:.0%})")
        if not plotted_any:
            plt.close(fig)
            logger.warning("combined sweep=%g: nothing plotted", sweep_val)
            return
        ax.set_xlabel(f"Lifetime ({unit_label})")
        ax.set_ylabel("CDF  P[lifetime ≤ t]  (1 − Kaplan–Meier)")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_title(title or f"Lifetime CDF — sweep={sweep_val:g}")
        ax.legend(fontsize=8, loc="lower right")
        out_path = out_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("wrote %s", out_path)

    safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    for label, points in curves_records.items():
        items: list[tuple[float | None, list[dict]]] = [(sv, recs) for sv, recs in points]
        _plot_one(label, items, f"{fname_prefix}lifetime_cdf__{safe(label)}.{ext}")
    for label, records in baselines_records.items():
        _plot_one(label, [(None, records)], f"{fname_prefix}lifetime_cdf__{safe(label)}__baseline.{ext}")

    tag_suffix = f"__{combined_tag}" if combined_tag else ""
    for sweep_val in combined_sweeps or []:
        _plot_combined(
            sweep_val,
            f"{fname_prefix}lifetime_cdf__combined_x{sweep_val:g}{tag_suffix}.{ext}",
        )


def plot_lifetime_scatters(
    curves_records: dict[str, list[tuple[float, list[dict]]]],
    baselines_records: dict[str, list[dict]],
    out_dir: Path,
    *,
    threshold: float,
    convention: str,
    lifetime_units: str,
    fname_prefix: str,
    ext: str,
    sweep_vals: list[float],
    include: list[str] | None,
) -> None:
    scale, unit_label = _time_scale(lifetime_units)
    include_set = set(include) if include else None
    safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "_", s)

    def _aggregate(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if x.size == 0:
            return x, y, np.array([], dtype=int)
        pairs = np.stack([x, y], axis=1)
        uniq, counts = np.unique(pairs, axis=0, return_counts=True)
        return uniq[:, 0], uniq[:, 1], counts

    def _scatter(label: str, records: list[dict], fname: str, sweep_val: float | None) -> None:
        gt_life, pr_life, gt_evt, pr_evt = extract_lifetimes(
            records, threshold=threshold, convention=convention
        )
        if gt_life.size == 0:
            logger.warning("scatter %s: no episodes; skipping", label)
            return
        gt_x = gt_life * scale
        pr_y = pr_life * scale
        both_evt = gt_evt & pr_evt
        either_cens = ~both_evt
        n_total = int(gt_x.size)
        n_both = int(both_evt.sum())
        n_cens = int(either_cens.sum())

        # Aggregate identical (x, y) so overplotting is encoded by marker area.
        xb, yb, cb = _aggregate(gt_x[both_evt], pr_y[both_evt])
        xc, yc, cc = _aggregate(gt_x[either_cens], pr_y[either_cens])

        all_counts = np.concatenate([cb, cc]) if cb.size or cc.size else np.array([1])
        cmax = int(all_counts.max())
        # Area scales linearly with count; a single point is ~12pt², top count ~400pt².
        s_min, s_max = 12.0, 400.0
        def _sizes(c: np.ndarray) -> np.ndarray:
            if cmax <= 1:
                return np.full_like(c, s_min, dtype=float)
            return s_min + (s_max - s_min) * (c - 1) / (cmax - 1)

        fig, ax = plt.subplots(figsize=(6, 6))
        if xb.size:
            ax.scatter(xb, yb, s=_sizes(cb), alpha=0.55, color="tab:blue",
                       edgecolors="none",
                       label=f"both observed (n={n_both})")
        if xc.size:
            ax.scatter(xc, yc, s=_sizes(cc), alpha=0.55,
                       facecolors="none", edgecolors="tab:red", linewidths=1.2,
                       label=f"≥1 censored (n={n_cens})")
        lim_max = max(float(gt_x.max(initial=0.0)), float(pr_y.max(initial=0.0)))
        if lim_max <= 0:
            lim_max = 1.0
        ax.plot([0, lim_max], [0, lim_max], color="black", linestyle="--", linewidth=1.0,
                label="y = x")
        ax.set_xlim(0, lim_max * 1.02)
        ax.set_ylim(0, lim_max * 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel(f"GT lifetime ({unit_label}, {convention})")
        ax.set_ylabel(f"Predicted lifetime ({unit_label}, {convention})")
        title_suffix = f" — x={sweep_val:g}" if sweep_val is not None else " (baseline)"
        ax.set_title(f"{label}{title_suffix}")
        ax.grid(True, alpha=0.3)
        cat_legend = ax.legend(fontsize=8, loc="upper left")
        ax.add_artist(cat_legend)
        # Size-legend: a few reference counts spanning [1, cmax].
        if cmax > 1:
            ticks = sorted({1, max(1, cmax // 10), max(1, cmax // 3), cmax})
            handles = [
                plt.scatter([], [], s=_sizes(np.array([t]))[0], color="gray",
                            alpha=0.55, edgecolors="none", label=f"n = {t}")
                for t in ticks
            ]
            ax.legend(handles=handles, fontsize=8, loc="lower right",
                      title="points per cell", title_fontsize=8,
                      labelspacing=1.2, borderpad=1.0)
        out_path = out_dir / fname
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("wrote %s", out_path)

    for label, points in curves_records.items():
        if include_set is not None and label not in include_set:
            continue
        for sweep_val in sweep_vals:
            match = next(((sv, recs) for sv, recs in points if sv == sweep_val), None)
            if match is None:
                logger.warning("scatter %s: no run at sweep=%g, skipping", label, sweep_val)
                continue
            _, recs = match
            _scatter(label,
                     recs,
                     f"{fname_prefix}lifetime_scatter__{safe(label)}__x{sweep_val:g}.{ext}",
                     sweep_val)

    for label, recs in baselines_records.items():
        if include_set is not None and label not in include_set:
            continue
        _scatter(label,
                 recs,
                 f"{fname_prefix}lifetime_scatter__{safe(label)}__baseline.{ext}",
                 None)


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
    p.add_argument("--x-units", choices=["fraction", "percent", "tracks", "days"],
                   action="append", default=None,
                   help="X-axis unit (repeatable). Default: percent. Pass "
                        "multiple times to emit one plot per unit "
                        "(filenames get a '__<unit>' suffix when >1 unit). "
                        "Use 'days' for day-count sweeps; pair with "
                        "--x-kind=days so paths are parsed as 'days<N>_seed<S>'.")
    p.add_argument("--x-kind", choices=["fraction", "days"], default="fraction",
                   help="Parser used to extract the x-axis value from the "
                        "method-segment of each run path. Default: fraction "
                        "(expects 'frac<X>p<Y>_seed<S>'). Use 'days' when the "
                        "method segment carries 'days<N>_seed<S>'.")
    p.add_argument("--n-total-tracks", type=int, default=None,
                   help="Total tracks in the train manifest, used to scale the "
                        "x-axis when --x-units tracks is given. Compute once "
                        "from the train manifest's track count.")
    p.add_argument("--lifetime-cdf", action="store_true",
                   help="Also emit per-label lifetime CDF plots comparing "
                        "predicted-lifetime distribution to ground-truth lifetime.")
    p.add_argument("--lifetime-threshold", type=float, default=0.5,
                   help="Predicted-probability threshold defining predicted lifetime "
                        "(first query_time at which predicted_prob <= threshold). "
                        "Default 0.5.")
    p.add_argument("--lifetime-units", choices=["seconds", "hours", "days"], default="hours",
                   help="X-axis units for lifetime CDF plots. Default: hours.")
    p.add_argument("--lifetime-combined-sweep", type=float, action="append", default=None,
                   help="Repeatable. For each value, emit a combined plot overlaying every "
                        "trained label's predicted-lifetime CDF at that sweep value, plus "
                        "all baselines (dashed) and the pooled GT curve. Example: "
                        "'--lifetime-combined-sweep 1 --lifetime-combined-sweep 35' produces "
                        "one figure for 1-day runs + LLM and another for 35-day runs + LLM.")
    p.add_argument("--lifetime-combined-include", action="append", default=None,
                   help="Repeatable. Restrict the combined plot(s) to these manifest "
                        "labels (trained and/or baseline). If omitted, includes all labels.")
    p.add_argument("--lifetime-scatter", action="store_true",
                   help="Emit per-(label, sweep) scatter plots of GT lifetime vs predicted "
                        "lifetime, one figure per trained run at each --lifetime-combined-sweep "
                        "value, plus one per baseline. Honors --lifetime-combined-include.")
    p.add_argument("--lifetime-combined-tag", type=str, default=None,
                   help="Filename suffix for combined plots, e.g. 'class_only'. Useful "
                        "when running multiple invocations with different include lists.")
    p.add_argument("--gt-lifetime", choices=list(LIFETIME_CONVENTIONS), default="midpoint",
                   help="How to convert the discrete observation grid into a per-episode "
                        "lifetime. 'midpoint' (default) = halfway between last positive and "
                        "first absent (interval-censoring midpoint). 'first_absent' = first "
                        "query_time where the trajectory flipped negative (upper bound). "
                        "'last_positive' = last query_time still positive (lower bound). "
                        "Same convention is applied to the predicted trajectory.")
    p.add_argument("--debug-csv", type=Path, default=None,
                   help="If set, write per-(label, train_pct) diagnostics here.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    metrics = args.metric or ALL_METRICS
    units_list = args.x_units or ["percent"]
    subdir = args.plot_subdir or args.manifest.stem
    out_dir = args.output_root / subdir
    fname_prefix = f"{args.prefix}_" if args.prefix else ""

    manifest_rows = load_manifest(args.manifest)
    curves, baselines = gather_points(manifest_rows, parse_kind=args.x_kind)
    for label, points in curves.items():
        logger.info("[%s] %d trained points: %s", label, len(points), [p[0] for p in points])
    for label in baselines:
        logger.info("[%s] baseline (no train)", label)

    for metric in metrics:
        for x_units in units_list:
            unit_suffix = f"__{x_units}" if len(units_list) > 1 else ""
            out_path = out_dir / f"{fname_prefix}{metric}{unit_suffix}.{args.ext}"
            plot_metric(metric, curves, baselines, args.title, out_path,
                        x_units, args.n_total_tracks)

    if args.debug_csv is not None:
        write_debug_csv(curves, baselines, args.debug_csv, metrics)

    if args.lifetime_cdf:
        curves_records: dict[str, list[tuple[float, list[dict]]]] = defaultdict(list)
        baselines_records: dict[str, list[dict]] = {}
        for label, run_path, has_tp in manifest_rows:
            try:
                records = load_prediction_records(run_path, label)
            except FileNotFoundError as e:
                logger.warning("%s — skipping for lifetime CDF", e)
                continue
            if has_tp:
                sweep_val = parse_train_pct_from_path(run_path, kind=args.x_kind)
                curves_records[label].append((sweep_val, records))
            else:
                if label in baselines_records:
                    logger.warning("baseline label '%s' specified more than once; overwriting", label)
                baselines_records[label] = records
        for label in curves_records:
            curves_records[label].sort(key=lambda kv: kv[0])
        plot_lifetime_cdfs(
            dict(curves_records),
            baselines_records,
            out_dir,
            threshold=args.lifetime_threshold,
            convention=args.gt_lifetime,
            lifetime_units=args.lifetime_units,
            fname_prefix=fname_prefix,
            ext=args.ext,
            title=args.title,
            combined_sweeps=args.lifetime_combined_sweep,
            combined_include=args.lifetime_combined_include,
            combined_tag=args.lifetime_combined_tag,
        )
        if args.lifetime_scatter:
            plot_lifetime_scatters(
                dict(curves_records),
                baselines_records,
                out_dir,
                threshold=args.lifetime_threshold,
                convention=args.gt_lifetime,
                lifetime_units=args.lifetime_units,
                fname_prefix=fname_prefix,
                ext=args.ext,
                sweep_vals=list(args.lifetime_combined_sweep or []),
                include=args.lifetime_combined_include,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
