"""Plot qualitative PKLot examples for one or more inference runs.

For each sampled track-anchor observation we render:
  - the RGB image with the anchor mask overlaid
  - a text panel listing the predicted parameters, derived p25/p50/p75
    persistence horizons (when defined for that model family), and the
    ground-truth remaining-persistence time

Paths in the input manifest are assumed to be container paths
(``/workspace/data/...``); run this inside the docker container.

Example:
    python3 plot_pklot_examples.py \\
        --manifest plot_manifests/three_way_train_pct/manifest.csv \\
        --n 8 --seed 0

    python3 plot_pklot_examples.py \\
        --run-dir /workspace/data/output/persistence_eval_outputs/.../inference_runs/20260511_093501 \\
        --methods perpetua_single_exp_location vision_anchor_midpoint
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from scipy.special import ndtri

CONFIGS_BENCHMARK_DATASET_DIR = Path(__file__).resolve().parent / "configs" / "train_confs" / "benchmark_dataset"
US_TO_S = 1_000_000.0
PERCENTILES = (0.25, 0.50, 0.75)


# --------------------------------------------------------------------------- #
# Dataset lookups
# --------------------------------------------------------------------------- #


@dataclass
class DatasetConfig:
    name: str
    manifest_path: Path  # container-side
    host_root: str | None
    container_root: str | None


def _load_dataset_configs() -> dict[str, DatasetConfig]:
    out: dict[str, DatasetConfig] = {}
    for yml in CONFIGS_BENCHMARK_DATASET_DIR.glob("*.yaml"):
        text = yml.read_text()
        # The hydra `# @package` directive isn't valid yaml on its own line.
        text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("# @package"))
        try:
            cfg = yaml.safe_load(text)
        except yaml.YAMLError:
            continue
        if not isinstance(cfg, dict) or "name" not in cfg or "manifest_path" not in cfg:
            continue
        out[cfg["name"]] = DatasetConfig(
            name=cfg["name"],
            manifest_path=Path(cfg["manifest_path"]),
            host_root=cfg.get("host_root"),
            container_root=cfg.get("container_root"),
        )
    return out


def _remap_to_container(path_str: str, ds: DatasetConfig) -> str:
    if ds.host_root and ds.container_root and path_str.startswith(ds.host_root):
        return ds.container_root + path_str[len(ds.host_root):]
    return path_str


# --------------------------------------------------------------------------- #
# Track manifest loader (anchor obs + boundaries only — no torch needed)
# --------------------------------------------------------------------------- #


@dataclass
class AnchorRecord:
    track_id: str
    anchor_time_s: float
    image_path: Path
    mask_path: Path
    first_absent_after_s: float | None
    last_observed_s: float


def _load_track_records(ds: DatasetConfig) -> dict[tuple[str, float], AnchorRecord]:
    manifest = json.loads(ds.manifest_path.read_text())
    location_root = Path(_remap_to_container(manifest["location_root"], ds))
    manifest_dir = ds.manifest_path.parent
    records: dict[tuple[str, float], AnchorRecord] = {}
    for rel in manifest["track_files"]:
        track = json.loads((manifest_dir / rel).read_text())
        b = track["boundaries"]
        first_absent_after_s = (
            b["first_absent_after_ts"] / US_TO_S
            if b.get("first_absent_after_ts") is not None else None
        )
        last_observed_s = b["last_observed_ts"] / US_TO_S
        for obs in track["observations"]:
            if obs["state"] != "Positive" or obs.get("mask_path") is None:
                continue
            anchor_time_s = obs["timestamp"] / US_TO_S
            records[(track["track_id"], anchor_time_s)] = AnchorRecord(
                track_id=track["track_id"],
                anchor_time_s=anchor_time_s,
                image_path=location_root / obs["image_path"],
                mask_path=location_root / _remap_to_container(obs["mask_path"], ds),
                first_absent_after_s=first_absent_after_s,
                last_observed_s=last_observed_s,
            )
    return records


# --------------------------------------------------------------------------- #
# Percentile computation per model family
# --------------------------------------------------------------------------- #


def _exp_percentile(lam: float, q: float) -> float:
    return -math.log(1.0 - q) / lam


def _lognormal_percentile(logmu: float, std: float, q: float) -> float:
    return math.exp(logmu + std * ndtri(q))


def _mixture_exp_survival(t: np.ndarray, lambdas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (weights * np.exp(-np.outer(t, lambdas))).sum(axis=1)


def _mixture_lognormal_survival(t: np.ndarray, logmus: np.ndarray, stds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    from scipy.stats import norm
    t_safe = np.clip(t, 1e-12, None)
    z = (np.log(t_safe)[:, None] - logmus) / stds
    S_k = norm.sf(z)  # 1 - Phi(z) = Phi(-z)
    return (weights * S_k).sum(axis=1)


def _bisect_percentile(S, q: float, hi: float = 1e10) -> float:
    target = 1.0 - q  # find t with S(t) == target
    lo = 0.0
    s_lo = float(S(np.array([lo]))[0])
    s_hi = float(S(np.array([hi]))[0])
    if s_hi > target:
        return float("inf")
    if s_lo < target:
        return 0.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if S(np.array([mid]))[0] > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-3:
            break
    return 0.5 * (lo + hi)


def compute_percentiles(params: dict, metadata: dict) -> dict[str, float | None]:
    """Return {"p25": ..., "p50": ..., "p75": ...} in seconds.

    Returns None for percentiles that aren't defined for the given model.
    """
    model_type = (metadata or {}).get("model_type")
    # Perpetua single-exp may store as {"lambda_": ..., "mean": ...}
    if model_type == "single_exponential" or "lambda_" in params and "lambdas" not in params:
        lam = float(params["lambda_"])
        return {f"p{int(q*100)}": _exp_percentile(lam, q) for q in PERCENTILES}
    if model_type == "single_lognormal" or ("logmu" in params and "std" in params and "logmus" not in params):
        logmu, std = float(params["logmu"]), float(params["std"])
        return {f"p{int(q*100)}": _lognormal_percentile(logmu, std, q) for q in PERCENTILES}
    if model_type == "mixture_exponential" or "lambdas" in params:
        lambdas = np.asarray(params["lambdas"], dtype=float)
        weights = np.asarray(params["weights"], dtype=float)
        weights = weights / weights.sum()
        S = lambda t: _mixture_exp_survival(t, lambdas, weights)
        return {f"p{int(q*100)}": _bisect_percentile(S, q) for q in PERCENTILES}
    if model_type == "mixture_lognormal" or "logmus" in params:
        logmus = np.asarray(params["logmus"], dtype=float)
        stds = np.asarray(params["stds"], dtype=float)
        weights = np.asarray(params["weights"], dtype=float)
        weights = weights / weights.sum()
        S = lambda t: _mixture_lognormal_survival(t, logmus, stds, weights)
        return {f"p{int(q*100)}": _bisect_percentile(S, q) for q in PERCENTILES}
    # Vision anchor midpoint emits a single t50 in seconds.
    if "t50" in params:
        return {"p25": None, "p50": float(params["t50"]), "p75": None}
    # Gemma / LLM pipeline emits hours.
    if "p25_persistence_hours" in params or "p75_persistence_hours" in params:
        p25 = params.get("p25_persistence_hours")
        p75 = params.get("p75_persistence_hours")
        return {
            "p25": float(p25) * 3600.0 if p25 is not None else None,
            "p50": None,
            "p75": float(p75) * 3600.0 if p75 is not None else None,
        }
    return {"p25": None, "p50": None, "p75": None}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def _format_seconds(secs: float | None) -> str:
    if secs is None:
        return "—"
    if not math.isfinite(secs):
        return "∞"
    if secs < 60:
        return f"{secs:.2f} s"
    if secs < 3600:
        return f"{secs/60:.2f} min"
    if secs < 86400:
        return f"{secs/3600:.2f} h"
    return f"{secs/86400:.2f} d"


def _format_params(params: dict) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        if isinstance(v, list):
            return "[" + ", ".join(f"{float(x):.4g}" for x in v) + "]"
        return str(v)
    return "\n".join(f"  {k} = {fmt(v)}" for k, v in params.items())


def plot_example(record: AnchorRecord, artifact: dict, label: str, out_path: Path,
                 bbox_margin: int = 30) -> None:
    img = np.array(Image.open(record.image_path).convert("RGB"))
    mask = np.array(Image.open(record.mask_path).convert("L")) > 0
    overlay = img.copy()
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([255, 80, 80])).astype(np.uint8)

    # Tight bbox around the mask, expanded by `bbox_margin` px and clipped to image.
    ys, xs = np.where(mask)
    if ys.size > 0:
        h, w = mask.shape
        y0 = max(int(ys.min()) - bbox_margin, 0)
        y1 = min(int(ys.max()) + bbox_margin, h - 1)
        x0 = max(int(xs.min()) - bbox_margin, 0)
        x1 = min(int(xs.max()) + bbox_margin, w - 1)
        bbox = (x0, y0, x1, y1)
    else:
        bbox = None

    est = artifact.get("estimate", {})
    params = est.get("params", {})
    metadata = est.get("metadata", {})
    pct = compute_percentiles(params, metadata)

    gt_remaining = (
        record.first_absent_after_s - record.anchor_time_s
        if record.first_absent_after_s is not None else None
    )
    gt_min_remaining = record.last_observed_s - record.anchor_time_s

    text_lines = [
        f"Method: {label}",
        f"Track: {record.track_id}",
        f"Anchor time: {record.anchor_time_s:.0f}",
        f"Model type: {metadata.get('model_type', '—')}",
        "",
        "Params:",
        _format_params(params),
        "",
        "Predicted horizons:",
        f"  p25 = {_format_seconds(pct.get('p25'))}",
        f"  p50 = {_format_seconds(pct.get('p50'))}",
        f"  p75 = {_format_seconds(pct.get('p75'))}",
        "",
        "Ground truth:",
        f"  remaining = {_format_seconds(gt_remaining)}"
        + ("  (censored ≥)" if gt_remaining is None else ""),
        f"  ≥ observed = {_format_seconds(gt_min_remaining)}",
        f"  confidence = {est.get('confidence')}",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.4, 1.0]})
    axes[0].imshow(overlay)
    axes[0].set_title("anchor (mask overlay + bbox)")
    axes[0].axis("off")
    if bbox is not None:
        from matplotlib.patches import Rectangle
        x0, y0, x1, y1 = bbox
        axes[0].add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor="yellow", linewidth=2.0,
        ))
    axes[1].axis("off")
    axes[1].text(
        0.0, 1.0, "\n".join(text_lines),
        family="monospace", fontsize=10, va="top", ha="left",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


_INFERENCE_RUN_RE = re.compile(r".*?/inference_runs/[^/]+")


def _inference_run_dir(path: Path) -> Path:
    """Given an inference_run or evaluation_run path, return the inference_run dir."""
    s = str(path)
    m = _INFERENCE_RUN_RE.match(s)
    if not m:
        raise ValueError(f"Not under an inference_runs/<ts> directory: {path}")
    return Path(m.group(0))


def _dataset_name_from_path(p: Path) -> str:
    parts = p.parts
    if "persistence_eval_outputs" not in parts:
        raise ValueError(f"Path doesn't contain persistence_eval_outputs: {p}")
    i = parts.index("persistence_eval_outputs")
    return parts[i + 1]


def _variant_from_path(p: Path) -> str:
    """Pull e.g. 'frac0p1_seed0' from the experiment-name dir."""
    parts = p.parts
    if "persistence_eval_outputs" not in parts:
        return "unknown"
    i = parts.index("persistence_eval_outputs")
    exp_dir = parts[i + 2]  # <method>_<...>_frac<F>_seed<S>
    m = re.search(r"(frac[\w.]+_seed\d+)", exp_dir)
    return m.group(1) if m else exp_dir


def _iter_manifest_rows(manifest_path: Path):
    with manifest_path.open() as f:
        for row in csv.DictReader(f):
            yield row["label"], Path(row["path"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Plot-manifest CSV (label,path,...). Mutually exclusive with --run-dir.")
    ap.add_argument("--run-dir", type=Path, nargs="*", default=None,
                    help="Specific inference_runs/<ts> (or evaluation_runs/.../<ts>) dirs.")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Filter manifest rows to these labels.")
    ap.add_argument("--label", default=None,
                    help="Label to attach to --run-dir entries (defaults to method dir).")
    ap.add_argument("--n", type=int, default=5,
                    help="Number of examples to sample per run.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-root", type=Path, default=Path("outputs/plot_examples"))
    ap.add_argument("--subdir", default=None,
                    help="Override the manifest-set subdir name "
                         "(default: stem of --manifest, or 'ad_hoc' for --run-dir).")
    args = ap.parse_args()

    if (args.manifest is None) == (args.run_dir is None):
        ap.error("Pass exactly one of --manifest or --run-dir.")

    dataset_configs = _load_dataset_configs()
    track_cache: dict[str, dict[tuple[str, float], AnchorRecord]] = {}

    if args.manifest is not None:
        rows = list(_iter_manifest_rows(args.manifest))
        if args.methods:
            rows = [(l, p) for l, p in rows if l in set(args.methods)]
        subdir_name = args.subdir or args.manifest.parent.name
    else:
        rows = [(args.label or _variant_from_path(p).split("/")[0], p) for p in args.run_dir]
        subdir_name = args.subdir or "ad_hoc"

    out_root = args.output_root / subdir_name
    print(f"writing plots under {out_root}")

    for label, eval_or_run_path in rows:
        try:
            run_dir = _inference_run_dir(eval_or_run_path)
        except ValueError as e:
            print(f"  skip {label}: {e}")
            continue
        artifacts_path = run_dir / "inference_artifacts.json"
        if not artifacts_path.exists():
            print(f"  skip {label}: missing {artifacts_path}")
            continue
        ds_name = _dataset_name_from_path(run_dir)
        if ds_name not in dataset_configs:
            print(f"  skip {label}: unknown dataset '{ds_name}'")
            continue
        ds_cfg = dataset_configs[ds_name]
        if ds_name not in track_cache:
            print(f"  loading dataset manifest: {ds_name}")
            track_cache[ds_name] = _load_track_records(ds_cfg)
        records = track_cache[ds_name]

        artifacts = json.loads(artifacts_path.read_text())
        # Restrict to anchors with a known image + mask.
        usable = [a for a in artifacts if (a["track_id"], a["anchor_time"]) in records]
        if not usable:
            print(f"  skip {label}: no artifact anchors matched dataset records")
            continue
        rng = random.Random(args.seed)
        sample = rng.sample(usable, min(args.n, len(usable)))

        variant = _variant_from_path(run_dir)
        run_subdir = out_root / f"{label}__{variant}"
        for art in sample:
            rec = records[(art["track_id"], art["anchor_time"])]
            out_path = run_subdir / f"{rec.track_id}__{int(rec.anchor_time_s)}.png"
            plot_example(rec, art, label, out_path)
        print(f"  {label}: wrote {len(sample)} → {run_subdir}")


if __name__ == "__main__":
    main()
