"""Plot the empirical CDF of observed track lifetimes from an annotation-tool
manifest. Right-censored tracks (no observed disappearance) do NOT contribute
to the rising portion of the CDF — they leave the curve plateaued below 1.0
by the censoring fraction.

Lifetime is defined as (last_observed_ts - first_observed_ts) per track. A
track is considered "completed" if its boundaries record a non-null
`first_absent_after_ts` — meaning we observed it disappear within the
manifest's observation window. Otherwise it's censored.

x-axis is cropped to 1.1 × max observed lifetime so the plateau is visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_US_TO_H = 1_000_000.0 * 3600.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, nargs="+", required=True,
                   help="One or more annotation-tool manifests to plot. "
                        "Multiple manifests are overlaid on the same axes.")
    p.add_argument("--label", type=str, nargs="+", default=None,
                   help="Curve labels (one per --manifest). Defaults to the "
                        "manifest's parent directory name.")
    p.add_argument("--out", type=Path, required=True, help="Output image path (.png/.pdf/.svg).")
    p.add_argument("--units", choices=["hours", "minutes", "seconds"], default="hours")
    p.add_argument("--title", type=str, default="Track lifetime ECDF")
    return p.parse_args()


def load_lifetimes(manifest_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (completed_lifetimes, censored_lifetimes), both in microseconds."""
    manifest = json.loads(manifest_path.read_text())
    manifest_dir = manifest_path.parent
    completed: list[int] = []
    censored: list[int] = []
    for rel in manifest["track_files"]:
        track = json.loads((manifest_dir / rel).read_text())
        b = track["boundaries"]
        lifetime_us = b["last_observed_ts"] - b["first_observed_ts"]
        if b.get("first_absent_after_ts") is not None:
            completed.append(lifetime_us)
        else:
            censored.append(lifetime_us)
    return np.array(completed, dtype=np.int64), np.array(censored, dtype=np.int64)


def main() -> int:
    args = parse_args()
    if args.label is not None and len(args.label) != len(args.manifest):
        raise SystemExit("--label count must match --manifest count.")
    labels = args.label or [m.parent.name for m in args.manifest]

    scale = {"hours": _US_TO_H, "minutes": _US_TO_H / 60.0, "seconds": _US_TO_H / 3600.0}[args.units]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    global_max_observed = 0.0
    for i, (manifest, label) in enumerate(zip(args.manifest, labels)):
        completed_us, censored_us = load_lifetimes(manifest)
        n_total = len(completed_us) + len(censored_us)
        if n_total == 0:
            print(f"WARNING: {manifest} contained no tracks; skipping.")
            continue
        n_completed = len(completed_us)
        completion_rate = n_completed / n_total
        completed = np.sort(completed_us) / scale
        max_observed = float(max(completed.max() if completed.size else 0.0,
                                 (censored_us.max() / scale) if censored_us.size else 0.0))
        global_max_observed = max(global_max_observed, max_observed)

        color = colors[i % len(colors)]
        # Step plot: y values are (1..n_completed)/n_total, so the curve plateaus
        # at completion_rate (below 1.0 by the censored fraction).
        if completed.size > 0:
            ys = np.arange(1, n_completed + 1) / n_total
            xs = np.concatenate([[0.0], completed])
            ys = np.concatenate([[0.0], ys])
            ax.step(xs, ys, where="post", color=color, linewidth=1.8,
                    label=f"{label}  (n={n_total}, completed={n_completed}, "
                          f"censored={n_total - n_completed})")
            # Plateau line out to right edge, annotate value.
            ax.hlines(completion_rate, xmin=completed[-1], xmax=max_observed * 1.1,
                      colors=color, linestyles="--", linewidth=1.0, alpha=0.6)
            ax.text(max_observed * 1.1, completion_rate, f" {completion_rate:.2f}",
                    color=color, va="center", fontsize=9)

        # Print summary stats to stdout for the user.
        print(f"{manifest}")
        print(f"  total tracks:        {n_total}")
        print(f"  completed (observed disappearance): {n_completed}  ({completion_rate*100:.1f}%)")
        print(f"  censored (no disappearance seen):    {n_total - n_completed}  "
              f"({(1-completion_rate)*100:.1f}%)")
        if completed.size:
            print(f"  observed lifetime ({args.units}): "
                  f"min={completed.min():.3f}  median={np.median(completed):.3f}  "
                  f"max={completed.max():.3f}")

    ax.set_xlabel(f"observed lifetime ({args.units})")
    ax.set_ylabel("fraction of tracks (ECDF of completed)")
    ax.set_xlim(0, global_max_observed * 1.1)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_title(args.title)
    ax.legend(loc="lower right", fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nWrote plot: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
