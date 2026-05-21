"""Quick check: does the test-set anchor-sampling seed materially change the
distribution of true (remaining) disappearance times under the
percentile_horizons eval scenario?

The benchmark_dataset preset hard-codes `shuffle_seed: 42` and
`max_anchor_observations: 1000`. This script replicates that sampling logic
*and* the percentile_horizons truth computation offline (no inferencer / no
GPU / no eval pipeline), sweeps multiple seeds, and overlays one ECDF per
seed.

Truth definition — bit-for-bit equivalent to the eval pipeline:

  1. Per-track bounds are read from the manifest (first/last_observed_ts,
     last_absent_before_ts, first_absent_after_ts), matching
     persistence_eval.datasets.annotation_tool.AnnotationToolTrackView.
  2. A single dataset-wide horizon list is built once via the
     percentile_horizons scenario: take the 90th percentile of per-track
     persistence times (last_observed - first_observed), then power-space 20
     horizons in [t_min, percentile_value * max_query_inflation].
     Mirrors persistence_eval.sampling.percentile_horizon_episode_provider.
  3. Per anchor obs at time t_a, ground_truth at horizon h is determined by
     TrackTemporalBounds.exists_at(t_a + h) — True/False/None (None = unknown).
  4. Per-episode disappearance time = smallest horizon h where exists_at
     returns False. If never False (right-censored), capped at max horizon
     (matches the cap used in plot_disappearance_distributions.py).

Usage (inside the container):
  python scripts/check_truth_seed_variance.py \\
      --manifest /workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split/splits/50_50/test_from_2012-10-16.json \\
      --n-seeds 10 \\
      --out /workspace/data/output/persistence_eval_plots/truth_seed_variance/ecdf_hours.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("check_truth_seed_variance")

_POSITIVE = "Positive"
_US_TO_S = 1_000_000.0


@dataclass
class Bounds:
    first_observed_time: float
    last_observed_time: float
    first_absent_before_observation: float | None
    first_missing_time: float | None

    def exists_at(self, t: float) -> bool | None:
        if self.first_observed_time <= t <= self.last_observed_time:
            return True
        if (self.first_absent_before_observation is not None
                and t <= self.first_absent_before_observation):
            return False
        if (self.first_missing_time is not None
                and t >= self.first_missing_time):
            return False
        return None


@dataclass
class Track:
    track_id: str
    bounds: Bounds
    positive_obs_us: list[int]  # timestamps (us) of positive observations with mask_path


def load_tracks(manifest_path: Path) -> list[Track]:
    manifest = json.loads(manifest_path.read_text())
    manifest_dir = manifest_path.parent

    tracks: list[Track] = []
    for rel in manifest["track_files"]:
        t = json.loads((manifest_dir / rel).read_text())
        b = t["boundaries"]
        bounds = Bounds(
            first_observed_time=b["first_observed_ts"] / _US_TO_S,
            last_observed_time=b["last_observed_ts"] / _US_TO_S,
            first_absent_before_observation=(
                b["last_absent_before_ts"] / _US_TO_S
                if b.get("last_absent_before_ts") is not None else None
            ),
            first_missing_time=(
                b["first_absent_after_ts"] / _US_TO_S
                if b.get("first_absent_after_ts") is not None else None
            ),
        )
        positive_obs_us = [
            int(o["timestamp"])
            for o in t["observations"]
            if o["state"] == _POSITIVE and o.get("mask_path") is not None
        ]
        if not positive_obs_us:
            continue
        tracks.append(Track(track_id=t["track_id"], bounds=bounds,
                            positive_obs_us=positive_obs_us))
    return tracks


def build_percentile_horizons(
    tracks: list[Track],
    *,
    percentile: float = 90.0,
    num_query_times: int = 20,
    t_min: float = 0.0,
    max_query_inflation: float = 1.0,
    power_exponent: float = 2.0,
) -> np.ndarray:
    """Replicates PercentileHorizonEpisodeProvider with spacing='power' — i.e.
    the defaults baked into configs/train_confs/eval_scenario/percentile_horizons.yaml.
    """
    persistence_times = [
        t.bounds.last_observed_time - t.bounds.first_observed_time for t in tracks
    ]
    if not persistence_times:
        raise ValueError("no tracks — cannot build horizons")
    percentile_value = float(np.percentile(persistence_times, percentile))
    cap = max_query_inflation * percentile_value
    u = np.linspace(0.0, 1.0, num_query_times)
    return t_min + (cap - t_min) * u ** power_exponent


def disappearance_time_for_anchor(
    bounds: Bounds, anchor_time_s: float, horizons: np.ndarray
) -> float:
    """Mirrors per-episode disappearance time used by
    plot_disappearance_distributions.py: smallest horizon at which
    bounds.exists_at(anchor + h) returns False; capped at max horizon if
    never False (None counts as "still here" for the cap — same as having
    unknown ground truth at that horizon).
    """
    cap = float(horizons[-1])
    for h in horizons:
        if bounds.exists_at(anchor_time_s + float(h)) is False:
            return float(h)
    return cap


def sample_disappearance_times(
    tracks: list[Track],
    horizons: np.ndarray,
    seed: int,
    n_anchors: int,
) -> list[float]:
    # obs_list mirrors AnnotationToolBackingDataset._get_image_observations:
    # one entry per positive observation across all tracks (after mask_path filter).
    obs_list: list[tuple[Bounds, int]] = []
    for tr in tracks:
        for ts_us in tr.positive_obs_us:
            obs_list.append((tr.bounds, ts_us))
    random.Random(seed).shuffle(obs_list)
    obs_list = obs_list[:n_anchors]
    return [
        disappearance_time_for_anchor(b, ts_us / _US_TO_S, horizons)
        for b, ts_us in obs_list
    ]


def ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    if not values:
        return [], []
    xs = sorted(values)
    n = len(xs)
    step_x = [xs[0]]
    step_y = [0.0]
    for i, x in enumerate(xs):
        step_x.append(x)
        step_y.append((i + 1) / n)
    return step_x, step_y


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True,
                   help="Path to test manifest (e.g., test_from_2012-10-16.json).")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--first-seed", type=int, default=0,
                   help="Seeds used are first_seed, first_seed+1, ... "
                        "first_seed+n_seeds-1. Set to 42 to include the "
                        "preset's default as the first curve.")
    p.add_argument("--n-anchors", type=int, default=1000,
                   help="Matches preset's max_anchor_observations (default 1000).")
    p.add_argument("--percentile", type=float, default=90.0,
                   help="Matches eval_scenario/percentile_horizons.yaml.")
    p.add_argument("--num-query-times", type=int, default=20)
    p.add_argument("--power-exponent", type=float, default=2.0)
    p.add_argument("--max-query-inflation", type=float, default=1.0)
    p.add_argument("--time-units", choices=["seconds", "hours", "days"],
                   default="hours")
    p.add_argument("--xlog", action="store_true")
    p.add_argument("--out", type=Path, required=True,
                   help="Output PDF/PNG path. Parent dir created if needed.")
    args = p.parse_args(argv)

    tracks = load_tracks(args.manifest)
    logger.info("loaded %d tracks from %s", len(tracks), args.manifest)
    n_pos_obs = sum(len(t.positive_obs_us) for t in tracks)
    logger.info("total positive observations (with mask): %d", n_pos_obs)

    horizons = build_percentile_horizons(
        tracks,
        percentile=args.percentile,
        num_query_times=args.num_query_times,
        max_query_inflation=args.max_query_inflation,
        power_exponent=args.power_exponent,
    )
    logger.info("horizons: %s",
                np.array2string(horizons, precision=2, suppress_small=True))

    if n_pos_obs < args.n_anchors:
        logger.warning("manifest has %d positive obs; preset asks for %d — "
                       "every seed will sample the same full set",
                       n_pos_obs, args.n_anchors)

    time_scale = {"seconds": 1.0, "hours": 1.0 / 3600.0,
                  "days": 1.0 / 86400.0}[args.time_units]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i in range(args.n_seeds):
        seed = args.first_seed + i
        ts = sample_disappearance_times(tracks, horizons, seed, args.n_anchors)
        xs, ys = ecdf(ts)
        ax.step([x * time_scale for x in xs], ys, where="post",
                alpha=0.75, label=f"seed={seed}")
        sorted_ts = sorted(ts)
        median = sorted_ts[len(sorted_ts) // 2]
        logger.info("seed=%d: n=%d, median=%.2f %s",
                    seed, len(ts), median * time_scale, args.time_units)

    ax.set_xlabel(f"True disappearance time ({args.time_units} past anchor)")
    ax.set_ylabel("ECDF")
    ax.set_ylim(0.0, 1.02)
    if args.xlog:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Ground-truth disappearance ECDF across {args.n_seeds} shuffle seeds "
                 f"(percentile_horizons, n_anchors={args.n_anchors})")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
