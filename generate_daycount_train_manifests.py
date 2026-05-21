"""Materialize day-count subsets of an annotation-tool train manifest.

For each requested N, writes a self-contained subdirectory with truncated
per-track JSONs (observations within the first N days of the train window)
plus a manifest pointing at them. Used by day-count training sweeps to vary
"how much data has been collected" instead of a random fraction.

Window semantics: anchored to the source manifest's earliest sequence_tag
(train_start). Observations are kept iff their date is in
[train_start, train_start + N) (exclusive cutoff). Tracks with no kept
positive observations are dropped. Boundaries are recomputed from the kept
observations.

Output layout per N:
  <output_root>/days_<N:04d>/
    train_first_<N>d_through_<cutoff_minus_one>.json
    tracks/<track_id>.json
    daycount_metadata.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

SCRIPT_VERSION = "1.0"
_POSITIVE = "Positive"
_ABSENT = "Absent"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-train-manifest", type=Path, required=True,
                   help="Path to source train manifest JSON (output of generate_temporal_split.py).")
    p.add_argument("--days", type=float, nargs="+", required=True,
                   help="Day-count values to materialize (each > 0). Fractional "
                        "values (e.g. 0.5) are supported and trigger timestamp-"
                        "based truncation instead of date-based.")
    p.add_argument("--output-root", type=Path, default=None,
                   help="Defaults to <source_manifest_parent>/daycount/")
    p.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return p.parse_args()


def load_manifest_and_tracks(manifest_path: Path) -> tuple[dict, bytes, list[tuple[Path, dict]]]:
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    manifest_dir = manifest_path.parent
    tracks: list[tuple[Path, dict]] = []
    for rel in manifest["track_files"]:
        track_path = (manifest_dir / rel).resolve()
        track = json.loads(track_path.read_text())
        tracks.append((track_path, track))
    return manifest, raw, tracks


def compute_train_start(tracks: list[tuple[Path, dict]]) -> dt.date:
    all_tags: set[str] = set()
    for _, track in tracks:
        for obs in track["observations"]:
            all_tags.add(obs["sequence_tag"])
    if not all_tags:
        raise ValueError("No observations found in source manifest.")
    return dt.date.fromisoformat(min(all_tags))


def truncate_track(track: dict, cutoff_date: dt.date, cutoff_ts: float | None) -> dict | None:
    """Return a new track dict containing observations strictly before the cutoff,
    with recomputed boundaries and sequence_tags. Returns None if no positive
    observations remain (track dropped).

    If `cutoff_ts` is given (fractional-day case), the cutoff is applied at the
    observation timestamp level. Otherwise the integer-day path keeps the
    original sequence_tag (date) comparison for bit-for-bit compatibility with
    previously-generated splits.
    """
    if cutoff_ts is not None:
        kept_obs = [o for o in track["observations"] if o["timestamp"] < cutoff_ts]
    else:
        cutoff_iso = cutoff_date.isoformat()
        kept_obs = [o for o in track["observations"] if o["sequence_tag"] < cutoff_iso]
    pos_obs = [o for o in kept_obs if o["state"] == _POSITIVE]
    if not pos_obs:
        return None

    pos_ts = [o["timestamp"] for o in pos_obs]
    first_observed_ts = min(pos_ts)
    last_observed_ts = max(pos_ts)

    # Recompute absent-flank bounds from the kept observations themselves.
    # `last_absent_before_ts`: latest Absent strictly before first positive.
    # `first_absent_after_ts`: earliest Absent strictly after last positive.
    absent_before = [o["timestamp"] for o in kept_obs
                     if o["state"] == _ABSENT and o["timestamp"] < first_observed_ts]
    absent_after = [o["timestamp"] for o in kept_obs
                    if o["state"] == _ABSENT and o["timestamp"] > last_observed_ts]
    last_absent_before_ts = max(absent_before) if absent_before else None
    first_absent_after_ts = min(absent_after) if absent_after else None

    new_track = dict(track)
    new_track["observations"] = kept_obs
    new_track["sequence_tags"] = sorted({o["sequence_tag"] for o in kept_obs})
    new_track["boundaries"] = {
        "first_observed_ts": first_observed_ts,
        "last_observed_ts": last_observed_ts,
        "last_absent_before_ts": last_absent_before_ts,
        "first_absent_after_ts": first_absent_after_ts,
    }
    return new_track


def write_json(path: Path, payload: dict, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; use --force to overwrite.")
    path.write_text(json.dumps(payload, indent=2))


def relpath_str(target: Path, start: Path) -> str:
    return os.path.relpath(target, start)


def _day_slug(n_days: float) -> str:
    """Filesystem-safe day token. Integers keep the legacy zero-padded form so
    pre-existing splits (days_0001, days_0035, ...) are unaffected; fractional
    values use a dot-free decimal (e.g. 0.5 -> '0p5', 1.25 -> '1p25')."""
    if float(n_days).is_integer():
        return f"{int(n_days):04d}"
    return f"{n_days:g}".replace(".", "p")


def _to_epoch_seconds(ts: float) -> float:
    """Normalise a timestamp to seconds-since-epoch regardless of whether the
    source stores it in seconds (~1e9), milliseconds (~1e12), microseconds
    (~1e15), or nanoseconds (~1e18). Used only for human-readable metadata;
    quantile comparisons stay in the source's native units."""
    abs_ts = abs(ts)
    if abs_ts >= 1e17:
        return ts / 1e9      # ns -> s
    if abs_ts >= 1e14:
        return ts / 1e6      # us -> s
    if abs_ts >= 1e11:
        return ts / 1e3      # ms -> s
    return ts                # already s


def _partial_day_cutoff_ts(
    tracks: list[tuple[Path, dict]],
    partial_day_date: dt.date,
    fraction: float,
) -> float:
    """For fractional-day truncation, anchor the cutoff at the empirical
    `fraction`-quantile of observation timestamps recorded ON `partial_day_date`,
    so e.g. fraction=0.5 picks the median image of that day. This respects the
    fact that observations aren't uniform over a 24h window (PKLot only collects
    during daylight) — "half a day of training data" should mean "half the
    images that exist for that day", not "midnight + 12h"."""
    day_iso = partial_day_date.isoformat()
    # Dedupe across tracks: a single image timestamp is shared by every track
    # visible in that frame, so pooling all (track, obs) pairs would weight the
    # quantile by tracks-per-frame instead of by frame. We want one count per
    # *image*.
    day_ts = sorted({
        o["timestamp"]
        for _, track in tracks
        for o in track["observations"]
        if o["sequence_tag"] == day_iso
    })
    if not day_ts:
        raise ValueError(
            f"No observations on partial-day date {day_iso}; cannot derive "
            f"empirical {fraction:g}-quantile cutoff."
        )
    # Linear-interp quantile (numpy-default style); avoids a numpy dep.
    if fraction <= 0.0:
        return float(day_ts[0])
    if fraction >= 1.0:
        return float(day_ts[-1])
    pos = fraction * (len(day_ts) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(day_ts) - 1)
    frac = pos - lo
    return float(day_ts[lo] * (1.0 - frac) + day_ts[hi] * frac)


def materialize_n(
    n_days: float,
    train_start: dt.date,
    src_manifest: dict,
    src_manifest_path: Path,
    src_manifest_sha: str,
    tracks: list[tuple[Path, dict]],
    output_root: Path,
    force: bool,
) -> tuple[Path, int, int, int]:
    if n_days <= 0:
        raise ValueError(f"--days values must be > 0; got {n_days}")
    is_integer_days = float(n_days).is_integer()
    cutoff_date = train_start + dt.timedelta(days=n_days)
    cutoff_ts: float | None
    if is_integer_days:
        cutoff_ts = None
        last_kept_date = cutoff_date - dt.timedelta(days=1)
        manifest_stub = f"train_first_{int(n_days)}d_through_{last_kept_date.isoformat()}.json"
    else:
        # n_days = k + f, 0 < f < 1: keep all observations from the k full days
        # before the partial day, plus those on day k whose timestamp is below
        # the empirical f-quantile of that day's images.
        k_full = int(n_days)
        fraction = n_days - k_full
        partial_day_date = train_start + dt.timedelta(days=k_full)
        cutoff_ts = _partial_day_cutoff_ts(tracks, partial_day_date, fraction)
        manifest_stub = f"train_first_{_day_slug(n_days)}d.json"
    slug = _day_slug(n_days)
    out_dir = output_root / f"days_{slug}"
    tracks_dir = out_dir / "tracks"
    manifest_path = out_dir / manifest_stub
    metadata_path = out_dir / "daycount_metadata.json"

    if not force and manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists; use --force to overwrite.")

    out_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    kept_track_paths: list[Path] = []
    n_dropped = 0
    n_kept_obs = 0
    for _src_path, track in sorted(tracks, key=lambda x: x[1]["track_id"]):
        truncated = truncate_track(track, cutoff_date, cutoff_ts)
        if truncated is None:
            n_dropped += 1
            continue
        n_kept_obs += len(truncated["observations"])
        out_track_path = tracks_dir / f"{truncated['track_id']}.json"
        # Always overwrite track files when force is on; if force is off and
        # the file is somehow already there from a partial prior run, error.
        if out_track_path.exists() and not force:
            raise FileExistsError(
                f"{out_track_path} exists; use --force to overwrite."
            )
        out_track_path.write_text(json.dumps(truncated, indent=2))
        kept_track_paths.append(out_track_path)

    new_manifest = {
        "run_id": f"{src_manifest['run_id']}_first{n_days}d",
        "location_root": src_manifest["location_root"],
        "track_files": [relpath_str(p, out_dir) for p in kept_track_paths],
    }
    metadata_payload = {
        "source_manifest": str(src_manifest_path),
        "source_manifest_sha256": src_manifest_sha,
        "train_start": train_start.isoformat(),
        "n_days": n_days,
        "cutoff_date_exclusive": cutoff_date.isoformat(),
        "cutoff_timestamp_exclusive": cutoff_ts,
        "cutoff_datetime_utc_exclusive": (
            # Track observation timestamps may be stored in seconds (~1.35e9),
            # milliseconds (~1.35e12), or microseconds (~1.35e15) depending on
            # the source. Scale down to the seconds-since-epoch regime expected
            # by fromtimestamp.
            dt.datetime.fromtimestamp(
                _to_epoch_seconds(cutoff_ts),
                tz=dt.timezone.utc,
            ).isoformat()
            if cutoff_ts is not None else None
        ),
        "partial_day_anchor": (
            {
                "date": (train_start + dt.timedelta(days=int(n_days))).isoformat(),
                "fraction": n_days - int(n_days),
                "quantile_method": "linear_interp_over_day_image_timestamps",
            } if not is_integer_days else None
        ),
        "last_kept_date": (
            (cutoff_date - dt.timedelta(days=1)).isoformat() if is_integer_days else None
        ),
        "n_kept_tracks": len(kept_track_paths),
        "n_dropped_tracks": n_dropped,
        "n_kept_observations": n_kept_obs,
        "generator_script_version": SCRIPT_VERSION,
    }
    write_json(manifest_path, new_manifest, force)
    write_json(metadata_path, metadata_payload, force)
    return manifest_path, len(kept_track_paths), n_dropped, n_kept_obs


def main() -> int:
    args = parse_args()
    src_path = args.source_train_manifest.resolve()
    if not src_path.is_file():
        print(f"ERROR: source manifest not found: {src_path}", file=sys.stderr)
        return 1
    if args.output_root is None:
        output_root = src_path.parent / "daycount"
    else:
        output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    src_manifest, raw_bytes, tracks = load_manifest_and_tracks(src_path)
    src_sha = hashlib.sha256(raw_bytes).hexdigest()
    train_start = compute_train_start(tracks)
    print(f"Source manifest: {src_path}")
    print(f"Train start (anchor): {train_start.isoformat()}")
    print(f"Total source tracks: {len(tracks)}")
    print(f"Output root: {output_root}")
    print(f"Days to materialize: {sorted(set(args.days))}\n")

    for n in sorted(set(args.days)):
        manifest_path, n_kept, n_dropped, n_obs = materialize_n(
            n, train_start, src_manifest, src_path, src_sha, tracks, output_root, args.force,
        )
        print(f"  N={n:>6g}: kept_tracks={n_kept:>5d}  dropped={n_dropped:>5d}  "
              f"kept_obs={n_obs:>7d}  ->  {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
