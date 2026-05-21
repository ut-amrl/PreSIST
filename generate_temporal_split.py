"""Offline temporal train/test splitter for annotation-tool manifests.

Reads a source manifest, classifies each track by the dates of its positive
observations, and emits new train and test manifests that reference the
original per-track JSONs unchanged. See plans for design rationale.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

SCRIPT_VERSION = "2.0"
_POSITIVE = "Positive"  # mirrors persistence_eval.datasets.annotation_tool._POSITIVE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="Path to source manifest JSON")
    p.add_argument("--train-fraction", type=float, default=0.7)
    p.add_argument("--val-fraction", type=float, default=0.0,
                   help="If >0, produces a 3-way train/val/test split. Val window sits "
                        "between train and test. Default 0.0 = legacy 2-way.")
    p.add_argument("--min-test-tracks", type=int, default=1000,
                   help="Hard floor on test-side track count; abort if not met. Default 1000.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and print the split + per-class track counts but write nothing.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Defaults to <manifest_parent>/splits/<train_pct>_<test_pct> (2-way) "
                        "or <manifest_parent>/splits/<train_pct>_<val_pct>_<test_pct> (3-way)")
    p.add_argument("--force", action="store_true", help="Overwrite existing output files")
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


def compute_date_split(
    tracks: list[tuple[Path, dict]],
    train_fraction: float,
    val_fraction: float = 0.0,
) -> dict:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"--train-fraction must be in (0, 1); got {train_fraction}")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"--val-fraction must be in [0, 1); got {val_fraction}")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            f"train_fraction + val_fraction must be < 1; got "
            f"{train_fraction} + {val_fraction} = {train_fraction + val_fraction}"
        )

    all_tags: set[str] = set()
    for _, track in tracks:
        for obs in track["observations"]:
            all_tags.add(obs["sequence_tag"])
    if not all_tags:
        raise ValueError("No observations found in any track.")

    min_date = dt.date.fromisoformat(min(all_tags))
    max_date = dt.date.fromisoformat(max(all_tags))
    total_days = (max_date - min_date).days + 1
    three_way = val_fraction > 0.0
    if three_way and total_days < 3:
        raise ValueError(f"Cannot 3-way split: only {total_days} day(s) of data.")
    if not three_way and total_days < 2:
        raise ValueError(f"Cannot split: only {total_days} day(s) of data.")

    train_days = int(train_fraction * total_days)  # floor for positive floats
    if three_way:
        val_days = int(val_fraction * total_days)
        test_days = total_days - train_days - val_days
        if train_days < 1 or val_days < 1 or test_days < 1:
            raise ValueError(
                f"train/val/test day counts must each be >= 1; got "
                f"train={train_days}, val={val_days}, test={test_days} "
                f"(total_days={total_days})"
            )
        train_end_date = min_date + dt.timedelta(days=train_days - 1)
        val_start_date = min_date + dt.timedelta(days=train_days)
        val_end_date = min_date + dt.timedelta(days=train_days + val_days - 1)
        test_start_date = min_date + dt.timedelta(days=train_days + val_days)
        return {
            "min_date": min_date.isoformat(),
            "max_date": max_date.isoformat(),
            "total_days": total_days,
            "train_days": train_days,
            "val_days": val_days,
            "test_days": test_days,
            "train_end_date": train_end_date.isoformat(),
            "val_start_date": val_start_date.isoformat(),
            "val_end_date": val_end_date.isoformat(),
            "test_start_date": test_start_date.isoformat(),
        }
    # legacy 2-way
    if train_days < 1 or train_days >= total_days:
        raise ValueError(
            f"train_fraction={train_fraction} with total_days={total_days} yields "
            f"train_days={train_days}; need 1 <= train_days < total_days."
        )
    train_end_date = min_date + dt.timedelta(days=train_days - 1)
    test_start_date = min_date + dt.timedelta(days=train_days)
    return {
        "min_date": min_date.isoformat(),
        "max_date": max_date.isoformat(),
        "total_days": total_days,
        "train_days": train_days,
        "test_days": total_days - train_days,
        "train_end_date": train_end_date.isoformat(),
        "test_start_date": test_start_date.isoformat(),
    }


def classify_track(track: dict, info: dict) -> str:
    """Return 'train', 'val', 'test', or 'discarded' for a single track.

    `info` is the dict returned by `compute_date_split`. 3-way mode is
    selected by the presence of a 'val_start_date' key.
    """
    three_way = "val_start_date" in info
    train_end = info["train_end_date"]
    test_start = info["test_start_date"]
    if three_way:
        val_start = info["val_start_date"]
        val_end = info["val_end_date"]

    has_train_pos = False
    has_val_pos = False
    has_test_pos = False
    has_any_pos = False
    for obs in track["observations"]:
        if obs["state"] != _POSITIVE:
            continue
        has_any_pos = True
        tag = obs["sequence_tag"]
        if tag <= train_end:
            has_train_pos = True
        elif three_way and val_start <= tag <= val_end:
            has_val_pos = True
        elif tag >= test_start:
            has_test_pos = True
        # short-circuit: any track with positives in >1 window is discarded
        if (int(has_train_pos) + int(has_val_pos) + int(has_test_pos)) > 1:
            return "discarded"
    if not has_any_pos:
        return "discarded"
    if has_train_pos:
        return "train"
    if has_val_pos:
        return "val"
    return "test"


def relpath_str(target: Path, start: Path) -> str:
    return os.path.relpath(target, start)


def build_manifest(orig_manifest: dict, suffix: str, track_paths: list[Path],
                   output_parent: Path) -> dict:
    return {
        "run_id": f"{orig_manifest['run_id']}_{suffix}",
        "location_root": orig_manifest["location_root"],
        "track_files": [relpath_str(tp, output_parent) for tp in track_paths],
    }


def write_json(path: Path, payload: dict, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; use --force to overwrite.")
    path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest, raw_manifest_bytes, tracks = load_manifest_and_tracks(manifest_path)
    info = compute_date_split(tracks, args.train_fraction, args.val_fraction)
    three_way = "val_start_date" in info

    train_paths: list[Path] = []
    val_paths: list[Path] = []
    test_paths: list[Path] = []
    discarded: list[tuple[str, Path]] = []
    train_pos_obs = val_pos_obs = test_pos_obs = 0

    for track_path, track in sorted(tracks, key=lambda x: x[1]["track_id"]):
        side = classify_track(track, info)
        n_pos = sum(1 for o in track["observations"] if o["state"] == _POSITIVE)
        if side == "train":
            train_paths.append(track_path)
            train_pos_obs += n_pos
        elif side == "val":
            val_paths.append(track_path)
            val_pos_obs += n_pos
        elif side == "test":
            test_paths.append(track_path)
            test_pos_obs += n_pos
        else:
            discarded.append((track["track_id"], track_path))

    train_pct = int(round(args.train_fraction * 100))
    if three_way:
        val_pct = int(round(args.val_fraction * 100))
        test_pct = 100 - train_pct - val_pct
        default_subdir = f"{train_pct}_{val_pct}_{test_pct}"
    else:
        test_pct = 100 - train_pct
        default_subdir = f"{train_pct}_{test_pct}"
    if args.output_dir is None:
        output_dir = manifest_path.parent / "splits" / default_subdir
    else:
        output_dir = args.output_dir.resolve()

    # Print summary first (always — works for dry-run too).
    print("Temporal split summary.")
    print(f"  source manifest:      {manifest_path}")
    if three_way:
        print(f"  date range:           {info['min_date']} .. {info['max_date']}  "
              f"({info['total_days']} days; train={info['train_days']}, "
              f"val={info['val_days']}, test={info['test_days']})")
        print(f"  train_end_date:       {info['train_end_date']}")
        print(f"  val window:           {info['val_start_date']} .. {info['val_end_date']}")
        print(f"  test_start_date:      {info['test_start_date']}")
        print(f"  tracks:               train={len(train_paths)}  val={len(val_paths)}  "
              f"test={len(test_paths)}  discarded={len(discarded)}  total={len(tracks)}")
        print(f"  positive observations: train={train_pos_obs}  val={val_pos_obs}  "
              f"test={test_pos_obs}")
    else:
        print(f"  date range:           {info['min_date']} .. {info['max_date']}  "
              f"({info['total_days']} days; train={info['train_days']}, test={info['test_days']})")
        print(f"  train_end_date:       {info['train_end_date']}")
        print(f"  test_start_date:      {info['test_start_date']}")
        print(f"  tracks:               train={len(train_paths)}  test={len(test_paths)}  "
              f"discarded={len(discarded)}  total={len(tracks)}")
        print(f"  positive observations: train={train_pos_obs}  test={test_pos_obs}")

    # Hard guard on test track count (applies in dry-run too: viability check).
    if len(test_paths) < args.min_test_tracks:
        print(
            f"\nERROR: test side has {len(test_paths)} tracks; --min-test-tracks="
            f"{args.min_test_tracks}. Adjust split fractions and rerun.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest_path = output_dir / f"train_through_{info['train_end_date']}.json"
    test_manifest_path = output_dir / f"test_from_{info['test_start_date']}.json"
    discarded_path = output_dir / "discarded_tracks.json"
    metadata_path = output_dir / "split_metadata.json"

    train_manifest = build_manifest(manifest, "train", train_paths, output_dir)
    test_manifest = build_manifest(manifest, "test", test_paths, output_dir)

    write_targets: list[tuple[Path, dict]] = [
        (train_manifest_path, train_manifest),
        (test_manifest_path, test_manifest),
    ]
    if three_way:
        val_manifest_path = (
            output_dir
            / f"val_from_{info['val_start_date']}_through_{info['val_end_date']}.json"
        )
        val_manifest = build_manifest(manifest, "val", val_paths, output_dir)
        write_targets.append((val_manifest_path, val_manifest))

    discarded_payload = {
        **{k: info[k] for k in (
            "train_end_date",
            "test_start_date",
            *(("val_start_date", "val_end_date") if three_way else ()),
        )},
        "source_manifest": str(manifest_path),
        "discarded": [
            {"track_id": tid, "track_file": relpath_str(tp, output_dir)}
            for tid, tp in discarded
        ],
    }
    metadata_payload = {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(raw_manifest_bytes).hexdigest(),
        "train_fraction_target": args.train_fraction,
        "val_fraction_target": args.val_fraction,
        "min_test_tracks": args.min_test_tracks,
        **info,
        "generator_script_version": SCRIPT_VERSION,
    }
    write_targets.extend([
        (discarded_path, discarded_payload),
        (metadata_path, metadata_payload),
    ])

    for path, payload in write_targets:
        write_json(path, payload, args.force)

    print("\n  output files:")
    for path, _ in write_targets:
        print(f"    {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
