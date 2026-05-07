#!/usr/bin/env python3
"""
build_gps_index.py

For each Oxford RobotCar traversal, maps every stereo camera timestamp
to its nearest GPS/RTK position fix. Works even if images haven't been
converted yet — reads directly from stereo.timestamps in downloads.

Priority:
  1. RTK (<rtk_root>/<traversal>/rtk.csv)  — most accurate
  2. GPS (<downloads_root>/<traversal>/gps/gps.csv)  — fallback

Output per traversal:
  <extracted_root>/<traversal>/gps_index.csv

Combined output:
  <extracted_root>/all_traversals_index.csv

Usage (build index):
  python3 build_gps_index.py \
    --downloads_root /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/robotCar/downloads \
    --rtk_root       /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/robotCar/rtk \
    --extracted_root /robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/robotcar

Usage (query only):
  python3 build_gps_index.py --query \
    --northing 5735800 --easting 620100 --radius 50 \
    --extracted_root /robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/robotcar

Usage (query + export matching images):
  python3 build_gps_index.py --query \
    --northing 5735800 --easting 620100 --radius 50 \
    --extracted_root /robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/robotcar \
    --copy_to /tmp/query_results
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# RTK folder name fuzzy matching
# ---------------------------------------------------------------------------

def find_rtk_csv(rtk_root: Path, traversal_tag: str) -> Path | None:
    """
    Find rtk.csv for a traversal, handling slight name mismatches.
    Tries exact match first, then fuzzy match by normalizing separators/zeros.
    """
    exact = rtk_root / traversal_tag / "rtk.csv"
    if exact.exists():
        return exact

    def normalize(tag):
        return tag.replace("-0", "-").replace("-", "")

    target_norm = normalize(traversal_tag)
    for d in rtk_root.iterdir():
        if d.is_dir() and normalize(d.name) == target_norm:
            candidate = d / "rtk.csv"
            if candidate.exists():
                return candidate

    return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_stereo_timestamps(downloads_dir: Path, traversal_tag: str) -> list[int]:
    """Load stereo.timestamps -> list of timestamps in microseconds."""
    ts_file = downloads_dir / traversal_tag / "stereo.timestamps"
    if not ts_file.exists():
        return []
    timestamps = []
    with open(ts_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                timestamps.append(int(parts[0]))
            except ValueError:
                continue
    return timestamps


def load_rtk(rtk_csv: Path):
    timestamps, rows = [], []
    with open(rtk_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                timestamps.append(int(row["timestamp"]))
                rows.append(row)
            except (ValueError, KeyError):
                continue
    return np.array(timestamps, dtype=np.int64), rows


def load_gps(gps_csv: Path):
    timestamps, rows = [], []
    with open(gps_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row.get("num_satellites", 0)) < 4:
                    continue
                timestamps.append(int(row["timestamp"]))
                rows.append(row)
            except (ValueError, KeyError):
                continue
    return np.array(timestamps, dtype=np.int64), rows


def nearest_index(sorted_ts: np.ndarray, query_us: int) -> int:
    idx = np.searchsorted(sorted_ts, query_us)
    if idx == 0:
        return 0
    if idx == len(sorted_ts):
        return len(sorted_ts) - 1
    before, after = idx - 1, idx
    if abs(sorted_ts[before] - query_us) <= abs(sorted_ts[after] - query_us):
        return before
    return after


# ---------------------------------------------------------------------------
# Per-traversal indexing
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "image_timestamp_ms", "source",
    "latitude", "longitude",
    "northing", "easting", "altitude",
    "roll", "pitch", "yaw",
    "nearest_fix_time_diff_ms",
]

COMBINED_FIELDNAMES = ["traversal"] + FIELDNAMES


def build_traversal_index(
    traversal_tag: str,
    downloads_root: Path,
    rtk_root: Path,
    output_dir: Path,
):
    stereo_ts_us = load_stereo_timestamps(downloads_root, traversal_tag)
    if not stereo_ts_us:
        print(f"  [{traversal_tag}] SKIP - no stereo.timestamps found")
        return None

    rtk_csv = find_rtk_csv(rtk_root, traversal_tag)
    gps_csv = downloads_root / traversal_tag / "gps" / "gps.csv"

    if rtk_csv is not None:
        ts_arr, pos_rows = load_rtk(rtk_csv)
        source = "rtk"
        print(f"  [{traversal_tag}] RTK  ({len(ts_arr)} fixes, {len(stereo_ts_us)} frames)")
    elif gps_csv.exists():
        ts_arr, pos_rows = load_gps(gps_csv)
        source = "gps"
        print(f"  [{traversal_tag}] GPS  ({len(ts_arr)} fixes, {len(stereo_ts_us)} frames)")
    else:
        print(f"  [{traversal_tag}] SKIP - no RTK or GPS found")
        return None

    if len(ts_arr) == 0:
        print(f"  [{traversal_tag}] SKIP - no valid position fixes")
        return None

    out_rows = []
    for ts_us in stereo_ts_us:
        ts_ms = ts_us // 1000
        idx = nearest_index(ts_arr, ts_us)
        pos = pos_rows[idx]
        time_diff_ms = int(abs(int(ts_arr[idx]) - ts_us) // 1000)

        def _f(key, fallback=""):
            v = pos.get(key, fallback)
            return v if v not in ("", None) else fallback

        lat      = _f("latitude",  "0")
        lon      = _f("longitude", "0")
        northing = _f("northing",  "0")
        easting  = _f("easting",   "0")
        altitude = _f("altitude",  _f("down", "0"))
        roll     = _f("roll",  "")
        pitch    = _f("pitch", "")
        yaw      = _f("yaw",   "")

        out_rows.append({
            "image_timestamp_ms":       ts_ms,
            "source":                   source,
            "latitude":                 f"{float(lat):.8f}",
            "longitude":                f"{float(lon):.8f}",
            "northing":                 f"{float(northing):.4f}",
            "easting":                  f"{float(easting):.4f}",
            "altitude":                 f"{float(altitude):.4f}",
            "roll":                     f"{float(roll):.6f}"  if roll  else "",
            "pitch":                    f"{float(pitch):.6f}" if pitch else "",
            "yaw":                      f"{float(yaw):.6f}"   if yaw   else "",
            "nearest_fix_time_diff_ms": time_diff_ms,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gps_index.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"           -> {len(out_rows)} rows written to {output_path}")
    return [{"traversal": traversal_tag, **r} for r in out_rows]


# ---------------------------------------------------------------------------
# Spatial query + optional image export
# ---------------------------------------------------------------------------

def run_query(
    extracted_root: Path,
    northing: float,
    easting: float,
    radius: float,
    copy_to: Path | None = None,
):
    index_path = extracted_root / "all_traversals_index.csv"
    if not index_path.exists():
        print(f"ERROR: {index_path} not found. Run without --query first.")
        sys.exit(1)

    results = []
    with open(index_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n = float(row["northing"])
                e = float(row["easting"])
                dist = ((n - northing) ** 2 + (e - easting) ** 2) ** 0.5
                if dist <= radius:
                    results.append((dist, row))
            except ValueError:
                continue

    results.sort(key=lambda x: x[0])

    print(f"\nFound {len(results)} frames within {radius}m of ({northing}, {easting})\n")
    print(f"{'dist_m':>7}  {'traversal':<25}  {'timestamp_ms':<15}  {'lat':>12}  {'lon':>12}  img")
    print("-" * 90)

    converted = 0
    not_converted = 0
    for dist, row in results:
        img_path = extracted_root / row["traversal"] / "rgb" / f"{row['image_timestamp_ms']}.png"
        exists = img_path.exists()
        status = "ok" if exists else "not converted"
        if exists:
            converted += 1
        else:
            not_converted += 1
        print(
            f"{dist:>7.1f}  {row['traversal']:<25}  "
            f"{row['image_timestamp_ms']:<15}  "
            f"{row['latitude']:>12}  {row['longitude']:>12}  {status}"
        )

    print(f"\nSummary: {converted} images ready, {not_converted} not yet converted")

    # Export images if --copy_to specified
    if copy_to is not None:
        copy_to.mkdir(parents=True, exist_ok=True)
        print(f"\nExporting converted images to {copy_to} ...")
        copied = 0
        skipped = 0
        for dist, row in results:
            src = extracted_root / row["traversal"] / "rgb" / f"{row['image_timestamp_ms']}.png"
            if not src.exists():
                skipped += 1
                continue

            # Organize by traversal subdirectory
            dest_dir = copy_to / row["traversal"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            shutil.copy2(src, dest)
            copied += 1

        print(f"Copied {copied} images to {copy_to}")
        if skipped > 0:
            print(f"Skipped {skipped} images (not yet converted)")

        # Write a summary CSV in the export dir
        summary_path = copy_to / "query_results.csv"
        with open(summary_path, "w", newline="") as f:
            fieldnames = ["dist_m", "traversal", "image_timestamp_ms",
                         "latitude", "longitude", "northing", "easting",
                         "image_path", "converted"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for dist, row in results:
                img_path = copy_to / row["traversal"] / f"{row['image_timestamp_ms']}.png"
                writer.writerow({
                    "dist_m":             f"{dist:.2f}",
                    "traversal":          row["traversal"],
                    "image_timestamp_ms": row["image_timestamp_ms"],
                    "latitude":           row["latitude"],
                    "longitude":          row["longitude"],
                    "northing":           row["northing"],
                    "easting":            row["easting"],
                    "image_path":         str(img_path) if img_path.exists() else "",
                    "converted":          "yes" if img_path.exists() else "no",
                })
        print(f"Query summary CSV: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extracted_root", required=True)
    p.add_argument("--downloads_root", default=None)
    p.add_argument("--rtk_root", default=None)
    p.add_argument("--query", action="store_true")
    p.add_argument("--northing", type=float)
    p.add_argument("--easting",  type=float)
    p.add_argument("--radius",   type=float, default=50.0)
    p.add_argument("--copy_to",  type=str,   default=None,
                   help="Directory to copy matching images into (organized by traversal)")
    args = p.parse_args()

    extracted_root = Path(args.extracted_root)

    if args.query:
        if args.northing is None or args.easting is None:
            print("ERROR: --query requires --northing and --easting")
            sys.exit(1)
        copy_to = Path(args.copy_to) if args.copy_to else None
        run_query(extracted_root, args.northing, args.easting, args.radius, copy_to)
        return

    if not args.downloads_root or not args.rtk_root:
        print("ERROR: --downloads_root and --rtk_root required")
        sys.exit(1)

    downloads_root = Path(args.downloads_root)
    rtk_root       = Path(args.rtk_root)

    # All traversal dirs in downloads — skip ones we can't access
    traversal_tags = []
    for d in downloads_root.iterdir():
        try:
            if d.is_dir() and (d / "stereo.timestamps").exists():
                traversal_tags.append(d.name)
        except PermissionError:
            print(f"  [WARN] Permission denied: {d.name} — skipping")
            continue
    traversal_tags = sorted(traversal_tags)

    print(f"Found {len(traversal_tags)} traversals in downloads\n")

    all_rows = []
    skipped_no_pos = []
    for tag in traversal_tags:
        out_dir = extracted_root / tag
        rows = build_traversal_index(
            traversal_tag=tag,
            downloads_root=downloads_root,
            rtk_root=rtk_root,
            output_dir=out_dir,
        )
        if rows:
            all_rows.extend(rows)
        else:
            skipped_no_pos.append(tag)

    combined_path = extracted_root / "all_traversals_index.csv"
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMBINED_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n=== Done ===")
    print(f"Combined index : {combined_path}")
    print(f"Total rows     : {len(all_rows)}")
    print(f"Traversals indexed : {len(traversal_tags) - len(skipped_no_pos)}")
    print(f"Traversals skipped : {len(skipped_no_pos)}")
    if skipped_no_pos:
        for t in skipped_no_pos:
            print(f"  - {t}")
    print(f"\nTo query images near a location:")
    print(f"  python3 build_gps_index.py --query \\")
    print(f"    --extracted_root {extracted_root} \\")
    print(f"    --northing <northing> --easting <easting> --radius 50")
    print(f"\nTo query AND export matching images:")
    print(f"  python3 build_gps_index.py --query \\")
    print(f"    --extracted_root {extracted_root} \\")
    print(f"    --northing <northing> --easting <easting> --radius 50 \\")
    print(f"    --copy_to /path/to/export/dir")


if __name__ == "__main__":
    main()
