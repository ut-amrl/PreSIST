#!/usr/bin/env python3
"""
robotcar_batch.py

Batch-converts multiple Oxford RobotCar traversals into the annotation tool's
expected directory structure:

  <location_root>/<drive>/<camera>/rgb/<timestamp_us>.png

This script handles debayering only (no lens undistortion). To add undistortion
later, pass --sdk_dir and --models_dir and set --undistort.

Sits alongside robotcar_convert.py and reuses its per-image logic directly,
so there is no duplication of debayering or pose-parsing code.

Output structure
----------------
<output_root>/                          ← location_root for annotation tool
  2014-05-06-12-54-54/
    stereo_centre/
      rgb/
        1399381444704913.png            ← debayered BGR PNG, filename = timestamp (us)
      camera_poses.txt                  ← from VO (if --poses)
      camera_intrinsics.txt             ← placeholder (if --poses)
    mono_left/
      rgb/
        1399381444659891.png
  2014-05-14-13-46-12/
    ...

Usage
-----
  # Dry run — preview what would be converted
  python3 robotcar_batch.py --dry-run \\
      --drives 2014-05-06-12-54-54,2014-05-14-13-46-12,2014-06-23-15-14-44

  # Convert 3 test drives, both cameras, 8 workers
  python3 robotcar_batch.py \\
      --drives 2014-05-06-12-54-54,2014-05-14-13-46-12,2014-06-23-15-14-44

  # Convert all drives (auto-discover)
  python3 robotcar_batch.py

  # Stereo centre only
  python3 robotcar_batch.py --cameras stereo_centre \\
      --drives 2014-05-06-12-54-54

  # With VO camera poses written out
  python3 robotcar_batch.py --poses \\
      --drives 2014-05-06-12-54-54

Defaults
--------
  --input-dir   /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/robotCar/downloads
  --output-dir  /robodata/tarunrav/robotcar-converted/oxford_route1
  --cameras     stereo_centre,mono_left
  --workers     8
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT  = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/robotCar/downloads"
DEFAULT_OUTPUT = "/robodata/tarunrav/robotcar-converted/oxford_route1"

# Per-camera Bayer patterns.
# stereo/centre: Bumblebee XB3 → GBRG
# mono_left:     Grasshopper2  → BGGR
BAYER_CODE = {
    "stereo_centre": cv2.COLOR_BayerGB2BGR,
    "mono_left":     cv2.COLOR_BayerBG2BGR,
}

# Map CLI camera name → subdirectory inside each traversal
CAMERA_SOURCE = {
    "stereo_centre": "stereo/centre",
    "mono_left":     "mono_left",
}

# ---------------------------------------------------------------------------
# Per-image worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _convert_one(args: Tuple[Path, Path, int]) -> Tuple[str, bool, str]:
    """Debayer src → dst. Returns (src_name, success, error_msg)."""
    src, dst, bayer_code = args
    try:
        raw = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            return (src.name, False, "cv2.imread returned None")
        bgr = cv2.cvtColor(raw, bayer_code)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), bgr)
        return (src.name, True, "")
    except Exception as e:
        return (src.name, False, str(e))


# ---------------------------------------------------------------------------
# VO → camera_poses.txt  (lifted from robotcar_convert.py)
# ---------------------------------------------------------------------------

def _write_camera_poses(vo_csv: Path, out_path: Path) -> int:
    """
    Parse vo/vo.csv and write camera_poses.txt.
    Returns number of rows written, 0 if vo.csv missing.
    """
    import csv as _csv

    if not vo_csv.exists():
        return 0

    rows = 0
    with open(vo_csv, newline="") as fin, open(out_path, "w") as fout:
        fout.write(
            "# camera_poses.txt\n"
            "# Format: timestamp_us x y z roll pitch yaw\n"
            "# SE(3) relative transform (source->dest) from RobotCar VO\n"
            "# Units: metres and radians\n"
        )
        reader = _csv.reader(fin)
        header = next(reader, None)
        # If header row is non-numeric, skip it; otherwise process it
        if header and header[0].strip().lstrip("-").isdigit():
            try:
                dst_ts = int(header[1])
                x, y, z = float(header[2]), float(header[3]), float(header[4])
                ro, pi, ya = float(header[5]), float(header[6]), float(header[7])
                fout.write(f"{dst_ts} {x:.6f} {y:.6f} {z:.6f} {ro:.6f} {pi:.6f} {ya:.6f}\n")
                rows += 1
            except (ValueError, IndexError):
                pass
        for row in reader:
            if len(row) < 8:
                continue
            try:
                dst_ts = int(row[1])
                x, y, z = float(row[2]), float(row[3]), float(row[4])
                ro, pi, ya = float(row[5]), float(row[6]), float(row[7])
                fout.write(f"{dst_ts} {x:.6f} {y:.6f} {z:.6f} {ro:.6f} {pi:.6f} {ya:.6f}\n")
                rows += 1
            except (ValueError, IndexError):
                pass
    return rows


def _write_intrinsics_placeholder(out_path: Path, w: int, h: int):
    with open(out_path, "w") as f:
        f.write(
            "# camera_intrinsics.txt\n"
            "# Placeholder — populate with SDK camera model when available.\n"
            "# Format: fx fy cx cy image_width image_height\n"
            f"fx 0.0\nfy 0.0\ncx 0.0\ncy 0.0\n"
            f"image_width {w}\nimage_height {h}\n"
        )


# ---------------------------------------------------------------------------
# Single-drive conversion
# ---------------------------------------------------------------------------

def convert_drive(
    drive_name: str,
    input_dir: Path,
    output_dir: Path,
    cameras: List[str],
    workers: int,
    write_poses: bool,
    dry_run: bool,
) -> bool:
    drive_input = input_dir / drive_name
    if not drive_input.exists():
        print(f"  [SKIP] Not found: {drive_input}")
        return False

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Drive: {drive_name}")

    for cam_name in cameras:
        src_subdir = CAMERA_SOURCE[cam_name]
        src_dir = drive_input / src_subdir.replace("/", os.sep)

        if not src_dir.exists():
            print(f"  [{cam_name}] Source not found, skipping: {src_dir}")
            continue

        png_files = sorted(src_dir.glob("*.png"))
        if not png_files:
            print(f"  [{cam_name}] No PNGs found.")
            continue

        dst_rgb = output_dir / drive_name / cam_name / "rgb"
        print(f"  [{cam_name}] {len(png_files)} images  →  {dst_rgb}")

        if dry_run:
            continue

        dst_rgb.mkdir(parents=True, exist_ok=True)

        # Build work list, skip already done
        bayer_code = BAYER_CODE[cam_name]
        work, skipped = [], 0
        for src in png_files:
            dst = dst_rgb / src.name
            if dst.exists():
                skipped += 1
            else:
                work.append((src, dst, bayer_code))

        if skipped:
            print(f"  [{cam_name}] {skipped} already converted, skipping.")
        if not work:
            print(f"  [{cam_name}] Nothing new to do.")
        else:
            failures = []
            done = 0
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_convert_one, w): w for w in work}
                for future in as_completed(futures):
                    name, ok, err = future.result()
                    done += 1
                    if not ok:
                        failures.append((name, err))
                    if done % 200 == 0 or done == len(work):
                        print(f"  [{cam_name}] {done}/{len(work)} ...", end="\r")
            print()

            if failures:
                print(f"  [{cam_name}] {len(failures)} failures:")
                for name, err in failures[:5]:
                    print(f"    {name}: {err}")
                if len(failures) > 5:
                    print(f"    ... and {len(failures) - 5} more.")
            else:
                print(f"  [{cam_name}] Done. {len(work)} images written.")

        # Write camera poses + intrinsics if requested (once per camera per drive)
        if write_poses and not dry_run:
            seq_dir = output_dir / drive_name / cam_name
            vo_csv = drive_input / "vo" / "vo.csv"
            poses_out = seq_dir / "camera_poses.txt"
            if not poses_out.exists():
                n = _write_camera_poses(vo_csv, poses_out)
                if n:
                    print(f"  [{cam_name}] camera_poses.txt: {n} entries.")
                else:
                    print(f"  [{cam_name}] vo.csv not found — camera_poses.txt skipped.")

            intrinsics_out = seq_dir / "camera_intrinsics.txt"
            if not intrinsics_out.exists():
                _write_intrinsics_placeholder(intrinsics_out, w=1280, h=960)
                print(f"  [{cam_name}] camera_intrinsics.txt: placeholder written.")

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch-convert Oxford RobotCar drives to annotation tool format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input-dir",  default=DEFAULT_INPUT,
                   help="Oxford downloads directory.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                   help="Output location_root directory.")
    p.add_argument("--drives", default=None,
                   help="Comma-separated drive names. Omit to convert all.")
    p.add_argument("--cameras", default="stereo_centre,mono_left",
                   help="Comma-separated cameras: stereo_centre, mono_left.")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel worker processes per camera.")
    p.add_argument("--poses", action="store_true",
                   help="Also write camera_poses.txt from VO and intrinsics placeholder.")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview without writing any files.")
    return p.parse_args()


def main():
    args = parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: input-dir not found: {input_dir}")
        sys.exit(1)

    cameras = [c.strip() for c in args.cameras.split(",")]
    for cam in cameras:
        if cam not in CAMERA_SOURCE:
            print(f"ERROR: unknown camera '{cam}'. Choose from: {list(CAMERA_SOURCE)}")
            sys.exit(1)

    if args.drives:
        drives = [d.strip() for d in args.drives.split(",")]
    else:
        drives = sorted([
            d.name for d in input_dir.iterdir()
            if d.is_dir() and d.name[0].isdigit()
        ])
        print(f"Auto-discovered {len(drives)} drives.")

    print(f"Input:    {input_dir}")
    print(f"Output:   {output_dir}")
    print(f"Cameras:  {cameras}")
    print(f"Drives:   {len(drives)}")
    print(f"Workers:  {args.workers}")
    print(f"Poses:    {args.poses}")
    if args.dry_run:
        print("*** DRY RUN — no files will be written ***")

    for drive in drives:
        convert_drive(
            drive_name=drive,
            input_dir=input_dir,
            output_dir=output_dir,
            cameras=cameras,
            workers=args.workers,
            write_poses=args.poses,
            dry_run=args.dry_run,
        )

    print("\nAll done.")
    if not args.dry_run:
        print(f"\nAnnotation tool location_root: {output_dir}")


if __name__ == "__main__":
    main()
