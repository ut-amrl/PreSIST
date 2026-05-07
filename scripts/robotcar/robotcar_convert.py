#!/usr/bin/env python3
"""
robotcar_convert.py

Converts Oxford RobotCar dataset traversals into the annotation tool's
expected directory structure:

  <location_root>/<sequence_tag>/rgb/<timestamp_ms>.png
  <location_root>/<sequence_tag>/camera_poses.txt
  <location_root>/<sequence_tag>/camera_intrinsics.txt
  <location_root>/<sequence_tag>/depth/  (optional, if --depth flag set)

Usage:
  python robotcar_convert.py \
    --traversal_dir /path/to/2014-05-06-12-54-54 \
    --output_root   /path/to/output/oxford_robotcar \
    [--depth]                                \
    [--max_frames   500]                     \
    [--sequence_tag 2014-05-06-12-54-54]

Dependencies:
  pip install numpy opencv-python pillow colour-demosaicing

  Also requires robotcar_image.py and camera_model.py from:
    /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/robotCar/code/
  And camera model files in:
    /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/robotCar/models/
"""

import argparse
import csv
import os
import sys
import struct
from pathlib import Path

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Load robotcar_image from the existing SDK code directory
# ---------------------------------------------------------------------------

ROBOTCAR_CODE_DIR = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/robotCar/code"
if ROBOTCAR_CODE_DIR not in sys.path:
    sys.path.insert(0, ROBOTCAR_CODE_DIR)

try:
    from robotcar_image import convert_image as _rc_convert_image
    HAVE_ROBOTCAR_IMAGE = True
    print("[INFO] robotcar_image loaded — correct demosaicing + undistortion enabled.")
except ImportError as e:
    HAVE_ROBOTCAR_IMAGE = False
    print(f"[WARN] Could not import robotcar_image: {e}")
    print("       Falling back to raw debayer only (colours may be wrong).")


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def process_image(img_path: Path, out_path: Path) -> np.ndarray:
    """
    Convert a raw Bayer PNG to an undistorted RGB PNG.

    If robotcar_image is available, use convert_image() which handles
    correct demosaicing and LUT-based undistortion.

    Falls back to naive BayerGB->BGR if unavailable.

    Returns the resulting BGR image as a numpy array (for shape detection).
    """
    if HAVE_ROBOTCAR_IMAGE:
        # convert_image writes directly to out_path
        _rc_convert_image(str(img_path), str(out_path))
        # Read back to get shape
        bgr = cv2.imread(str(out_path))
        if bgr is None:
            raise RuntimeError(f"convert_image wrote nothing to {out_path}")
        return bgr
    else:
        # Fallback — wrong colours but better than nothing
        raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        bgr = cv2.cvtColor(raw, cv2.COLOR_BayerGB2BGR)
        cv2.imwrite(str(out_path), bgr)
        return bgr


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def load_timestamps(timestamps_path: Path) -> list[tuple[int, int]]:
    """
    Parse a RobotCar .timestamps file.
    Each line: <timestamp_us>  <chunk_id>
    Returns list of (timestamp_us, chunk_id).
    """
    entries = []
    with open(timestamps_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 1:
                ts_us = int(parts[0])
                chunk = int(parts[1]) if len(parts) > 1 else 0
                entries.append((ts_us, chunk))
    return entries


def us_to_ms(ts_us: int) -> int:
    """Microseconds → milliseconds (truncating)."""
    return ts_us // 1000


# ---------------------------------------------------------------------------
# VO / camera poses
# ---------------------------------------------------------------------------

def convert_vo_to_camera_poses(vo_csv_path: Path, output_path: Path):
    """
    Convert vo/vo.csv to camera_poses.txt.

    RobotCar VO format (vo.csv):
      source_timestamp, destination_timestamp, x, y, z, roll, pitch, yaw

    Output format (one line per frame):
      <timestamp_ms> <x> <y> <z> <roll> <pitch> <yaw>
    """
    if not vo_csv_path.exists():
        print(f"[WARN] vo.csv not found at {vo_csv_path}, skipping camera_poses.txt")
        return

    rows_written = 0
    with open(vo_csv_path, newline="") as fin, open(output_path, "w") as fout:
        fout.write(
            "# camera_poses.txt\n"
            "# Format: timestamp_ms x y z roll pitch yaw\n"
            "# Pose is SE(3) relative transform (source->dest) from RobotCar VO\n"
            "# Units: metres and radians\n"
        )
        reader = csv.reader(fin)
        header = next(reader, None)  # skip header row if present
        if header and not header[0].strip().lstrip('-').isdigit():
            pass  # already skipped
        else:
            if header:
                src_ts, dst_ts = int(header[0]), int(header[1])
                x, y, z = float(header[2]), float(header[3]), float(header[4])
                roll, pitch, yaw = float(header[5]), float(header[6]), float(header[7])
                dst_ms = us_to_ms(dst_ts)
                fout.write(f"{dst_ms} {x:.6f} {y:.6f} {z:.6f} {roll:.6f} {pitch:.6f} {yaw:.6f}\n")
                rows_written += 1

        for row in reader:
            if len(row) < 8:
                continue
            try:
                dst_ts = int(row[1])
                x, y, z = float(row[2]), float(row[3]), float(row[4])
                roll, pitch, yaw = float(row[5]), float(row[6]), float(row[7])
                dst_ms = us_to_ms(dst_ts)
                fout.write(f"{dst_ms} {x:.6f} {y:.6f} {z:.6f} {roll:.6f} {pitch:.6f} {yaw:.6f}\n")
                rows_written += 1
            except (ValueError, IndexError) as e:
                print(f"[WARN] Skipping malformed VO row: {row} ({e})")

    print(f"  [camera_poses.txt] Wrote {rows_written} pose entries.")


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

def write_camera_intrinsics(output_path: Path, image_shape: tuple):
    """
    Write camera_intrinsics.txt.
    stereo/centre uses stereo_narrow_left model:
      fx=fy=964.828979, cx=643.788025, cy=484.407990
    """
    fx, fy, cx, cy = 964.828979, 964.828979, 643.788025, 484.407990
    with open(output_path, "w") as f:
        f.write("# camera_intrinsics.txt\n")
        f.write("# Format: fx fy cx cy  (stereo/centre = stereo_narrow_left model)\n")
        f.write(f"fx {fx:.6f}\n")
        f.write(f"fy {fy:.6f}\n")
        f.write(f"cx {cx:.6f}\n")
        f.write(f"cy {cy:.6f}\n")
        f.write(f"image_width {image_shape[1]}\n")
        f.write(f"image_height {image_shape[0]}\n")
    print(f"  [camera_intrinsics.txt] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")


# ---------------------------------------------------------------------------
# Depth (LD-MRS projection) — optional
# ---------------------------------------------------------------------------

def load_ldmrs_scan(bin_path: Path) -> np.ndarray:
    """Load a LD-MRS 3D scan (.bin). Returns Nx3 float64 array."""
    with open(bin_path, "rb") as f:
        data = f.read()
    n_doubles = len(data) // 8
    n_points = n_doubles // 3
    pts = np.frombuffer(data, dtype=np.float64).reshape(n_points, 3)
    return pts


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_traversal(
    traversal_dir: Path,
    output_root: Path,
    sequence_tag: str,
    include_depth: bool,
    max_frames: int | None,
    camera: str = "stereo/centre",
):
    print(f"\n=== Converting traversal: {traversal_dir.name} ===")
    print(f"    Sequence tag : {sequence_tag}")
    print(f"    Camera       : {camera}")
    print(f"    Output root  : {output_root}")

    # Paths within traversal
    images_dir = traversal_dir / camera.replace("/", os.sep)
    timestamps_file = traversal_dir / "stereo.timestamps"
    vo_csv = traversal_dir / "vo" / "vo.csv"

    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not timestamps_file.exists():
        raise FileNotFoundError(f"Timestamps file not found: {timestamps_file}")

    # Output directories
    seq_root = output_root / sequence_tag
    rgb_out = seq_root / "rgb"
    rgb_out.mkdir(parents=True, exist_ok=True)

    # Load timestamps
    print("  Loading timestamps...")
    all_timestamps = load_timestamps(timestamps_file)
    print(f"  Found {len(all_timestamps)} timestamps in stereo.timestamps")

    # Collect actual image files present on disk
    available_images = {p.stem: p for p in sorted(images_dir.glob("*.png"))}
    print(f"  Found {len(available_images)} PNG files in {images_dir}")

    # Match timestamps to available images
    matched = []
    for ts_us, chunk in all_timestamps:
        stem = str(ts_us)
        if stem in available_images:
            matched.append((ts_us, available_images[stem]))

    if not matched:
        print("  [WARN] No timestamp matches — falling back to filename-as-timestamp")
        matched = [(int(p.stem), p) for p in sorted(available_images.values())]

    print(f"  Matched {len(matched)} frames")

    if max_frames and len(matched) > max_frames:
        print(f"  Limiting to first {max_frames} frames (--max_frames)")
        matched = matched[:max_frames]

    # Process images
    print(f"  Processing {len(matched)} images...")
    first_image_shape = None
    converted = 0
    skipped = 0
    for i, (ts_us, img_path) in enumerate(matched):
        ts_ms = us_to_ms(ts_us)
        out_path = rgb_out / f"{ts_ms}.png"

        if out_path.exists():
            if first_image_shape is None:
                tmp = cv2.imread(str(out_path))
                if tmp is not None:
                    first_image_shape = tmp.shape
            skipped += 1
            continue

        try:
            bgr = process_image(img_path, out_path)
            if first_image_shape is None:
                first_image_shape = bgr.shape
            converted += 1
        except Exception as e:
            print(f"  [WARN] Skipping {img_path.name}: {e}")
            continue

        if (i + 1) % 100 == 0 or (i + 1) == len(matched):
            print(f"    {i+1}/{len(matched)} frames processed...", end="\r")

    print(f"\n  [rgb/] Converted {converted} new, skipped {skipped} existing → {rgb_out}")

    # Camera poses from VO
    poses_out = seq_root / "camera_poses.txt"
    convert_vo_to_camera_poses(vo_csv, poses_out)

    # Camera intrinsics (hardcoded from stereo_narrow_left.txt)
    intrinsics_out = seq_root / "camera_intrinsics.txt"
    write_camera_intrinsics(intrinsics_out, first_image_shape or (960, 1280, 3))

    print(f"\n  Done. Output at: {seq_root}")
    return seq_root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert Oxford RobotCar traversal to annotation tool format."
    )
    p.add_argument(
        "--traversal_dir", required=True,
        help="Path to a single RobotCar traversal directory (e.g. .../2014-05-06-12-54-54)"
    )
    p.add_argument(
        "--output_root", required=True,
        help="Root output directory. Sequence will be placed at <output_root>/<sequence_tag>/"
    )
    p.add_argument(
        "--camera", default="stereo/centre",
        help="Camera subdirectory within traversal (default: stereo/centre)"
    )
    p.add_argument(
        "--sequence_tag", default=None,
        help="Sequence tag to use as output directory name. "
             "Defaults to the traversal directory name."
    )
    p.add_argument(
        "--depth", action="store_true",
        help="Reserved for future depth map generation."
    )
    p.add_argument(
        "--max_frames", type=int, default=None,
        help="Limit number of frames to convert (useful for testing)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    traversal_dir = Path(args.traversal_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    sequence_tag = args.sequence_tag or traversal_dir.name

    if not traversal_dir.exists():
        print(f"ERROR: traversal_dir does not exist: {traversal_dir}")
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    seq_root = convert_traversal(
        traversal_dir=traversal_dir,
        output_root=output_root,
        sequence_tag=sequence_tag,
        include_depth=args.depth,
        max_frames=args.max_frames,
        camera=args.camera,
    )

    print("\n=== Conversion complete ===")
    print(f"Output structure:")
    for item in sorted(seq_root.rglob("*"))[:30]:
        rel = item.relative_to(output_root)
        indent = "  " * (len(rel.parts) - 1)
        print(f"  {indent}{item.name}{'/' if item.is_dir() else ''}")
    total_rgb = len(list((seq_root / "rgb").glob("*.png")))
    print(f"\n  Total RGB frames: {total_rgb}")
    print(f"\nTo annotate this sequence, point your annotation tool at:")
    print(f"  location_root : {output_root}")
    print(f"  sequence_tag  : {sequence_tag}")


if __name__ == "__main__":
    main()
