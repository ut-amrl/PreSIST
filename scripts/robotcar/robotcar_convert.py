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
    --sdk_dir       /path/to/robotcar-dataset-sdk \
    [--models_dir   /path/to/camera_models]  \
    [--depth]                                \
    [--max_frames   500]                     \
    [--sequence_tag 2014-05-06-12-54-54]

Dependencies:
  pip install numpy opencv-python pillow

  Also requires the Oxford RobotCar SDK:
    https://github.com/ori-mrg/robotcar-dataset-sdk
  Clone it and pass its path via --sdk_dir.
"""

import argparse
import csv
import os
import sys
import shutil
import struct
from pathlib import Path

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# SDK import helpers
# ---------------------------------------------------------------------------

def _add_sdk(sdk_dir: str):
    """Add the RobotCar SDK python directory to sys.path."""
    sdk_python = Path(sdk_dir) / "python"
    if not sdk_python.exists():
        # Some SDK versions put scripts directly in sdk root
        sdk_python = Path(sdk_dir)
    if str(sdk_python) not in sys.path:
        sys.path.insert(0, str(sdk_python))


def _load_camera_model(sdk_dir: str, models_dir: str, camera: str = "stereo/centre"):
    """
    Load the RobotCar CameraModel for undistortion.
    Returns None if the SDK or model files are unavailable.
    """
    try:
        _add_sdk(sdk_dir)
        from camera_model import CameraModel  # from SDK
        model = CameraModel(models_dir, camera)
        return model
    except Exception as e:
        print(f"[WARN] Could not load CameraModel: {e}")
        print("       Images will be debayered but NOT undistorted.")
        return None


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def debayer_image(raw_path: Path) -> np.ndarray:
    """
    Load a raw Bayer PNG (GBRG for stereo/centre) and return a BGR uint8 array.
    """
    raw = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Could not read image: {raw_path}")
    # Bayer pattern for Bumblebee XB3 stereo = GBRG
    bgr = cv2.cvtColor(raw, cv2.COLOR_BayerGB2BGR)
    return bgr


def process_image(raw_path: Path, camera_model=None) -> np.ndarray:
    """Debayer and optionally undistort."""
    bgr = debayer_image(raw_path)
    if camera_model is not None:
        try:
            bgr = camera_model.undistort(bgr)
        except Exception as e:
            print(f"[WARN] Undistortion failed for {raw_path.name}: {e}")
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

    The pose represents the SE(3) transform from source to destination frame
    expressed as (x, y, z, α, β, γ) Euler parameterisation.
    We emit one line per destination frame using destination_timestamp.
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
        # Detect if first row is a header
        if header and not header[0].strip().lstrip('-').isdigit():
            pass  # already skipped
        else:
            # It was data — process it
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

def write_camera_intrinsics(camera_model, output_path: Path, image_shape: tuple):
    """
    Write camera_intrinsics.txt from SDK CameraModel.
    Format:
      fx fy cx cy [k1 k2 p1 p2 k3]  (pinhole + distortion)
      image_width image_height
    If no model, write a placeholder.
    """
    with open(output_path, "w") as f:
        f.write("# camera_intrinsics.txt\n")
        f.write("# Format: fx fy cx cy  (pinhole intrinsics after undistortion)\n")
        f.write(f"# image_width {image_shape[1]}\n")
        f.write(f"# image_height {image_shape[0]}\n")
        if camera_model is not None and hasattr(camera_model, "focal_length"):
            try:
                fx = camera_model.focal_length[0]
                fy = camera_model.focal_length[1]
                cx = camera_model.principal_point[0]
                cy = camera_model.principal_point[1]
                f.write(f"fx {fx:.6f}\n")
                f.write(f"fy {fy:.6f}\n")
                f.write(f"cx {cx:.6f}\n")
                f.write(f"cy {cy:.6f}\n")
                f.write(f"image_width {image_shape[1]}\n")
                f.write(f"image_height {image_shape[0]}\n")
                print(f"  [camera_intrinsics.txt] fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
                return
            except AttributeError:
                pass
        # Fallback placeholder — user should fill in from SDK model files
        f.write("# WARNING: CameraModel not loaded. Fill in manually from SDK.\n")
        f.write("fx 0.0\nfy 0.0\ncx 0.0\ncy 0.0\n")
        f.write(f"image_width {image_shape[1]}\n")
        f.write(f"image_height {image_shape[0]}\n")
        print("  [camera_intrinsics.txt] Placeholder written — fill in manually.")


# ---------------------------------------------------------------------------
# Depth (LD-MRS projection) — optional
# ---------------------------------------------------------------------------

def load_ldmrs_scan(bin_path: Path) -> np.ndarray:
    """
    Load a LD-MRS 3D scan (.bin).
    Returns Nx3 float64 array of (x, y, z) points in sensor frame.
    """
    with open(bin_path, "rb") as f:
        data = f.read()
    n_doubles = len(data) // 8
    n_points = n_doubles // 3
    pts = np.frombuffer(data, dtype=np.float64).reshape(n_points, 3)
    return pts


def project_lidar_to_depth(
    pts_3d: np.ndarray,
    camera_model,
    image_shape: tuple,
) -> np.ndarray:
    """
    Project LD-MRS points into the stereo/centre image plane.
    Returns a float32 depth image (same H×W as image_shape), 0 = no data,
    negative values = invalid as per spec.
    """
    depth_img = np.zeros(image_shape[:2], dtype=np.float32)

    if camera_model is None:
        print("[WARN] No camera model for LiDAR projection — depth skipped.")
        return depth_img

    # Filter points in front of sensor
    valid = pts_3d[:, 0] > 0.1
    pts_3d = pts_3d[valid]

    try:
        # SDK CameraModel.project expects Nx3 in camera frame
        uv, depths = camera_model.project(pts_3d, image_shape)
        for i in range(uv.shape[1]):
            u, v = int(uv[0, i]), int(uv[1, i])
            if 0 <= v < image_shape[0] and 0 <= u < image_shape[1]:
                d = depths[i]
                # Keep closest point (smallest positive depth)
                existing = depth_img[v, u]
                if existing == 0 or d < existing:
                    depth_img[v, u] = float(d)
    except Exception as e:
        print(f"[WARN] LiDAR projection error: {e}")

    return depth_img


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_traversal(
    traversal_dir: Path,
    output_root: Path,
    sequence_tag: str,
    sdk_dir: str | None,
    models_dir: str | None,
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
    ldmrs_dir = traversal_dir / "ldmrs"

    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not timestamps_file.exists():
        raise FileNotFoundError(f"Timestamps file not found: {timestamps_file}")

    # Output directories
    seq_root = output_root / sequence_tag
    rgb_out = seq_root / "rgb"
    rgb_out.mkdir(parents=True, exist_ok=True)
    if include_depth:
        depth_out = seq_root / "depth"
        depth_out.mkdir(parents=True, exist_ok=True)

    # Load camera model (optional)
    camera_model = None
    if sdk_dir and models_dir:
        camera_model = _load_camera_model(sdk_dir, models_dir, camera)

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
        # Fall back: just use files sorted by name (filename = timestamp)
        print("  [WARN] No timestamp matches — falling back to filename-as-timestamp")
        matched = [(int(p.stem), p) for p in sorted(available_images.values())]

    print(f"  Matched {len(matched)} frames")

    if max_frames and len(matched) > max_frames:
        print(f"  Limiting to first {max_frames} frames (--max_frames)")
        matched = matched[:max_frames]

    # Process images
    print(f"  Processing {len(matched)} images...")
    first_image_shape = None
    for i, (ts_us, img_path) in enumerate(matched):
        ts_ms = us_to_ms(ts_us)
        out_path = rgb_out / f"{ts_ms}.png"

        if out_path.exists():
            if i == 0:
                tmp = cv2.imread(str(out_path))
                if tmp is not None:
                    first_image_shape = tmp.shape
            continue  # skip already converted

        try:
            bgr = process_image(img_path, camera_model)
            cv2.imwrite(str(out_path), bgr)
            if first_image_shape is None:
                first_image_shape = bgr.shape
        except Exception as e:
            print(f"  [WARN] Skipping {img_path.name}: {e}")
            continue

        if (i + 1) % 100 == 0 or (i + 1) == len(matched):
            print(f"    {i+1}/{len(matched)} frames converted...", end="\r")

    print(f"\n  [rgb/] Wrote {len(matched)} images to {rgb_out}")

    # Camera poses from VO
    poses_out = seq_root / "camera_poses.txt"
    convert_vo_to_camera_poses(vo_csv, poses_out)

    # Camera intrinsics
    intrinsics_out = seq_root / "camera_intrinsics.txt"
    write_camera_intrinsics(camera_model, intrinsics_out, first_image_shape or (960, 1280, 3))

    # Depth (optional)
    if include_depth and ldmrs_dir.exists() and camera_model is not None:
        print("  Generating depth maps from LD-MRS...")
        ldmrs_timestamps_file = traversal_dir / "ldmrs.timestamps"
        if ldmrs_timestamps_file.exists():
            ldmrs_ts = load_timestamps(ldmrs_timestamps_file)
            ldmrs_bins = {p.stem: p for p in sorted(ldmrs_dir.glob("*.bin"))}

            img_ts_ms_list = [us_to_ms(ts_us) for ts_us, _ in matched]
            h = first_image_shape[0] if first_image_shape else 960
            w = first_image_shape[1] if first_image_shape else 1280

            depth_written = 0
            for img_ts_ms in img_ts_ms_list:
                img_ts_us = img_ts_ms * 1000
                # Find closest LiDAR scan to this image timestamp
                closest = min(ldmrs_ts, key=lambda t: abs(t[0] - img_ts_us), default=None)
                if closest is None:
                    continue
                lidar_stem = str(closest[0])
                if lidar_stem not in ldmrs_bins:
                    continue
                try:
                    pts = load_ldmrs_scan(ldmrs_bins[lidar_stem])
                    depth = project_lidar_to_depth(pts, camera_model, (h, w))
                    depth_path = depth_out / f"{img_ts_ms}.npy"
                    np.save(str(depth_path), depth)
                    depth_written += 1
                except Exception as e:
                    pass

            print(f"  [depth/] Wrote {depth_written} depth maps")
        else:
            print("  [WARN] ldmrs.timestamps not found, skipping depth")
    elif include_depth:
        if not ldmrs_dir.exists():
            print("  [WARN] ldmrs/ directory not found — skipping depth generation")
        elif camera_model is None:
            print("  [WARN] No camera model — skipping depth generation (needed for projection)")

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
        "--sdk_dir", default=None,
        help="Path to robotcar-dataset-sdk clone (for CameraModel / undistortion)"
    )
    p.add_argument(
        "--models_dir", default=None,
        help="Path to camera model files (.bin undistortion LUTs). "
             "Download from robotcar-dataset.robots.ox.ac.uk/downloads"
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
        help="Generate sparse depth maps by projecting LD-MRS LiDAR into the camera."
             " Requires --sdk_dir and --models_dir."
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
        sdk_dir=args.sdk_dir,
        models_dir=args.models_dir,
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
