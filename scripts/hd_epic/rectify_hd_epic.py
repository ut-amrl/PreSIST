#!/usr/bin/env python3
"""
rectify_hd_epic.py

Rectifies HD-EPIC fisheye frames using calibration from device_calibration.json.
Implements Aria FISHEYE624 model directly — no VRS or projectaria_tools needed.

Usage:
  # Single participant
  python3 rectify_hd_epic.py --location P01

  # All participants
  python3 rectify_hd_epic.py

  # Force redo already-rectified frames
  python3 rectify_hd_epic.py --location P01 --re_rectify

Defaults:
  --frames_root  /robodata/tarunrav/hd-epic-processed
  --calib_root   /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/intermediate_data
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_FRAMES_ROOT = "/robodata/tarunrav/hd-epic-processed"
DEFAULT_CALIB_ROOT  = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/intermediate_data"


# ---------------------------------------------------------------------------
# Calibration loading
# ---------------------------------------------------------------------------

def load_rgb_calibration(calib_json_path: str) -> np.ndarray:
    """Load and return the 15-element FISHEYE624 projection params array."""
    with open(calib_json_path) as f:
        data = json.load(f)
    rgb = data["cameras"]["camera-rgb"]
    assert "FISHEYE624" in rgb["model_name"], f"Unexpected model: {rgb['model_name']}"
    p = rgb["projection_params"]
    assert len(p) == 15, f"Expected 15 projection params, got {len(p)}"
    return np.array(p)


# ---------------------------------------------------------------------------
# FISHEYE624 forward projection: 3D ray -> fisheye pixel
# ---------------------------------------------------------------------------

def fisheye624_project(params: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """
    Project (N,3) rays to (N,2) pixel coords using Aria FISHEYE624 model.

    params layout: [f, cx, cy, k0..k5, p0..p5]

    Model (validated to <0.02px vs projectaria_tools ground truth):
      1. r_u    = sqrt(x^2 + y^2)
      2. theta  = atan2(r_u, z)
      3. theta_d = theta * (1 + k0*t2 + k1*t2^2 + ... + k5*t2^6)   where t2=theta^2
      4. mx,my  = (x,y)/r_u * theta_d       (distorted direction)
      5. tangential correction on mx,my
      6. u = f*mx + cx,  v = f*my + cy
    """
    f  = params[0]
    cx = params[1]
    cy = params[2]
    k  = params[3:9]   # k0..k5 radial
    p  = params[9:15]  # p0..p5 tangential

    x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]

    r_u   = np.sqrt(x**2 + y**2)
    theta = np.arctan2(r_u, z)
    t2    = theta * theta

    theta_d = theta * (1 + k[0]*t2     + k[1]*t2**2 + k[2]*t2**3
                         + k[3]*t2**4  + k[4]*t2**5 + k[5]*t2**6)

    safe_r = np.where(r_u > 1e-8, r_u, 1.0)
    mx = np.where(r_u > 1e-8, (x / safe_r) * theta_d, 0.0)
    my = np.where(r_u > 1e-8, (y / safe_r) * theta_d, 0.0)

    r2 = mx**2 + my**2
    r4 = r2**2

    dx = (p[0]*(r2 + 2*mx**2) + 2*p[1]*mx*my
          + r2 *(p[2]*mx + p[3]*my)
          + r4 *(p[4]*mx + p[5]*my))
    dy = (2*p[0]*mx*my + p[1]*(r2 + 2*my**2)
          + r2 *(p[2]*my + p[3]*mx)
          + r4 *(p[4]*my + p[5]*mx))

    u = f * (mx + dx) + cx
    v = f * (my + dy) + cy

    return np.stack([u, v], axis=-1)


# ---------------------------------------------------------------------------
# Build rectification maps
# ---------------------------------------------------------------------------

def build_rectify_maps(params: np.ndarray, out_w: int, out_h: int):
    """
    Build cv2.remap source maps (map1, map2).

    For each output pixel (pinhole model centered at image center),
    compute the ray then project through fisheye to get the source pixel.
    Uses fisheye focal length f so FOV is preserved.
    """
    f      = params[0]
    cx_out = out_w / 2.0
    cy_out = out_h / 2.0

    xs, ys = np.meshgrid(np.arange(out_w, dtype=np.float64),
                         np.arange(out_h, dtype=np.float64))
    rays = np.stack([
        (xs.ravel() - cx_out) / f,
        (ys.ravel() - cy_out) / f,
        np.ones(out_w * out_h, dtype=np.float64),
    ], axis=-1)

    src = fisheye624_project(params, rays)

    map1 = src[:, 0].reshape(out_h, out_w).astype(np.float32)
    map2 = src[:, 1].reshape(out_h, out_w).astype(np.float32)

    # Mark out-of-bounds as -1 (cv2.remap fills with black)
    map1 = np.where(np.isnan(map1), -1.0, map1)
    map2 = np.where(np.isnan(map2), -1.0, map2)

    return map1, map2


# ---------------------------------------------------------------------------
# Crop computation
# ---------------------------------------------------------------------------

def compute_crop(map1: np.ndarray, map2: np.ndarray,
                 out_w: int, out_h: int, src_w: int, src_h: int):
    """Binary search for largest centered square with all valid source pixels."""
    valid = (map1 >= 0) & (map1 < src_w) & (map2 >= 0) & (map2 < src_h)

    cx = out_w // 2
    cy = out_h // 2

    lo, hi, best = 0, min(cx, cy), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if valid[cy-mid:cy+mid, cx-mid:cx+mid].all():
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    x0, y0 = cx - best, cy - best
    x1, y1 = cx + best, cy + best
    print(f"  Crop: ({x0},{y0})→({x1},{y1})  =  {x1-x0}x{y1-y0} px")
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Per-sequence
# ---------------------------------------------------------------------------

def rectify_sequence(seq_dir: Path, map1: np.ndarray, map2: np.ndarray,
                     crop: tuple, re_rectify: bool):
    frames_dir = seq_dir / "frames"
    rect_dir   = seq_dir / "frames_rectified"

    if not frames_dir.exists():
        print(f"    [SKIP] No frames/ in {seq_dir.name}")
        return

    if re_rectify and rect_dir.exists():
        shutil.rmtree(rect_dir)
    rect_dir.mkdir(exist_ok=True)

    x0, y0, x1, y1 = crop
    frame_files = sorted(frames_dir.glob("*.jpg"))
    converted = 0

    for src in frame_files:
        dst = rect_dir / src.name
        if dst.exists():
            continue
        img = cv2.imread(str(src))
        if img is None:
            print(f"    Warning: could not read {src}")
            continue
        rectified = cv2.remap(img, map1, map2, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(str(dst), rectified[y0:y1, x0:x1])
        converted += 1

    skipped = len(frame_files) - converted
    print(f"    → {converted} rectified, {skipped} already done  [{seq_dir.name}]")


# ---------------------------------------------------------------------------
# Per-participant
# ---------------------------------------------------------------------------

def get_calib_json(calib_root: Path, participant: str) -> Path | None:
    p_dir = calib_root / participant
    if not p_dir.exists():
        return None
    for seq_dir in sorted(p_dir.iterdir()):
        calib_file = seq_dir / "device_calibration.json"
        if calib_file.exists():
            return calib_file
    return None


def rectify_participant(participant: str, frames_root: Path, calib_root: Path,
                        out_w: int, out_h: int, re_rectify: bool):
    print(f"\n=== {participant} ===")

    calib_file = get_calib_json(calib_root, participant)
    if calib_file is None:
        print(f"  [SKIP] No device_calibration.json found for {participant}")
        return

    params = load_rgb_calibration(str(calib_file))
    print(f"  f={params[0]:.2f}  cx={params[1]:.2f}  cy={params[2]:.2f}")

    print(f"  Building rectification maps ({out_w}x{out_h})...")
    map1, map2 = build_rectify_maps(params, out_w, out_h)

    print(f"  Computing crop bounds...")
    crop = compute_crop(map1, map2, out_w, out_h, out_w, out_h)

    p_dir = frames_root / participant
    if not p_dir.exists():
        print(f"  [SKIP] {p_dir} not found in frames_root")
        return

    sequences = sorted([d for d in p_dir.iterdir() if d.is_dir()])
    print(f"  Processing {len(sequences)} sequences...")
    for seq_dir in sequences:
        rectify_sequence(seq_dir, map1, map2, crop, re_rectify)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rectify HD-EPIC fisheye frames using device_calibration.json.")
    parser.add_argument("--frames_root", default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--calib_root",  default=DEFAULT_CALIB_ROOT)
    parser.add_argument("--location", default=None,
                        help="Single participant e.g. P01. Omit for all.")
    parser.add_argument("--output_width",  type=int, default=1408)
    parser.add_argument("--output_height", type=int, default=1408)
    parser.add_argument("--re_rectify", action="store_true",
                        help="Delete and redo existing frames_rectified/ contents.")
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    calib_root  = Path(args.calib_root)

    if not frames_root.exists():
        print(f"ERROR: frames_root not found: {frames_root}")
        sys.exit(1)
    if not calib_root.exists():
        print(f"ERROR: calib_root not found: {calib_root}")
        sys.exit(1)

    if args.location:
        participants = [args.location]
    else:
        participants = sorted([
            d.name for d in frames_root.iterdir()
            if d.is_dir() and d.name.startswith("P")
        ])
        print(f"Auto-discovered {len(participants)} participants: {participants}")

    for participant in participants:
        rectify_participant(participant, frames_root, calib_root,
                            args.output_width, args.output_height, args.re_rectify)

    print("\nAll done.")


if __name__ == "__main__":
    main()
