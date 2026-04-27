#!/usr/bin/env python3
"""
rectify_masks_hd_epic.py

Rectifies HD-EPIC fisheye segmentation masks using the same FISHEYE624 calibration
as rectify_hd_epic.py. Uses NEAREST-NEIGHBOR interpolation to preserve color-encoded
object IDs exactly (no blending at boundaries).

Folder structure:
  Input  masks:  MASKS_ROOT/<PXX-YYYYMMDD-HHMMSS>/*.png
  Output masks:  OUTPUT_ROOT/rectified_masks/<PXX>/<PXX-YYYYMMDD-HHMMSS>/*.png

The crop applied is identical to what rectify_hd_epic.py computes, so rectified masks
align pixel-perfectly with the rectified RGB frames.

Usage:
  # Single participant
  python3 rectify_masks_hd_epic.py --participant P01

  # All participants
  python3 rectify_masks_hd_epic.py

  # Force redo already-rectified masks
  python3 rectify_masks_hd_epic.py --participant P01 --re_rectify

Defaults:
  --masks_root   /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/hd_epic_association_masks/masks
  --calib_root   /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/intermediate_data
  --output_root  /robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data
  --frames_root  /robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/hd-epic

The crop is derived from the actual rectified RGB frame size (not recomputed from
calibration) so masks align pixel-perfectly with the already-rectified frames.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MASKS_ROOT  = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/hd_epic_association_masks/masks"
DEFAULT_CALIB_ROOT  = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/intermediate_data"
DEFAULT_OUTPUT_ROOT = "/robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data"
DEFAULT_FRAMES_ROOT = "/robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/hd-epic"


# ---------------------------------------------------------------------------
# Calibration loading  (identical to rectify_hd_epic.py)
# ---------------------------------------------------------------------------

def load_rgb_calibration(calib_json_path: str) -> np.ndarray:
    with open(calib_json_path) as f:
        data = json.load(f)
    rgb = data["cameras"]["camera-rgb"]
    assert "FISHEYE624" in rgb["model_name"], f"Unexpected model: {rgb['model_name']}"
    p = rgb["projection_params"]
    assert len(p) == 15, f"Expected 15 projection params, got {len(p)}"
    return np.array(p)


# ---------------------------------------------------------------------------
# FISHEYE624 forward projection  (identical to rectify_hd_epic.py)
# ---------------------------------------------------------------------------

def fisheye624_project(params: np.ndarray, rays: np.ndarray) -> np.ndarray:
    f  = params[0]
    cx = params[1]
    cy = params[2]
    k  = params[3:9]
    p  = params[9:15]

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
    """Build cv2.remap source maps for a pinhole output of size out_w x out_h."""
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

    map1 = np.where(np.isnan(map1), -1.0, map1)
    map2 = np.where(np.isnan(map2), -1.0, map2)

    return map1, map2


# ---------------------------------------------------------------------------
# Detect mask source dimensions from the first PNG in a sequence folder
# ---------------------------------------------------------------------------

def detect_mask_size(seq_mask_dir: Path):
    """Return (w, h) of the first mask found, or None if folder is empty."""
    for f in seq_mask_dir.glob("*.png"):
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    return None


# ---------------------------------------------------------------------------
# Look up the actual rectified RGB frame size for a participant
# ---------------------------------------------------------------------------

def get_rgb_frame_size(frames_root: Path, participant: str):
    """
    Read one rectified RGB frame to get the actual cropped output size.
    Returns (w, h) or None if no frames found.
    """
    p_dir = frames_root / participant
    if not p_dir.exists():
        return None
    for seq_dir in sorted(p_dir.iterdir()):
        rgb_dir = seq_dir / "rgb"
        if not rgb_dir.exists():
            continue
        for f in rgb_dir.glob("*.png"):
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                return w, h
    return None


def crop_from_frame_size(mask_w: int, mask_h: int, frame_w: int, frame_h: int):
    """
    Derive the centered crop (x0,y0,x1,y1) that maps mask_w x mask_h
    down to frame_w x frame_h — matching what rectify_hd_epic.py applied.
    """
    cx, cy = mask_w // 2, mask_h // 2
    x0 = cx - frame_w // 2
    y0 = cy - frame_h // 2
    x1 = cx + frame_w // 2
    y1 = cy + frame_h // 2
    return x0, y0, x1, y1


# ---------------------------------------------------------------------------
# Rectify one sequence of masks
# ---------------------------------------------------------------------------

def rectify_mask_sequence(seq_mask_dir: Path, out_seq_dir: Path,
                           map1: np.ndarray, map2: np.ndarray,
                           crop: tuple, re_rectify: bool):
    if re_rectify and out_seq_dir.exists():
        shutil.rmtree(out_seq_dir)
    out_seq_dir.mkdir(parents=True, exist_ok=True)

    x0, y0, x1, y1 = crop
    mask_files = sorted(seq_mask_dir.glob("*.png"))
    converted = 0

    for src in mask_files:
        dst = out_seq_dir / src.name
        if dst.exists():
            continue

        # Read as-is (preserves all channels and bit depth)
        mask = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"    Warning: could not read {src.name}")
            continue

        # INTER_NEAREST is mandatory — prevents color blending that would
        # corrupt object-ID encodings at region boundaries
        rectified = cv2.remap(mask, map1, map2, cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        cv2.imwrite(str(dst), rectified[y0:y1, x0:x1])
        converted += 1

    skipped = len(mask_files) - converted
    print(f"    → {converted} rectified, {skipped} already done  [{seq_mask_dir.name}]")


# ---------------------------------------------------------------------------
# Per-participant
# ---------------------------------------------------------------------------

def get_calib_json(calib_root: Path, participant: str):
    p_dir = calib_root / participant
    if not p_dir.exists():
        return None
    for seq_dir in sorted(p_dir.iterdir()):
        calib_file = seq_dir / "device_calibration.json"
        if calib_file.exists():
            return calib_file
    return None


def rectify_participant_masks(participant: str,
                               masks_root: Path,
                               calib_root: Path,
                               output_root: Path,
                               frames_root: Path,
                               re_rectify: bool):
    print(f"\n=== {participant} ===")

    # Find all sequence folders for this participant in the masks root
    # Mask folders are named like P01-20240202-110250 (no sub-participant dir)
    seq_dirs = sorted([
        d for d in masks_root.iterdir()
        if d.is_dir() and d.name.startswith(participant + "-")
    ])

    if not seq_dirs:
        print(f"  [SKIP] No mask sequence folders found for {participant}")
        return

    print(f"  Found {len(seq_dirs)} sequence folder(s)")

    # Load calibration
    calib_file = get_calib_json(calib_root, participant)
    if calib_file is None:
        print(f"  [SKIP] No device_calibration.json found for {participant}")
        return

    params = load_rgb_calibration(str(calib_file))
    print(f"  f={params[0]:.2f}  cx={params[1]:.2f}  cy={params[2]:.2f}")

    # Detect mask size from the first available sequence
    mask_size = None
    for seq_dir in seq_dirs:
        mask_size = detect_mask_size(seq_dir)
        if mask_size is not None:
            break

    if mask_size is None:
        print(f"  [SKIP] Could not detect mask dimensions (no .png files found)")
        return

    src_w, src_h = mask_size
    print(f"  Detected mask size: {src_w}x{src_h}")

    # Look up actual rectified RGB frame size — use this to derive crop
    # so masks align pixel-perfectly with the already-rectified frames.
    frame_size = get_rgb_frame_size(frames_root, participant)
    if frame_size is None:
        print(f"  [SKIP] No rectified RGB frames found under {frames_root / participant}")
        print(f"         Run rectify_hd_epic.py first, then re-run this script.")
        return
    frame_w, frame_h = frame_size
    print(f"  Rectified RGB frame size: {frame_w}x{frame_h}")

    crop = crop_from_frame_size(src_w, src_h, frame_w, frame_h)
    x0, y0, x1, y1 = crop
    print(f"  Derived crop: ({x0},{y0})→({x1},{y1})  =  {x1-x0}x{y1-y0} px")

    # Build remap maps at mask resolution
    print(f"  Building rectification maps ({src_w}x{src_h})...")
    map1, map2 = build_rectify_maps(params, src_w, src_h)

    # Output root: .../extracted_data/rectified_masks/P01/
    out_participant_dir = output_root / "rectified_masks" / participant

    for seq_dir in seq_dirs:
        out_seq_dir = out_participant_dir / seq_dir.name
        rectify_mask_sequence(seq_dir, out_seq_dir, map1, map2, crop, re_rectify)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rectify HD-EPIC fisheye segmentation masks.")
    parser.add_argument("--masks_root",  default=DEFAULT_MASKS_ROOT,
                        help="Root folder containing per-sequence mask subfolders")
    parser.add_argument("--calib_root",  default=DEFAULT_CALIB_ROOT,
                        help="Root folder containing per-participant calibration data")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT,
                        help="Root under which rectified_masks/ will be created")
    parser.add_argument("--frames_root", default=DEFAULT_FRAMES_ROOT,
                        help="Root of already-rectified RGB frames (hd-epic/PXX/seq/rgb/)")
    parser.add_argument("--participant", default=None,
                        help="Single participant e.g. P01. Omit to process all.")
    parser.add_argument("--re_rectify", action="store_true",
                        help="Delete and redo existing output contents.")
    args = parser.parse_args()

    masks_root  = Path(args.masks_root)
    calib_root  = Path(args.calib_root)
    output_root = Path(args.output_root)
    frames_root = Path(args.frames_root)

    if not masks_root.exists():
        print(f"ERROR: masks_root not found: {masks_root}")
        sys.exit(1)
    if not calib_root.exists():
        print(f"ERROR: calib_root not found: {calib_root}")
        sys.exit(1)
    if not frames_root.exists():
        print(f"ERROR: frames_root not found: {frames_root}")
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    if args.participant:
        participants = [args.participant]
    else:
        # Auto-discover participants from mask folder names (e.g. P01-..., P02-...)
        participants = sorted(set(
            d.name.split("-")[0]
            for d in masks_root.iterdir()
            if d.is_dir() and d.name[0] == "P"
        ))
        print(f"Auto-discovered {len(participants)} participant(s): {participants}")

    for participant in participants:
        rectify_participant_masks(participant, masks_root, calib_root,
                                  output_root, frames_root, args.re_rectify)

    print("\nAll done.")
    print(f"Rectified masks saved under: {output_root / 'rectified_masks'}")


if __name__ == "__main__":
    main()
