#!/usr/bin/env python3
"""
extract_mask_frames_hd_epic.py

Extracts only the exact MP4 frames that have mask annotations, rectifies them,
and saves them into the existing extracted_data/hd-epic/ folder structure
alongside the base fps-grid frames produced by extract_hd_epic.py + rectify_hd_epic.py
(target fps is whatever extract_hd_epic.py was run with — 5 fps is the current default).

MEMORY SAFE: Processes one frame at a time using direct ffmpeg seeking.
Never loads more than one raw frame into memory at a time.

Frame filenames use Unix microsecond epoch timestamps (same format as the base
grid frames) derived from mp4_to_vrs_time_ns.csv for exact accuracy.

Folder structure (output merges into existing):
  extracted_data/hd-epic/<PXX>/<PXX-YYYYMMDD-HHMMSS>/rgb/<timestamp_us>.png

Usage:
  # Single participant
  python3 extract_mask_frames_hd_epic.py --participant P01

  # All participants
  python3 extract_mask_frames_hd_epic.py

  # Force redo already-extracted frames
  python3 extract_mask_frames_hd_epic.py --participant P01 --re_extract

Defaults:
  --mask_json    /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/mask_info.json
  --videos_root  /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/Videos
  --calib_root   /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/intermediate_data
  --output_root  /robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/hd-epic
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytz

DEFAULT_MASK_JSON   = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/mask_info.json"
DEFAULT_VIDEOS_ROOT = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/Videos"
DEFAULT_CALIB_ROOT  = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/intermediate_data"
DEFAULT_OUTPUT_ROOT = "/robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/hd-epic"

RECORDING_TIMEZONE = pytz.timezone("Europe/London")


# ---------------------------------------------------------------------------
# Timestamp utilities
# ---------------------------------------------------------------------------

def video_id_to_start_us(video_id: str) -> int:
    parts    = video_id.split("-")
    dt_naive = datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S")
    dt_local = RECORDING_TIMEZONE.localize(dt_naive)
    return int(dt_local.astimezone(timezone.utc).timestamp() * 1e6)


def load_csv_rows(csv_path: Path) -> list:
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def compute_vrs_offset(csv_rows: list, start_us: int) -> int:
    vrs_ns_frame0 = float(csv_rows[0]["vrs_device_time_ns"])
    return int(start_us * 1000) - int(vrs_ns_frame0)


def frame_to_unix_us(csv_rows: list, frame_number: int, vrs_offset_ns: int) -> int:
    vrs_ns = float(csv_rows[frame_number]["vrs_device_time_ns"])
    return int((vrs_ns + vrs_offset_ns) / 1000)


# ---------------------------------------------------------------------------
# Calibration + FISHEYE624  (identical to rectify_hd_epic.py)
# ---------------------------------------------------------------------------

def load_rgb_calibration(calib_json_path: str) -> np.ndarray:
    with open(calib_json_path) as f:
        data = json.load(f)
    rgb = data["cameras"]["camera-rgb"]
    assert "FISHEYE624" in rgb["model_name"]
    p = rgb["projection_params"]
    assert len(p) == 15
    return np.array(p)


def fisheye624_project(params: np.ndarray, rays: np.ndarray) -> np.ndarray:
    f  = params[0]; cx = params[1]; cy = params[2]
    k  = params[3:9]; p  = params[9:15]
    x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]
    r_u     = np.sqrt(x**2 + y**2)
    theta   = np.arctan2(r_u, z)
    t2      = theta * theta
    theta_d = theta * (1 + k[0]*t2    + k[1]*t2**2 + k[2]*t2**3
                         + k[3]*t2**4 + k[4]*t2**5 + k[5]*t2**6)
    safe_r  = np.where(r_u > 1e-8, r_u, 1.0)
    mx = np.where(r_u > 1e-8, (x / safe_r) * theta_d, 0.0)
    my = np.where(r_u > 1e-8, (y / safe_r) * theta_d, 0.0)
    r2 = mx**2 + my**2; r4 = r2**2
    dx = (p[0]*(r2+2*mx**2) + 2*p[1]*mx*my
          + r2*(p[2]*mx+p[3]*my) + r4*(p[4]*mx+p[5]*my))
    dy = (2*p[0]*mx*my + p[1]*(r2+2*my**2)
          + r2*(p[2]*my+p[3]*mx) + r4*(p[4]*my+p[5]*mx))
    return np.stack([f*(mx+dx)+cx, f*(my+dy)+cy], axis=-1)


def build_rectify_maps(params: np.ndarray, out_w: int, out_h: int):
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
    src  = fisheye624_project(params, rays)
    map1 = np.where(np.isnan(src[:, 0]), -1.0, src[:, 0]).reshape(out_h, out_w).astype(np.float32)
    map2 = np.where(np.isnan(src[:, 1]), -1.0, src[:, 1]).reshape(out_h, out_w).astype(np.float32)
    return map1, map2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_calib_json(calib_root: Path, participant: str, video_id: str):
    seq_calib = calib_root / participant / video_id / "device_calibration.json"
    if seq_calib.exists():
        return seq_calib
    p_dir = calib_root / participant
    if p_dir.exists():
        for seq_dir in sorted(p_dir.iterdir()):
            calib_file = seq_dir / "device_calibration.json"
            if calib_file.exists():
                return calib_file
    return None


def get_crop_from_existing_frames(output_root: Path, participant: str, video_id: str):
    rgb_dir = output_root / participant / video_id / "rgb"
    if not rgb_dir.exists():
        return None
    for f in sorted(rgb_dir.glob("*.png")):
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is not None:
            fh, fw = img.shape[:2]
            cx = 1408 // 2; cy = 1408 // 2
            return cx - fw//2, cy - fh//2, cx + fw//2, cy + fh//2
    return None


def get_native_fps(mp4_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(mp4_path)
    ]
    out = subprocess.check_output(cmd).decode().strip()
    num, den = map(int, out.split("/"))
    return num / den


# ---------------------------------------------------------------------------
# Extract ONE frame — memory safe, uses direct seek
# ---------------------------------------------------------------------------

def extract_one_frame(mp4_path: Path, frame_number: int,
                      native_fps: float, out_path: Path) -> bool:
    """
    Seek to just before the target frame and extract exactly one frame.
    Uses -ss BEFORE -i for fast keyframe seek, minimizing decode work.
    Only ever decodes a handful of frames — never buffers the full video.
    """
    seek_s = max(0.0, (frame_number / native_fps) - 1.0)
    # After the seek, the frame index resets so we use select on the
    # remaining frames relative to the seek point
    frames_after_seek = frame_number - int(seek_s * native_fps)
    frames_after_seek = max(0, frames_after_seek)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{seek_s:.6f}",
        "-i", str(mp4_path),
        "-vf", f"select=eq(n\\,{frames_after_seek})",
        "-vframes", "1",
        "-q:v", "1",
        str(out_path)
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and out_path.exists()


# ---------------------------------------------------------------------------
# Per-sequence
# ---------------------------------------------------------------------------

def process_sequence(video_id: str, participant: str,
                     mask_frames: set,
                     mp4_path: Path, csv_path: Path,
                     calib_root: Path, output_root: Path,
                     re_extract: bool):

    print(f"\n  [{video_id}] {len(mask_frames)} mask frames to extract")

    # Timestamps
    start_us      = video_id_to_start_us(video_id)
    csv_rows      = load_csv_rows(csv_path)
    vrs_offset_ns = compute_vrs_offset(csv_rows, start_us)

    # Calibration + remap maps (built once, reused for all frames)
    calib_file = get_calib_json(calib_root, participant, video_id)
    if calib_file is None:
        print(f"    [SKIP] No calibration found")
        return
    params     = load_rgb_calibration(str(calib_file))
    map1, map2 = build_rectify_maps(params, 1408, 1408)

    # Crop
    crop = get_crop_from_existing_frames(output_root, participant, video_id)
    if crop is None:
        print(f"    [SKIP] No existing base-grid frames — run rectify_hd_epic.py first")
        return
    x0, y0, x1, y1 = crop
    print(f"    Crop: ({x0},{y0})→({x1},{y1})")

    native_fps = get_native_fps(mp4_path)

    rgb_dir = output_root / participant / video_id / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped   = 0
    errors    = 0
    total     = len(mask_frames)

    # Single reusable temp file — never more than one frame on disk at a time
    tmp_fd, tmp_str = tempfile.mkstemp(suffix=".png")
    import os; os.close(tmp_fd)
    tmp_path = Path(tmp_str)

    try:
        for frame_number in sorted(mask_frames):
            if frame_number >= len(csv_rows):
                print(f"    Warning: frame {frame_number} out of CSV range")
                errors += 1
                continue

            unix_us  = frame_to_unix_us(csv_rows, frame_number, vrs_offset_ns)
            out_file = rgb_dir / f"{unix_us}.png"

            if out_file.exists() and not re_extract:
                skipped += 1
                continue

            # Extract to temp file (overwrites previous)
            if tmp_path.exists():
                tmp_path.unlink()

            if not extract_one_frame(mp4_path, frame_number, native_fps, tmp_path):
                print(f"    Warning: ffmpeg failed for frame {frame_number}")
                errors += 1
                continue

            # Read → rectify → crop → save → delete from memory immediately
            raw = cv2.imread(str(tmp_path))
            if raw is None:
                print(f"    Warning: could not read frame {frame_number}")
                errors += 1
                continue

            rectified = cv2.remap(raw, map1, map2, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            del raw

            cv2.imwrite(str(out_file), rectified[y0:y1, x0:x1])
            del rectified

            extracted += 1
            if extracted % 10 == 0:
                print(f"    ... {extracted + skipped}/{total} done ({extracted} new)")

    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print(f"    → {extracted} extracted, {skipped} already done, {errors} errors")


# ---------------------------------------------------------------------------
# Per-participant
# ---------------------------------------------------------------------------

def process_participant(participant: str, mask_data: dict,
                        videos_root: Path, calib_root: Path,
                        output_root: Path, re_extract: bool):
    print(f"\n=== {participant} ===")

    video_ids = sorted([k for k in mask_data.keys() if k.startswith(participant + "-")])
    if not video_ids:
        print(f"  [SKIP] No mask sequences found")
        return

    print(f"  {len(video_ids)} sequences with masks")

    for video_id in video_ids:
        mp4_path = videos_root / participant / f"{video_id}.mp4"
        csv_path = videos_root / participant / f"{video_id}_mp4_to_vrs_time_ns.csv"

        if not mp4_path.exists():
            print(f"\n  [{video_id}] [SKIP] MP4 not found")
            continue
        if not csv_path.exists():
            print(f"\n  [{video_id}] [SKIP] CSV not found")
            continue

        mask_frames = set(
            info["frame_number"] for info in mask_data[video_id].values()
        )

        process_sequence(video_id, participant, mask_frames,
                         mp4_path, csv_path, calib_root,
                         output_root, re_extract)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract exact mask frames from HD-EPIC MP4s and rectify them.")
    parser.add_argument("--mask_json",   default=DEFAULT_MASK_JSON)
    parser.add_argument("--videos_root", default=DEFAULT_VIDEOS_ROOT)
    parser.add_argument("--calib_root",  default=DEFAULT_CALIB_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--participant", default=None,
                        help="Single participant e.g. P01. Omit for all.")
    parser.add_argument("--re_extract",  action="store_true",
                        help="Re-extract and re-rectify even if output already exists.")
    args = parser.parse_args()

    for p, name in [(Path(args.mask_json),   "mask_json"),
                    (Path(args.videos_root), "videos_root"),
                    (Path(args.calib_root),  "calib_root"),
                    (Path(args.output_root), "output_root")]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}")
            sys.exit(1)

    with open(args.mask_json) as f:
        mask_data = json.load(f)

    videos_root = Path(args.videos_root)
    calib_root  = Path(args.calib_root)
    output_root = Path(args.output_root)

    if args.participant:
        participants = [args.participant]
    else:
        participants = sorted(set(k.split("-")[0] for k in mask_data.keys()))
        print(f"Auto-discovered {len(participants)} participant(s): {participants}")

    for participant in participants:
        process_participant(participant, mask_data, videos_root,
                            calib_root, output_root, args.re_extract)

    print("\nAll done.")


if __name__ == "__main__":
    main()
