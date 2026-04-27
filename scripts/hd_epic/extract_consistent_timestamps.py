#!/usr/bin/env python3
"""
extract_consistent_timestamps.py

Builds a parallel HD-EPIC output tree at
  /robodata/aaadkins/.../extracted_data/hd-epic-consistent-timestamps/<P>/<seq>/

with three pieces per sequence:
  rgb/<unix_us>.png            — RGB frames named ENTIRELY via VRS device time:
                                 unix_us = (vrs_ns[N] + vrs_offset_ns) / 1000.
                                 Coverage = 1 fps source-frame grid ∪ all mask-annotated frames.
  masks/<mask_id>.png          — segmentation masks re-rectified with the per-sequence
                                 calibration (fixes the latent bug in rectify_masks_hd_epic.py
                                 that uses the FIRST per-participant calibration for ALL sequences).
  frame_index_to_unix_us.csv   — self-describing source_frame_number → unix_us map.

Plus one CSV per participant at
  /robodata/aaadkins/.../extracted_data/hd-epic-consistent-timestamps/<P>/mapping.csv
listing one row per (video, frame_number, mask) and one row per (video, frame_number) for
RGB-only frames.

Idempotent by default: every per-frame and per-mask write checks for an existing file.
Pass --force to regenerate from scratch.

Usage:
  # Sanity-check single sequence (default):
  python3 extract_consistent_timestamps.py --participant P01 --video P01-20240202-110250

  # Full rollout across all participants and sequences:
  python3 extract_consistent_timestamps.py --all

  # Force regenerate everything:
  python3 extract_consistent_timestamps.py --all --force

  # Subsetting:
  python3 extract_consistent_timestamps.py --participant P03
  python3 extract_consistent_timestamps.py --participant P03 --video P03-20240210-101122
"""

import argparse
import csv as csvlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Allow importing from sibling scripts in scripts/hd_epic/
sys.path.insert(0, str(Path(__file__).parent))

from extract_mask_frames_hd_epic import (
    video_id_to_start_us,
    load_csv_rows,
    compute_vrs_offset,
    frame_to_unix_us,
    load_rgb_calibration,
    build_rectify_maps,
    get_calib_json,
    get_native_fps,
    extract_one_frame,
)
from rectify_masks_hd_epic import detect_mask_size, crop_from_frame_size


def compute_crop_strict(map1: np.ndarray, map2: np.ndarray,
                        cx_src: float, cy_src: float, valid_radius: float,
                        out_w: int, out_h: int):
    """Largest centered square whose every output pixel maps to a source pixel
    within `valid_radius` of the principal point — i.e. inside the fisheye lens'
    actual illumination circle. Matches projectaria_tools' criterion.

    Output coordinates are in the rectified pinhole space; (cx_src, cy_src,
    valid_radius) are in the source fisheye image's pixel space.
    """
    dx = map1 - cx_src
    dy = map2 - cy_src
    valid = (dx * dx + dy * dy) <= valid_radius * valid_radius

    cx_out = out_w // 2
    cy_out = out_h // 2
    lo, hi, best = 0, min(cx_out, cy_out), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if valid[cy_out - mid:cy_out + mid, cx_out - mid:cx_out + mid].all():
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return cx_out - best, cy_out - best, cx_out + best, cy_out + best


def load_rgb_calib_meta(calib_json_path: str):
    """Return (projection_params: np.ndarray[15], cx_src, cy_src, valid_radius)."""
    with open(calib_json_path) as f:
        data = json.load(f)
    rgb = data["cameras"]["camera-rgb"]
    assert "FISHEYE624" in rgb["model_name"], f"Unexpected model: {rgb['model_name']}"
    p = np.array(rgb["projection_params"])
    assert len(p) == 15
    return p, float(p[1]), float(p[2]), float(rgb["valid_radius"])


DEFAULT_MASK_JSON   = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/mask_info.json"
DEFAULT_VIDEOS_ROOT = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/Videos"
DEFAULT_CALIB_ROOT  = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/intermediate_data"
DEFAULT_MASKS_ROOT  = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/hd_epic_association_masks/masks"
DEFAULT_OUTPUT_ROOT = "/robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/hd-epic-consistent-timestamps"

RECTIFIED_SIZE = 1408  # Aria FISHEYE624 RGB rectifies into a 1408x1408 pinhole image
EXPECTED_FPS   = 30    # HD-EPIC is uniformly 30 fps; assert and fail loud otherwise

MAPPING_FIELDS = [
    "video_id", "frame_number", "unix_us",
    "rgb_filename", "rgb_path", "rgb_exists",
    "mask_id", "mask_filename", "mask_path", "mask_exists",
    "fixture", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
]


# ---------------------------------------------------------------------------
# Per-sequence
# ---------------------------------------------------------------------------

def process_sequence(participant: str,
                     video_id: str,
                     mask_data_for_video: dict,
                     videos_root: Path,
                     calib_root: Path,
                     masks_root: Path,
                     output_root: Path,
                     target_fps: float,
                     force: bool,
                     mapping_rows: list) -> bool:
    """Process one sequence end-to-end. Returns True on success, False on skip-with-error."""

    print(f"\n=== {participant}/{video_id} ===")

    mp4_path = videos_root / participant / f"{video_id}.mp4"
    csv_path = videos_root / participant / f"{video_id}_mp4_to_vrs_time_ns.csv"

    if not mp4_path.exists():
        print(f"  [SKIP] MP4 not found: {mp4_path}")
        return False
    if not csv_path.exists():
        print(f"  [SKIP] CSV not found: {csv_path}")
        return False

    # 1. Calibration (per-sequence-preferred)
    calib_file = get_calib_json(calib_root, participant, video_id)
    if calib_file is None:
        print(f"  [SKIP] No device_calibration.json found for {participant}")
        return False
    params, cx_src, cy_src, valid_radius = load_rgb_calib_meta(str(calib_file))
    print(f"  calib: {calib_file.parent.name}  f={params[0]:.3f}  "
          f"cx={cx_src:.3f}  cy={cy_src:.3f}  valid_radius={valid_radius:.2f}")

    # 2. CSV / VRS offset
    csv_rows = load_csv_rows(csv_path)
    start_us = video_id_to_start_us(video_id)
    vrs_offset_ns = compute_vrs_offset(csv_rows, start_us)

    # 3. Determine source frame index set
    native_fps = get_native_fps(mp4_path)
    if abs(native_fps - EXPECTED_FPS) > 1e-3:
        print(f"  [SKIP] Expected {EXPECTED_FPS} fps, got {native_fps}")
        return False

    stride = round(native_fps / target_fps)
    fps_indices  = set(range(0, len(csv_rows), stride))
    mask_indices = set(int(info["frame_number"]) for info in mask_data_for_video.values())
    # Drop any mask frame_numbers that fall outside the CSV (defensive)
    mask_indices = {fn for fn in mask_indices if 0 <= fn < len(csv_rows)}
    union_indices = sorted(fps_indices | mask_indices)
    print(f"  frames to process: {len(union_indices)}  "
          f"(fps={len(fps_indices)}, mask={len(mask_indices)}, "
          f"both={len(fps_indices & mask_indices)})")

    # 4. Build remap maps + crop (always recomputed from per-sequence calib).
    # Use the strict valid-radius criterion (matches projectaria_tools): rejects
    # rectified-pinhole pixels whose source coordinate falls outside the fisheye
    # lens' illumination circle. Result varies per calibration.
    map1_rgb, map2_rgb = build_rectify_maps(params, RECTIFIED_SIZE, RECTIFIED_SIZE)
    crop = compute_crop_strict(map1_rgb, map2_rgb, cx_src, cy_src, valid_radius,
                               RECTIFIED_SIZE, RECTIFIED_SIZE)
    x0, y0, x1, y1 = crop
    print(f"  crop (strict): ({x0},{y0})->({x1},{y1})  =  {x1-x0}x{y1-y0} px")
    rgb_w_actual = x1 - x0
    rgb_h_actual = y1 - y0

    # 5. Phase 1 — RGB extraction
    seq_out_dir = output_root / participant / video_id
    rgb_dir = seq_out_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    fn_to_unix = {}
    frame_source = {}
    for fn in fps_indices:  frame_source[fn] = "fps"
    for fn in mask_indices:
        frame_source[fn] = "both" if fn in fps_indices else "mask"

    rgb_extracted = 0
    rgb_skipped   = 0
    rgb_errors    = 0

    tmp_fd, tmp_str = tempfile.mkstemp(suffix=".png")
    os.close(tmp_fd)
    tmp_path = Path(tmp_str)

    try:
        for i, fn in enumerate(union_indices):
            unix_us = frame_to_unix_us(csv_rows, fn, vrs_offset_ns)
            fn_to_unix[fn] = unix_us
            out_file = rgb_dir / f"{unix_us}.png"

            if out_file.exists() and not force:
                rgb_skipped += 1
                continue

            if tmp_path.exists():
                tmp_path.unlink()

            if not extract_one_frame(mp4_path, fn, native_fps, tmp_path):
                print(f"    WARN: ffmpeg failed for frame {fn}")
                rgb_errors += 1
                continue

            raw = cv2.imread(str(tmp_path))
            if raw is None:
                print(f"    WARN: could not read extracted frame {fn}")
                rgb_errors += 1
                continue

            rectified = cv2.remap(raw, map1_rgb, map2_rgb, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            del raw
            cv2.imwrite(str(out_file), rectified[y0:y1, x0:x1])
            del rectified
            rgb_extracted += 1

            if rgb_extracted % 25 == 0:
                print(f"    Phase 1: {rgb_extracted + rgb_skipped}/{len(union_indices)} "
                      f"({rgb_extracted} new)")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print(f"  Phase 1 RGB: {rgb_extracted} new, {rgb_skipped} skipped, "
          f"{rgb_errors} errors  (crop {rgb_w_actual}x{rgb_h_actual})")

    # 6. Phase 2 — mask re-rectification
    mask_src_dir = masks_root / video_id
    mask_dst_dir = seq_out_dir / "masks"

    mask_files = []
    if mask_src_dir.exists():
        mask_files = sorted(mask_src_dir.glob("*.png"))

    if not mask_files:
        print(f"  Phase 2: no mask source files at {mask_src_dir}")
    else:
        mask_dst_dir.mkdir(parents=True, exist_ok=True)
        msize = detect_mask_size(mask_src_dir)
        if msize is None:
            print(f"  Phase 2: detect_mask_size returned None")
        else:
            ms_w, ms_h = msize
            mmap1, mmap2 = build_rectify_maps(params, ms_w, ms_h)
            mcrop = crop_from_frame_size(ms_w, ms_h, rgb_w_actual, rgb_h_actual)
            mx0, my0, mx1, my1 = mcrop

            mask_extracted = 0
            mask_skipped = 0
            mask_errors  = 0
            for src in mask_files:
                dst = mask_dst_dir / src.name
                if dst.exists() and not force:
                    mask_skipped += 1
                    continue
                m = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
                if m is None:
                    print(f"    WARN: could not read mask {src.name}")
                    mask_errors += 1
                    continue
                rect = cv2.remap(m, mmap1, mmap2, cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                cv2.imwrite(str(dst), rect[my0:my1, mx0:mx1])
                mask_extracted += 1

            print(f"  Phase 2 masks: {mask_extracted} new, {mask_skipped} skipped, "
                  f"{mask_errors} errors  (mask src {ms_w}x{ms_h}, crop {mx1-mx0}x{my1-my0})")

    # 7. Phase 3 — frame_index_to_unix_us.csv
    f_csv = seq_out_dir / "frame_index_to_unix_us.csv"
    if force or not f_csv.exists():
        with open(f_csv, "w", newline="") as f:
            writer = csvlib.writer(f)
            writer.writerow(["source_frame_number", "unix_us", "source"])
            for fn in union_indices:
                writer.writerow([fn, fn_to_unix[fn], frame_source[fn]])

    # 8. Accumulate mapping rows for the per-participant CSV.
    # Bbox is in source-mask 1408x1408 space; convert to rectified-cropped space by
    # subtracting the mask crop offset (matches existing hd_epic_lookup.py CROP_OFFSET
    # behavior, but uses the actual mx0/my0 from the per-sequence crop rather than
    # a hardcoded 57).
    mx0_for_bbox = (RECTIFIED_SIZE // 2) - rgb_w_actual // 2
    my0_for_bbox = (RECTIFIED_SIZE // 2) - rgb_h_actual // 2

    frame_to_masks = {}
    for mid, info in mask_data_for_video.items():
        fn_i = int(info["frame_number"])
        frame_to_masks.setdefault(fn_i, []).append((mid, info))

    for fn in union_indices:
        unix_us = fn_to_unix[fn]
        rgb_filename = f"{unix_us}.png"
        rgb_path = rgb_dir / rgb_filename
        rgb_exists = rgb_path.exists()

        masks_here = frame_to_masks.get(fn, [])

        if masks_here:
            for mid, info in masks_here:
                mask_filename = f"{mid}.png"
                mask_path = mask_dst_dir / mask_filename
                mask_exists = mask_path.exists()

                bbox = info.get("bbox", [None, None, None, None])
                if bbox and None not in bbox:
                    bbox_rect = [
                        int(bbox[0]) - mx0_for_bbox,
                        int(bbox[1]) - my0_for_bbox,
                        int(bbox[2]) - mx0_for_bbox,
                        int(bbox[3]) - my0_for_bbox,
                    ]
                else:
                    bbox_rect = ["", "", "", ""]

                mapping_rows.append({
                    "video_id":      video_id,
                    "frame_number":  fn,
                    "unix_us":       unix_us,
                    "rgb_filename":  rgb_filename,
                    "rgb_path":      str(rgb_path),
                    "rgb_exists":    rgb_exists,
                    "mask_id":       mid,
                    "mask_filename": mask_filename,
                    "mask_path":     str(mask_path),
                    "mask_exists":   mask_exists,
                    "fixture":       info.get("fixture", ""),
                    "bbox_x1":       bbox_rect[0],
                    "bbox_y1":       bbox_rect[1],
                    "bbox_x2":       bbox_rect[2],
                    "bbox_y2":       bbox_rect[3],
                })
        else:
            mapping_rows.append({
                "video_id":      video_id,
                "frame_number":  fn,
                "unix_us":       unix_us,
                "rgb_filename":  rgb_filename,
                "rgb_path":      str(rgb_path),
                "rgb_exists":    rgb_exists,
                "mask_id":       "",
                "mask_filename": "",
                "mask_path":     "",
                "mask_exists":   "",
                "fixture":       "",
                "bbox_x1":       "", "bbox_y1": "", "bbox_x2": "", "bbox_y2": "",
            })

    return True


# ---------------------------------------------------------------------------
# Per-participant
# ---------------------------------------------------------------------------

def process_participant(participant: str,
                        mask_data: dict,
                        videos_root: Path,
                        calib_root: Path,
                        masks_root: Path,
                        output_root: Path,
                        target_fps: float,
                        force: bool,
                        only_video: str = None):

    video_ids = sorted([k for k in mask_data.keys() if k.startswith(participant + "-")])
    if only_video:
        if only_video not in video_ids:
            # Video may not have masks but we still want to process it. Build the index
            # from the videos dir if needed.
            if (videos_root / participant / f"{only_video}.mp4").exists():
                video_ids = [only_video]
            else:
                print(f"[{participant}] Requested video {only_video} not found")
                return
        else:
            video_ids = [only_video]

    if not video_ids:
        print(f"[{participant}] No sequences with masks; skipping")
        return

    print(f"\n--- Participant {participant}: {len(video_ids)} sequence(s) ---")

    mapping_rows = []
    for video_id in video_ids:
        mask_for_video = mask_data.get(video_id, {})
        process_sequence(participant, video_id, mask_for_video,
                         videos_root, calib_root, masks_root, output_root,
                         target_fps, force, mapping_rows)

    # Write per-participant mapping.csv
    if not mapping_rows:
        print(f"[{participant}] No mapping rows accumulated; not writing mapping.csv")
        return

    mapping_path = output_root / participant / "mapping.csv"
    if mapping_path.exists() and not force:
        print(f"[{participant}] mapping.csv exists; not overwriting (use --force)")
        return

    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    # Sort rows by (video_id, frame_number, mask_id) for stable output
    mapping_rows.sort(key=lambda r: (r["video_id"], r["frame_number"], r["mask_id"]))

    with open(mapping_path, "w", newline="") as f:
        writer = csvlib.DictWriter(f, fieldnames=MAPPING_FIELDS)
        writer.writeheader()
        for row in mapping_rows:
            writer.writerow(row)

    print(f"[{participant}] Wrote {len(mapping_rows)} rows to {mapping_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Re-extract HD-EPIC RGB + masks with VRS-derived RGB filenames "
                    "and per-sequence calibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mask_json",   default=DEFAULT_MASK_JSON)
    parser.add_argument("--videos_root", default=DEFAULT_VIDEOS_ROOT)
    parser.add_argument("--calib_root",  default=DEFAULT_CALIB_ROOT)
    parser.add_argument("--masks_root",  default=DEFAULT_MASKS_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target_fps",  type=float, default=1.0)
    parser.add_argument("--participant", default="P01",
                        help="Participant ID (e.g. P01). Ignored if --all is set.")
    parser.add_argument("--video", default="P01-20240202-110250",
                        help="Video ID (e.g. P01-20240202-110250). "
                             "If empty string, process all videos for the participant. "
                             "Ignored if --all is set.")
    parser.add_argument("--all", action="store_true",
                        help="Process every participant and sequence.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing PNGs and mapping.csv (default: skip).")
    args = parser.parse_args()

    for p, name in [(Path(args.mask_json),   "mask_json"),
                    (Path(args.videos_root), "videos_root"),
                    (Path(args.calib_root),  "calib_root"),
                    (Path(args.masks_root),  "masks_root")]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}")
            sys.exit(1)

    videos_root = Path(args.videos_root)
    calib_root  = Path(args.calib_root)
    masks_root  = Path(args.masks_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(args.mask_json) as f:
        mask_data = json.load(f)

    if args.all:
        participants = sorted(set(k.split("-")[0] for k in mask_data.keys()))
        print(f"Processing all participants: {participants}")
        for participant in participants:
            process_participant(participant, mask_data, videos_root, calib_root,
                                masks_root, output_root, args.target_fps,
                                args.force, only_video=None)
    else:
        only_video = args.video if args.video else None
        process_participant(args.participant, mask_data, videos_root, calib_root,
                            masks_root, output_root, args.target_fps,
                            args.force, only_video=only_video)

    print("\nAll done.")


if __name__ == "__main__":
    main()
