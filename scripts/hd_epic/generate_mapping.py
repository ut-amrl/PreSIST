#!/usr/bin/env python3
"""
generate_mapping.py

Generates a master mapping CSV covering the entire HD-EPIC dataset.
Every row is one (participant, video, frame) combination, with all
the information needed to go from any identifier to any other.

Columns:
  participant       - e.g. P01
  video_id          - e.g. P01-20240202-110250
  frame_number      - MP4 frame number (30fps)
  unix_us           - Unix epoch microseconds (RGB filename stem)
  rgb_filename      - e.g. 1706871897724417.png
  rgb_path          - full path to rectified RGB frame
  rgb_exists        - True/False
  mask_id           - hash e.g. 8b139c81bb56c874 (empty if no mask for this frame)
  mask_filename     - e.g. 8b139c81bb56c874.png (empty if no mask)
  mask_path         - full path to rectified mask (empty if no mask)
  mask_exists       - True/False (empty if no mask)
  fixture           - object label e.g. P01_counter.007 (empty if no mask)
  bbox_x1           - bounding box in rectified space (empty if no mask)
  bbox_y1
  bbox_x2
  bbox_y2

Usage:
  python3 generate_mapping.py

  # Single participant only
  python3 generate_mapping.py --participant P01

  # Custom output path
  python3 generate_mapping.py --out /path/to/mapping.csv

Defaults:
  --mask_json    /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/mask_info.json
  --videos_root  /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/Videos
  --frames_root  /robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/hd-epic
  --masks_root   /robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/rectified_masks
  --out          /robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/master_mapping.csv
"""

import argparse
import csv as csvlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytz

DEFAULT_MASK_JSON   = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/mask_info.json"
DEFAULT_VIDEOS_ROOT = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/Videos"
DEFAULT_FRAMES_ROOT = "/robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/hd-epic"
DEFAULT_MASKS_ROOT  = "/robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/rectified_masks"
DEFAULT_OUT         = "/robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/master_mapping.csv"

RECORDING_TIMEZONE  = pytz.timezone("Europe/London")
CROP_OFFSET         = 57  # (1408 - 1294) // 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def video_id_to_start_us(video_id: str) -> int:
    parts    = video_id.split("-")
    dt_naive = datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S")
    dt_local = RECORDING_TIMEZONE.localize(dt_naive)
    return int(dt_local.astimezone(timezone.utc).timestamp() * 1e6)


def load_csv_rows(csv_path: Path) -> list:
    rows = []
    with open(csv_path) as f:
        for row in csvlib.DictReader(f):
            rows.append(row)
    return rows


def frame_to_unix_us(csv_rows: list, frame_number: int, vrs_offset_ns: int) -> int:
    vrs_ns = float(csv_rows[frame_number]["vrs_device_time_ns"])
    return int((vrs_ns + vrs_offset_ns) / 1000)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate master mapping CSV.")
    parser.add_argument("--mask_json",   default=DEFAULT_MASK_JSON)
    parser.add_argument("--videos_root", default=DEFAULT_VIDEOS_ROOT)
    parser.add_argument("--frames_root", default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--masks_root",  default=DEFAULT_MASKS_ROOT)
    parser.add_argument("--out",         default=DEFAULT_OUT)
    parser.add_argument("--participant", default=None,
                        help="Limit to one participant e.g. P01")
    args = parser.parse_args()

    mask_json   = Path(args.mask_json)
    videos_root = Path(args.videos_root)
    frames_root = Path(args.frames_root)
    masks_root  = Path(args.masks_root)
    out_path    = Path(args.out)

    for p, name in [(mask_json, "mask_json"), (videos_root, "videos_root"),
                    (frames_root, "frames_root"), (masks_root, "masks_root")]:
        if not p.exists():
            print(f"ERROR: {name} not found: {p}")
            sys.exit(1)

    with open(mask_json) as f:
        mask_data = json.load(f)

    # Discover participants
    if args.participant:
        participants = [args.participant]
    else:
        participants = sorted(set(k.split("-")[0] for k in mask_data.keys()))
        print(f"Processing {len(participants)} participants: {participants}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    FIELDNAMES = [
        "participant", "video_id", "frame_number", "unix_us",
        "rgb_filename", "rgb_path", "rgb_exists",
        "mask_id", "mask_filename", "mask_path", "mask_exists",
        "fixture", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"
    ]

    total_rows = 0

    with open(out_path, "w", newline="") as out_f:
        writer = csvlib.DictWriter(out_f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for participant in participants:
            print(f"\n=== {participant} ===")

            # Get all video_ids for this participant
            video_ids = sorted([k for k in mask_data if k.startswith(participant + "-")])

            for video_id in video_ids:
                csv_path = videos_root / participant / f"{video_id}_mp4_to_vrs_time_ns.csv"
                if not csv_path.exists():
                    print(f"  [{video_id}] SKIP — CSV not found")
                    continue

                # Load timestamp mapping
                csv_rows      = load_csv_rows(csv_path)
                start_us      = video_id_to_start_us(video_id)
                vrs_offset_ns = int(start_us * 1000) - int(float(csv_rows[0]["vrs_device_time_ns"]))

                # Build frame_number -> list of masks lookup
                frame_to_masks = {}
                for mask_id, info in mask_data[video_id].items():
                    fn = info["frame_number"]
                    if fn not in frame_to_masks:
                        frame_to_masks[fn] = []
                    frame_to_masks[fn].append((mask_id, info))

                # Collect all frame numbers — both extracted RGB frames and mask frames
                rgb_dir   = frames_root / participant / video_id / "rgb"
                mask_dir  = masks_root  / participant / video_id

                # Get all extracted RGB frame timestamps
                rgb_timestamp_to_file = {}
                if rgb_dir.exists():
                    for f in rgb_dir.glob("*.png"):
                        try:
                            rgb_timestamp_to_file[int(f.stem)] = f
                        except ValueError:
                            pass

                # Write one row per (frame, mask) combination
                # For frames with multiple masks, write one row per mask
                # For frames with no masks, write one row with empty mask columns
                seen_frames = set()

                # First pass — all frames that have masks
                for frame_number, mask_list in sorted(frame_to_masks.items()):
                    seen_frames.add(frame_number)

                    if frame_number >= len(csv_rows):
                        print(f"  [{video_id}] Warning: frame {frame_number} out of CSV range")
                        continue

                    unix_us      = frame_to_unix_us(csv_rows, frame_number, vrs_offset_ns)
                    rgb_filename = f"{unix_us}.png"
                    rgb_path     = rgb_dir / rgb_filename
                    rgb_exists   = rgb_path.exists()

                    for mask_id, info in mask_list:
                        mask_filename = f"{mask_id}.png"
                        mask_path     = mask_dir / mask_filename
                        mask_exists   = mask_path.exists()

                        bbox = info.get("bbox", [None, None, None, None])
                        bbox_x1 = int(bbox[0]) - CROP_OFFSET if bbox[0] is not None else ""
                        bbox_y1 = int(bbox[1]) - CROP_OFFSET if bbox[1] is not None else ""
                        bbox_x2 = int(bbox[2]) - CROP_OFFSET if bbox[2] is not None else ""
                        bbox_y2 = int(bbox[3]) - CROP_OFFSET if bbox[3] is not None else ""

                        writer.writerow({
                            "participant":    participant,
                            "video_id":       video_id,
                            "frame_number":   frame_number,
                            "unix_us":        unix_us,
                            "rgb_filename":   rgb_filename,
                            "rgb_path":       str(rgb_path),
                            "rgb_exists":     rgb_exists,
                            "mask_id":        mask_id,
                            "mask_filename":  mask_filename,
                            "mask_path":      str(mask_path),
                            "mask_exists":    mask_exists,
                            "fixture":        info.get("fixture", ""),
                            "bbox_x1":        bbox_x1,
                            "bbox_y1":        bbox_y1,
                            "bbox_x2":        bbox_x2,
                            "bbox_y2":        bbox_y2,
                        })
                        total_rows += 1

                # Second pass — extracted RGB frames that have no masks
                for unix_us, rgb_file in sorted(rgb_timestamp_to_file.items()):
                    # Check if this timestamp matches any mask frame
                    # (skip if already covered above)
                    already_covered = False
                    for fn in seen_frames:
                        if fn < len(csv_rows):
                            fn_us = frame_to_unix_us(csv_rows, fn, vrs_offset_ns)
                            if fn_us == unix_us:
                                already_covered = True
                                break
                    if already_covered:
                        continue

                    writer.writerow({
                        "participant":    participant,
                        "video_id":       video_id,
                        "frame_number":   "",
                        "unix_us":        unix_us,
                        "rgb_filename":   f"{unix_us}.png",
                        "rgb_path":       str(rgb_file),
                        "rgb_exists":     True,
                        "mask_id":        "",
                        "mask_filename":  "",
                        "mask_path":      "",
                        "mask_exists":    "",
                        "fixture":        "",
                        "bbox_x1":        "",
                        "bbox_y1":        "",
                        "bbox_x2":        "",
                        "bbox_y2":        "",
                    })
                    total_rows += 1

                print(f"  [{video_id}] {len(frame_to_masks)} mask frames, "
                      f"{len(rgb_timestamp_to_file)} total rgb frames")

    print(f"\nDone. {total_rows} rows written to {out_path}")
    print(f"\nYou can now query it like:")
    print(f"  # participant + video + frame -> timestamp")
    print(f"  grep 'P01,P01-20240202-110250,3831' {out_path}")
    print(f"  # timestamp -> frame")
    print(f"  grep '1706871897' {out_path}")
    print(f"  # find all masks for a fixture")
    print(f"  grep 'P01_counter.007' {out_path}")


if __name__ == "__main__":
    main()
