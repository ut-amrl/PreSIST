#!/usr/bin/env python3
"""
inspect_masks.py - Visualize segmentation masks overlaid on extracted frames
Usage: python3 inspect_masks.py --participant P01 --video P01-20240202-110250
"""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import csv, pytz, argparse
from datetime import datetime, timezone

MASK_JSON    = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/mask_info.json"
OUTPUT_ROOT  = "/robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/hd-epic"
MASKS_ROOT   = "/robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/rectified_masks"
CALIB_CSV    = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/Videos"

CROP_X0, CROP_Y0 = 57, 57  # bbox coords are in 1408x1408 space

parser = argparse.ArgumentParser()
parser.add_argument("--participant", required=True)
parser.add_argument("--video",       required=True)
parser.add_argument("--fixture",     default=None)
parser.add_argument("--out_dir", default="/robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/inspection_output")
args = parser.parse_args()

with open(MASK_JSON) as f:
    mask_data = json.load(f)

video_masks = mask_data.get(args.video, {})
rgb_dir  = Path(OUTPUT_ROOT) / args.participant / args.video / "rgb"
mask_dir = Path(MASKS_ROOT)  / args.participant / args.video
out_dir  = Path(args.out_dir) / args.video
out_dir.mkdir(parents=True, exist_ok=True)

# Group masks by frame_number
frames = defaultdict(list)
for mask_id, info in video_masks.items():
    if args.fixture and info["fixture"] != args.fixture:
        continue
    frames[info["frame_number"]].append((mask_id, info))

# Load CSV for frame -> timestamp mapping
RECORDING_TIMEZONE = pytz.timezone("Europe/London")
csv_path = Path(CALIB_CSV) / args.participant / f"{args.video}_mp4_to_vrs_time_ns.csv"

def video_id_to_start_us(video_id):
    parts = video_id.split("-")
    dt_naive = datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S")
    dt_local = RECORDING_TIMEZONE.localize(dt_naive)
    return int(dt_local.astimezone(timezone.utc).timestamp() * 1e6)

with open(csv_path) as f:
    csv_rows = list(csv.DictReader(f))

start_us = video_id_to_start_us(args.video)
vrs_ns_frame0 = float(csv_rows[0]["vrs_device_time_ns"])
vrs_offset_ns = int(start_us * 1000) - int(vrs_ns_frame0)

# Consistent colors per fixture
rng = np.random.default_rng(42)
fixture_colors = {}
def get_color(fixture):
    if fixture not in fixture_colors:
        fixture_colors[fixture] = tuple(int(x) for x in rng.integers(50, 255, 3))
    return fixture_colors[fixture]

print(f"Found {len(frames)} unique frames with masks")

for frame_number, mask_list in sorted(frames.items()):
    vrs_ns  = float(csv_rows[frame_number]["vrs_device_time_ns"])
    unix_us = int((vrs_ns + vrs_offset_ns) / 1000)
    png_path = rgb_dir / f"{unix_us}.png"

    if not png_path.exists():
        print(f"  [MISSING] frame {frame_number} -> {png_path.name}")
        continue

    img = cv2.imread(str(png_path))

    for mask_id, info in mask_list:
        color = get_color(info["fixture"])

        # Overlay rectified mask
        mask_path = mask_dir / f"{mask_id}.png"
        if mask_path.exists():
            mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask_img.ndim == 2:
                binary_mask = mask_img > 0
            else:
                binary_mask = mask_img[:, :, 2] > 0  # R channel
            overlay = img.copy()
            overlay[binary_mask] = color
            img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        else:
            print(f"  [MISSING MASK] {mask_id}")

        # Label (bbox still in 1408x1408 space, subtract crop offset)
        x1 = int(info["bbox"][0]) - CROP_X0
        y1 = int(info["bbox"][1]) - CROP_Y0
        cv2.putText(img, f"{info['fixture']} [{mask_id[:8]}]",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    out_path = out_dir / f"frame_{frame_number:05d}_{unix_us}.png"
    cv2.imwrite(str(out_path), img)
    print(f"  frame {frame_number:5d} | {len(mask_list)} masks | {out_path.name}")

print(f"\nSaved to {out_dir}")
