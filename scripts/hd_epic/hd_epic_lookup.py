#!/usr/bin/env python3
"""
hd_epic_lookup.py

Lookup library for HD-EPIC dataset mappings. Reads directly from source files
(mask_info.json and mp4_to_vrs_time_ns.csv) rather than derived data.

Supports:
  - participant + video + frame_number  ->  unix_us timestamp
  - participant + video + unix_us       ->  frame_number
  - participant + video + mask_id       ->  frame_number, timestamp, fixture, bbox
  - participant + video + fixture       ->  all matching masks
  - participant + fixture               ->  all matching masks across all videos

Usage as a library:
  from hd_epic_lookup import HDEpicLookup
  lookup = HDEpicLookup()

  # frame number -> timestamp
  unix_us = lookup.frame_to_timestamp('P01', 'P01-20240202-110250', 3831)

  # timestamp -> nearest frame number
  frame_number = lookup.timestamp_to_frame('P01', 'P01-20240202-110250', 1706871897724417)

  # mask id -> full info including paths
  info = lookup.mask_info('P01', 'P01-20240202-110250', '8b139c81bb56c874')

  # fixture -> all masks across dataset
  masks = lookup.find_by_fixture('P01_counter.007')

Usage from command line:
  # frame -> timestamp
  python3 hd_epic_lookup.py frame_to_timestamp P01 P01-20240202-110250 3831

  # timestamp -> frame
  python3 hd_epic_lookup.py timestamp_to_frame P01 P01-20240202-110250 1706871897724417

  # mask info
  python3 hd_epic_lookup.py mask_info P01 P01-20240202-110250 8b139c81bb56c874

  # find by fixture
  python3 hd_epic_lookup.py find_by_fixture P01_counter.007

  # find by fixture for one participant
  python3 hd_epic_lookup.py find_by_fixture P01_counter.007 --participant P01
"""

import csv
import json
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pytz

# ---------------------------------------------------------------------------
# Default paths — edit these if your layout differs
# ---------------------------------------------------------------------------
DEFAULT_MASK_JSON   = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/mask_info.json"
DEFAULT_VIDEOS_ROOT = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/Videos"
DEFAULT_FRAMES_ROOT = "/robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/hd-epic"
DEFAULT_MASKS_ROOT  = "/robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/rectified_masks"

RECORDING_TIMEZONE  = pytz.timezone("Europe/London")
CROP_OFFSET         = 57  # (1408 - 1294) // 2


class HDEpicLookup:
    def __init__(self,
                 mask_json   = DEFAULT_MASK_JSON,
                 videos_root = DEFAULT_VIDEOS_ROOT,
                 frames_root = DEFAULT_FRAMES_ROOT,
                 masks_root  = DEFAULT_MASKS_ROOT):

        self.videos_root = Path(videos_root)
        self.frames_root = Path(frames_root)
        self.masks_root  = Path(masks_root)

        with open(mask_json) as f:
            self._mask_data = json.load(f)

        # Build fixture -> [(video_id, mask_id, info)] index at init time
        self._fixture_index = {}
        for video_id, masks in self._mask_data.items():
            for mask_id, info in masks.items():
                fixture = info.get("fixture", "")
                if fixture not in self._fixture_index:
                    self._fixture_index[fixture] = []
                self._fixture_index[fixture].append((video_id, mask_id, info))

    # -----------------------------------------------------------------------
    # CSV loading — cached per video_id so we only read each file once
    # -----------------------------------------------------------------------

    @lru_cache(maxsize=256)
    def _load_csv(self, video_id: str):
        """Load mp4_to_vrs_time_ns.csv for a video. Cached after first load."""
        participant = video_id.split("-")[0]
        csv_path = self.videos_root / participant / f"{video_id}_mp4_to_vrs_time_ns.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        rows = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                rows.append(float(row["vrs_device_time_ns"]))
        return rows

    @lru_cache(maxsize=256)
    def _vrs_offset(self, video_id: str) -> int:
        """VRS device clock offset to convert to Unix epoch nanoseconds."""
        parts    = video_id.split("-")
        dt_naive = datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S")
        dt_local = RECORDING_TIMEZONE.localize(dt_naive)
        start_us = int(dt_local.astimezone(timezone.utc).timestamp() * 1e6)
        rows = self._load_csv(video_id)
        return int(start_us * 1000) - int(rows[0])

    def _vrs_to_unix_us(self, video_id: str, vrs_ns: float) -> int:
        return int((vrs_ns + self._vrs_offset(video_id)) / 1000)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def frame_to_timestamp(self, participant: str, video_id: str,
                            frame_number: int) -> int:
        """
        Convert a frame number to Unix epoch microseconds.

        Args:
            participant:  e.g. 'P01'
            video_id:     e.g. 'P01-20240202-110250'
            frame_number: MP4 frame index (0-based, 30fps)

        Returns:
            Unix epoch microseconds (int) — matches the RGB filename stem
        """
        rows = self._load_csv(video_id)
        if frame_number >= len(rows):
            raise ValueError(f"frame_number {frame_number} out of range "
                             f"(video has {len(rows)} frames)")
        return self._vrs_to_unix_us(video_id, rows[frame_number])


    def timestamp_to_frame(self, participant: str, video_id: str,
                           unix_us: int) -> dict:
        """
        Find the frame number closest to a Unix epoch microsecond timestamp.

        Args:
            participant: e.g. 'P01'
            video_id:    e.g. 'P01-20240202-110250'
            unix_us:     Unix epoch microseconds

        Returns:
            dict with keys:
              frame_number  - closest frame index
              unix_us       - actual timestamp of that frame
              offset_us     - how far off from the requested timestamp
        """
        rows   = self._load_csv(video_id)
        offset = self._vrs_offset(video_id)

        target_vrs = unix_us * 1000 - offset
        best_idx   = min(range(len(rows)), key=lambda i: abs(rows[i] - target_vrs))
        actual_us  = self._vrs_to_unix_us(video_id, rows[best_idx])

        return {
            "frame_number": best_idx,
            "unix_us":      actual_us,
            "offset_us":    abs(actual_us - unix_us),
        }


    def mask_info(self, participant: str, video_id: str, mask_id: str) -> dict:
        """
        Get full info for a mask including file paths.

        Args:
            participant: e.g. 'P01'
            video_id:    e.g. 'P01-20240202-110250'
            mask_id:     hash e.g. '8b139c81bb56c874'

        Returns:
            dict with keys:
              mask_id, frame_number, unix_us, fixture, bbox_rectified,
              rgb_path, mask_path, rgb_exists, mask_exists
        """
        if video_id not in self._mask_data:
            raise KeyError(f"video_id {video_id} not found in mask_info.json")
        if mask_id not in self._mask_data[video_id]:
            raise KeyError(f"mask_id {mask_id} not found in {video_id}")

        info         = self._mask_data[video_id][mask_id]
        frame_number = info["frame_number"]
        unix_us      = self.frame_to_timestamp(participant, video_id, frame_number)

        rgb_path  = self.frames_root / participant / video_id / "rgb" / f"{unix_us}.png"
        mask_path = self.masks_root  / participant / video_id / f"{mask_id}.png"

        bbox = info.get("bbox", [None]*4)
        bbox_rectified = {
            "x1": int(bbox[0]) - CROP_OFFSET if bbox[0] is not None else None,
            "y1": int(bbox[1]) - CROP_OFFSET if bbox[1] is not None else None,
            "x2": int(bbox[2]) - CROP_OFFSET if bbox[2] is not None else None,
            "y2": int(bbox[3]) - CROP_OFFSET if bbox[3] is not None else None,
        }

        return {
            "mask_id":        mask_id,
            "frame_number":   frame_number,
            "unix_us":        unix_us,
            "fixture":        info.get("fixture", ""),
            "bbox_rectified": bbox_rectified,
            "rgb_path":       rgb_path,
            "mask_path":      mask_path,
            "rgb_exists":     rgb_path.exists(),
            "mask_exists":    mask_path.exists(),
        }


    def find_by_fixture(self, fixture: str, participant: str = None) -> list:
        """
        Find all masks for a given fixture label across the dataset.

        Args:
            fixture:     e.g. 'P01_counter.007'
            participant: optional filter e.g. 'P01'

        Returns:
            list of dicts, each with:
              participant, video_id, mask_id, frame_number, unix_us,
              rgb_path, mask_path, rgb_exists, mask_exists
        """
        matches = self._fixture_index.get(fixture, [])
        results = []
        for video_id, mask_id, info in matches:
            p = video_id.split("-")[0]
            if participant and p != participant:
                continue
            try:
                frame_number = info["frame_number"]
                unix_us      = self.frame_to_timestamp(p, video_id, frame_number)
                rgb_path     = self.frames_root / p / video_id / "rgb" / f"{unix_us}.png"
                mask_path    = self.masks_root  / p / video_id / f"{mask_id}.png"
                results.append({
                    "participant": p,
                    "video_id":    video_id,
                    "mask_id":     mask_id,
                    "frame_number": frame_number,
                    "unix_us":     unix_us,
                    "rgb_path":    rgb_path,
                    "mask_path":   mask_path,
                    "rgb_exists":  rgb_path.exists(),
                    "mask_exists": mask_path.exists(),
                })
            except Exception as e:
                results.append({
                    "participant": p,
                    "video_id":    video_id,
                    "mask_id":     mask_id,
                    "error":       str(e),
                })
        return results


    def all_masks_for_video(self, participant: str, video_id: str) -> list:
        """
        Get info for every mask in a video, sorted by frame number.

        Returns:
            list of dicts (same format as mask_info())
        """
        if video_id not in self._mask_data:
            return []
        results = []
        for mask_id in self._mask_data[video_id]:
            try:
                results.append(self.mask_info(participant, video_id, mask_id))
            except Exception as e:
                results.append({"mask_id": mask_id, "error": str(e)})
        return sorted(results, key=lambda x: x.get("frame_number", 0))


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="HD-EPIC dataset lookup tool.")
    subparsers = parser.add_subparsers(dest="command")

    # frame_to_timestamp
    p1 = subparsers.add_parser("frame_to_timestamp",
                                help="Convert frame number to Unix timestamp")
    p1.add_argument("participant")
    p1.add_argument("video_id")
    p1.add_argument("frame_number", type=int)

    # timestamp_to_frame
    p2 = subparsers.add_parser("timestamp_to_frame",
                                help="Find frame number closest to timestamp")
    p2.add_argument("participant")
    p2.add_argument("video_id")
    p2.add_argument("unix_us", type=int)

    # mask_info
    p3 = subparsers.add_parser("mask_info",
                                help="Get full info for a mask")
    p3.add_argument("participant")
    p3.add_argument("video_id")
    p3.add_argument("mask_id")

    # find_by_fixture
    p4 = subparsers.add_parser("find_by_fixture",
                                help="Find all masks for a fixture label")
    p4.add_argument("fixture")
    p4.add_argument("--participant", default=None)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    lookup = HDEpicLookup()

    if args.command == "frame_to_timestamp":
        unix_us = lookup.frame_to_timestamp(args.participant, args.video_id,
                                             args.frame_number)
        print(f"unix_us:      {unix_us}")
        print(f"rgb_filename: {unix_us}.png")

    elif args.command == "timestamp_to_frame":
        result = lookup.timestamp_to_frame(args.participant, args.video_id,
                                            args.unix_us)
        for k, v in result.items():
            print(f"{k}: {v}")

    elif args.command == "mask_info":
        result = lookup.mask_info(args.participant, args.video_id, args.mask_id)
        for k, v in result.items():
            print(f"{k}: {v}")

    elif args.command == "find_by_fixture":
        results = lookup.find_by_fixture(args.fixture, args.participant)
        print(f"Found {len(results)} masks for fixture '{args.fixture}'")
        for r in results:
            if "error" in r:
                print(f"  ERROR {r['video_id']} {r['mask_id']}: {r['error']}")
            else:
                print(f"  {r['participant']}  {r['video_id']}  "
                      f"frame={r['frame_number']}  mask={r['mask_id']}  "
                      f"rgb_exists={r['rgb_exists']}  mask_exists={r['mask_exists']}")


if __name__ == "__main__":
    main()
