#!/usr/bin/env python3
"""
hd_epic_lookup.py - Look up any mask by ID and get all details

Usage:
  python3 hd_epic_lookup.py <mask_id>
  python3 hd_epic_lookup.py frame_to_timestamp P01-20240202-110250 3831
  python3 hd_epic_lookup.py timestamp_to_frame P01-20240202-110250 1706871897724417
  python3 hd_epic_lookup.py masks_at_frame P01 P01-20240202-110250 3831
  python3 hd_epic_lookup.py find_by_fixture P01_counter.007 [--participant P01]
  python3 hd_epic_lookup.py all_fixtures
"""

import csv
import json
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pytz

DEFAULT_MASK_JSON   = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/mask_info.json"
DEFAULT_VIDEOS_ROOT = "/robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/Videos"
DEFAULT_FRAMES_ROOT = "/robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/hd-epic"
DEFAULT_MASKS_ROOT  = "/robodata/aaadkins/semantic-persistence-filter/external_annotated_data/extracted_data/rectified_masks"

RECORDING_TIMEZONE = pytz.timezone("Europe/London")
CROP_OFFSET        = 57  # (1408 - 1294) // 2


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

        # Build all indexes in one pass
        self._mask_index    = {}   # mask_id    -> (video_id, info)
        self._fixture_index = {}   # fixture    -> [(video_id, mask_id, info)]
        self._frame_index   = {}   # (video_id, frame_number) -> [mask_id]

        for video_id, masks in self._mask_data.items():
            for mask_id, info in masks.items():
                self._mask_index[mask_id] = (video_id, info)

                fixture = info.get("fixture", "")
                self._fixture_index.setdefault(fixture, []).append(
                    (video_id, mask_id, info)
                )

                frame_number = info["frame_number"]
                key = (video_id, frame_number)
                self._frame_index.setdefault(key, []).append(mask_id)

    # -----------------------------------------------------------------------
    # CSV / timestamp helpers
    # -----------------------------------------------------------------------

    @lru_cache(maxsize=256)
    def _load_csv(self, video_id: str):
        participant = video_id.split("-")[0]
        csv_path = self.videos_root / participant / f"{video_id}_mp4_to_vrs_time_ns.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        with open(csv_path) as f:
            return [float(row["vrs_device_time_ns"]) for row in csv.DictReader(f)]

    @lru_cache(maxsize=256)
    def _vrs_offset(self, video_id: str) -> int:
        parts    = video_id.split("-")
        dt_naive = datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S")
        start_us = int(RECORDING_TIMEZONE.localize(dt_naive)
                       .astimezone(timezone.utc).timestamp() * 1e6)
        return int(start_us * 1000) - int(self._load_csv(video_id)[0])

    def _vrs_to_unix_us(self, video_id: str, vrs_ns: float) -> int:
        return int((vrs_ns + self._vrs_offset(video_id)) / 1000)

    # -----------------------------------------------------------------------
    # Core lookups
    # -----------------------------------------------------------------------

    def frame_to_timestamp(self, video_id: str, frame_number: int) -> int:
        """Convert a frame number to Unix epoch microseconds."""
        rows = self._load_csv(video_id)
        if frame_number >= len(rows):
            raise ValueError(f"frame {frame_number} out of range ({len(rows)} frames)")
        return self._vrs_to_unix_us(video_id, rows[frame_number])

    def timestamp_to_frame(self, video_id: str, unix_us: int) -> dict:
        """
        Find the frame number closest to a Unix epoch microsecond timestamp.

        Returns:
            frame_number  - closest frame index
            unix_us       - actual timestamp of that frame
            offset_us     - how far off from the requested timestamp (microseconds)
        """
        rows       = self._load_csv(video_id)
        offset     = self._vrs_offset(video_id)
        target_vrs = unix_us * 1000 - offset
        best_idx   = min(range(len(rows)), key=lambda i: abs(rows[i] - target_vrs))
        actual_us  = self._vrs_to_unix_us(video_id, rows[best_idx])
        return {
            "frame_number": best_idx,
            "unix_us":      actual_us,
            "offset_us":    abs(actual_us - unix_us),
        }

    def lookup(self, mask_id: str) -> dict:
        """
        Give a mask_id, get everything back.
        This is the main entry point for most use cases.
        """
        if mask_id not in self._mask_index:
            raise KeyError(f"mask_id '{mask_id}' not found in dataset")

        video_id, info = self._mask_index[mask_id]
        participant    = video_id.split("-")[0]
        frame_number   = info["frame_number"]
        unix_us        = self.frame_to_timestamp(video_id, frame_number)

        rgb_path  = self.frames_root / participant / video_id / "rgb" / f"{unix_us}.png"
        mask_path = self.masks_root  / participant / video_id / f"{mask_id}.png"

        bbox = info.get("bbox", [None]*4)
        bbox_rectified = [
            int(bbox[0]) - CROP_OFFSET,
            int(bbox[1]) - CROP_OFFSET,
            int(bbox[2]) - CROP_OFFSET,
            int(bbox[3]) - CROP_OFFSET,
        ] if None not in bbox else None

        return {
            "mask_id":        mask_id,
            "participant":    participant,
            "video_id":       video_id,
            "frame_number":   frame_number,
            "unix_us":        unix_us,
            "fixture":        info.get("fixture", ""),
            "3d_location":    info.get("3d_location"),
            "bbox_original":  bbox,
            "bbox_rectified": bbox_rectified,
            "rgb_path":       str(rgb_path),
            "mask_path":      str(mask_path),
            "rgb_exists":     rgb_path.exists(),
            "mask_exists":    mask_path.exists(),
        }

    def masks_at_frame(self, participant: str, video_id: str,
                       frame_number: int) -> list:
        """
        Given participant + video + frame_number, return all masks on that frame.

        Returns:
            list of dicts (same format as lookup()), sorted by fixture name
        """
        mask_ids = self._frame_index.get((video_id, frame_number), [])
        if not mask_ids:
            return []
        results = []
        for mask_id in mask_ids:
            try:
                results.append(self.lookup(mask_id))
            except Exception as e:
                results.append({"mask_id": mask_id, "error": str(e)})
        return sorted(results, key=lambda x: x.get("fixture", ""))

    def find_by_fixture(self, fixture: str, participant: str = None) -> list:
        """Get all masks for a fixture label, optionally filtered by participant."""
        results = []
        for video_id, mask_id, info in self._fixture_index.get(fixture, []):
            p = video_id.split("-")[0]
            if participant and p != participant:
                continue
            try:
                results.append(self.lookup(mask_id))
            except Exception as e:
                results.append({"mask_id": mask_id, "video_id": video_id, "error": str(e)})
        return sorted(results, key=lambda x: (x.get("video_id", ""), x.get("frame_number", 0)))

    def all_fixtures(self) -> list:
        """List all unique fixture labels in the dataset."""
        return sorted(self._fixture_index.keys())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_mask(result: dict):
    print()
    for k, v in result.items():
        print(f"  {k:<16} {v}")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 hd_epic_lookup.py <mask_id>")
        print("  python3 hd_epic_lookup.py frame_to_timestamp <video_id> <frame_number>")
        print("  python3 hd_epic_lookup.py timestamp_to_frame <video_id> <unix_us>")
        print("  python3 hd_epic_lookup.py masks_at_frame <participant> <video_id> <frame_number>")
        print("  python3 hd_epic_lookup.py find_by_fixture <fixture> [--participant P01]")
        print("  python3 hd_epic_lookup.py all_fixtures")
        sys.exit(1)

    lookup = HDEpicLookup()
    cmd    = sys.argv[1]

    # Plain mask_id lookup
    if cmd not in ("frame_to_timestamp", "timestamp_to_frame",
                   "masks_at_frame", "find_by_fixture", "all_fixtures"):
        try:
            _print_mask(lookup.lookup(cmd))
        except KeyError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    elif cmd == "frame_to_timestamp":
        video_id     = sys.argv[2]
        frame_number = int(sys.argv[3])
        unix_us      = lookup.frame_to_timestamp(video_id, frame_number)
        print(f"\n  frame_number  {frame_number}")
        print(f"  unix_us       {unix_us}")
        print(f"  rgb_filename  {unix_us}.png\n")

    elif cmd == "timestamp_to_frame":
        video_id = sys.argv[2]
        unix_us  = int(sys.argv[3])
        result   = lookup.timestamp_to_frame(video_id, unix_us)
        print()
        for k, v in result.items():
            print(f"  {k:<16} {v}")
        print()

    elif cmd == "masks_at_frame":
        participant  = sys.argv[2]
        video_id     = sys.argv[3]
        frame_number = int(sys.argv[4])
        results      = lookup.masks_at_frame(participant, video_id, frame_number)
        if not results:
            print(f"\n  No masks found for frame {frame_number} in {video_id}\n")
        else:
            print(f"\n  {len(results)} mask(s) on frame {frame_number} in {video_id}")
            for r in results:
                _print_mask(r)

    elif cmd == "find_by_fixture":
        fixture     = sys.argv[2]
        participant = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--participant" else None
        results     = lookup.find_by_fixture(fixture, participant)
        print(f"\n  Found {len(results)} masks for '{fixture}'\n")
        for r in results:
            if "error" in r:
                print(f"  ERROR {r['mask_id']}: {r['error']}")
            else:
                print(f"  {r['participant']}  {r['video_id']}  "
                      f"frame={r['frame_number']}  fixture={r['fixture']}  "
                      f"rgb_exists={r['rgb_exists']}  mask_exists={r['mask_exists']}")
        print()

    elif cmd == "all_fixtures":
        fixtures = lookup.all_fixtures()
        print(f"\n  {len(fixtures)} fixtures:\n")
        for f in fixtures:
            print(f"  {f}")
        print()


if __name__ == "__main__":
    main()
