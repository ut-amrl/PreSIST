#/usr/bin/env python3
"""
extract_hd_epic.py
Extracts frames from HD-EPIC MP4s and generates millisecond epoch timestamps.

Usage:
  python extract_hd_epic.py \
    --input_root /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/Videos \
    --output_root /robodata/<yourdir>/hd-epic-processed \
    --fps 1          # set to None to extract every frame
    --location P01   # optional: only process one location
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
import pytz

RECORDING_TIMEZONE = pytz.timezone("Europe/London")

def filename_to_epoch_ms(stem: str) -> int:
    """
    Convert a stem like 'P01-20240202-110250' to milliseconds since epoch.
    Interprets the datetime as Europe/London local time.
    """
    # stem format: P01-YYYYMMDD-HHMMSS
    parts = stem.split("-")
    date_str = parts[1]  # '20240202'
    time_str = parts[2]  # '110250'
    dt_naive = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    dt_local = RECORDING_TIMEZONE.localize(dt_naive)
    dt_utc = dt_local.astimezone(timezone.utc)
    return int(dt_utc.timestamp() * 1000)


def extract_sequence(mp4_path: str, out_dir: str, target_fps: float = None):
    """Extract frames from an MP4 and write timestamps_ms.txt."""
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)

    # Get native FPS and frame count via ffprobe
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,nb_frames,duration",
        "-of", "json", mp4_path
    ]
    probe_out = subprocess.check_output(probe_cmd).decode()
    probe_data = json.loads(probe_out)["streams"][0]

    native_fps_str = probe_data["r_frame_rate"]  # e.g. "30000/1001"
    num, den = map(int, native_fps_str.split("/"))
    native_fps = num / den

    extract_fps = target_fps if target_fps else native_fps

    # Extract frames with ffmpeg
    frame_pattern = os.path.join(out_dir, "frames", "%06d.jpg")
    vf_filter = f"fps={extract_fps}" if target_fps else "fps=source_fps"
    ffmpeg_cmd = [
        "ffmpeg", "-i", mp4_path,
        "-vf", f"fps={extract_fps}",
        "-q:v", "2",          # high quality JPEG
        frame_pattern, "-y"
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    # Count extracted frames
    frames_dir = os.path.join(out_dir, "frames")
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    num_frames = len(frame_files)

    # Compute timestamps
    stem = os.path.splitext(os.path.basename(mp4_path))[0]
    start_ms = filename_to_epoch_ms(stem)
    frame_interval_ms = 1000.0 / extract_fps

    timestamps = [int(start_ms + i * frame_interval_ms) for i in range(num_frames)]

    ts_path = os.path.join(out_dir, "timestamps_ms.txt")
    with open(ts_path, "w") as f:
        for ts in timestamps:
            f.write(f"{ts}\n")

    # Write metadata
    meta = {
        "source_mp4": mp4_path,
        "native_fps": native_fps,
        "extract_fps": extract_fps,
        "num_frames": num_frames,
        "start_epoch_ms": start_ms,
        "timezone_assumed": str(RECORDING_TIMEZONE),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  → {num_frames} frames, start={start_ms} ms, fps={extract_fps:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--fps", type=float, default=None,
                        help="Target FPS for extraction. None = native FPS.")
    parser.add_argument("--location", default=None,
                        help="Only process this location (e.g. P01)")
    args = parser.parse_args()

    locations = [args.location] if args.location else sorted(os.listdir(args.input_root))

    for loc in locations:
        loc_dir = os.path.join(args.input_root, loc)
        if not os.path.isdir(loc_dir):
            continue
        mp4_files = sorted(f for f in os.listdir(loc_dir) if f.endswith(".mp4"))
        print(f"\n[{loc}] Found {len(mp4_files)} sequences")

        for mp4_file in mp4_files:
            stem = os.path.splitext(mp4_file)[0]
            mp4_path = os.path.join(loc_dir, mp4_file)
            out_dir = os.path.join(args.output_root, loc, stem)
            print(f"  Processing {stem}...")
            extract_sequence(mp4_path, out_dir, target_fps=args.fps)


if __name__ == "__main__":
    main()
