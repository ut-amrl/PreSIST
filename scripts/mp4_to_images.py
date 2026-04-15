#!/usr/bin/env python3
"""
Convert MP4 videos to folders of images, preserving directory structure.

For each MP4 found under <input_dir>, frames are written to:
    <output_dir>/<relative_parent>/<video_stem>/

Frame naming:
  - If the filename contains a parseable timestamp (YYYYMMDD-HHMMSS) AND
    ffprobe can determine the FPS, frames are named <millis_since_epoch>.jpg
    where the epoch is computed as (timestamp_in_name + frame_offset).
  - Otherwise, frames are named as zero-padded frame numbers (000001.jpg, ...).

Usage:
    python mp4_to_images.py <input_dir> <output_dir> [--ext {jpg,png}] [--utc]

Example:
    python scripts/mp4_to_images.py \
        /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/HD-EPIC/Videos/ \
        /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/hd-epic/images_only/

Arguments:
    input_dir   Root directory to search for .mp4 files (recursive)
    output_dir  Root directory where image folders will be written
    --ext       Output image format: jpg (default) or png
    --utc       Treat timestamps in filenames as UTC (default: local time)
    --skip-existing
                Skip videos whose output folder already has images (default: True)
    --overwrite Overwrite existing output folders
"""

import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Matches YYYYMMDD-HHMMSS anywhere in the filename stem.
# Example: P01-20240202-110250  →  date=20240202, time=110250
TIMESTAMP_RE = re.compile(r'(\d{8})-(\d{6})')


def parse_stem_timestamp(stem: str, assume_utc: bool) -> Optional[datetime]:
    """Return a timezone-aware datetime parsed from the filename stem, or None."""
    match = TIMESTAMP_RE.search(stem)
    if not match:
        return None
    date_part, time_part = match.group(1), match.group(2)
    try:
        dt_naive = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")
    except ValueError:
        return None
    tz = timezone.utc if assume_utc else datetime.now(timezone.utc).astimezone().tzinfo
    return dt_naive.replace(tzinfo=tz)


def get_video_fps(video_path: Path) -> Optional[float]:
    """Return the video FPS via ffprobe, or None on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        fps_str = result.stdout.strip()
        if not fps_str:
            return None
        if "/" in fps_str:
            num, den = fps_str.split("/", 1)
            den_val = float(den)
            if den_val == 0:
                return None
            return float(num) / den_val
        return float(fps_str)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None


def extract_frames_sequential(video_path: Path, output_dir: Path, ext: str) -> list[dict]:
    """
    Extract frames as 000001.ext, 000002.ext, ...
    Returns a list of row dicts for the CSV (epoch_ms and timestamp_iso are empty).
    """
    out_pattern = str(output_dir / f"%06d.{ext}")
    _run_ffmpeg(video_path, out_pattern, ext)
    rows = [
        {"frame_id": p.stem, "epoch_ms": "", "timestamp_iso": ""}
        for p in sorted(output_dir.glob(f"*.{ext}"))
    ]
    return rows


def extract_frames_epoch_ms(
    video_path: Path,
    output_dir: Path,
    start_dt: datetime,
    fps: float,
    ext: str,
) -> list[dict]:
    """
    Extract frames named by milliseconds-since-epoch.
    Strategy: extract to numbered temp files, then rename.
    Returns a list of row dicts for the CSV.
    """
    tmp_pattern = str(output_dir / f"_tmp_%06d.{ext}")
    _run_ffmpeg(video_path, tmp_pattern, ext)

    tmp_frames = sorted(output_dir.glob(f"_tmp_*.{ext}"))
    start_epoch_ms = int(start_dt.timestamp() * 1000)
    rows = []

    for frame_path in tmp_frames:
        try:
            frame_index = int(frame_path.stem.replace("_tmp_", ""))
        except ValueError:
            frame_path.rename(output_dir / frame_path.name.replace("_tmp_", ""))
            continue
        offset_ms = round((frame_index - 1) / fps * 1000)
        epoch_ms = start_epoch_ms + offset_ms
        frame_path.rename(output_dir / f"{epoch_ms}.{ext}")
        frame_dt = datetime.fromtimestamp(epoch_ms / 1000, tz=start_dt.tzinfo)
        rows.append({
            "frame_id": str(epoch_ms),
            "epoch_ms": epoch_ms,
            "timestamp_iso": frame_dt.isoformat(),
        })

    return rows


def write_csv(csv_path: Path, rows: list[dict]):
    """Write a CSV mapping frame_id → epoch_ms, timestamp_iso next to the image folder."""
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "epoch_ms", "timestamp_iso"])
        writer.writeheader()
        writer.writerows(rows)


def _run_ffmpeg(video_path: Path, out_pattern: str, ext: str):
    """Run ffmpeg to extract all frames. Raises subprocess.CalledProcessError on failure."""
    quality_args = ["-q:v", "2"] if ext == "jpg" else []
    cmd = [
        "ffmpeg", "-i", str(video_path),
        *quality_args,
        out_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )


def convert_videos(
    input_root: Path,
    output_root: Path,
    ext: str,
    assume_utc: bool,
    overwrite: bool,
):
    mp4_files = sorted(input_root.rglob("*.mp4"))
    if not mp4_files:
        print(f"No .mp4 files found under {input_root}")
        return

    print(f"Found {len(mp4_files)} .mp4 file(s)\n")

    for video_path in mp4_files:
        rel_path = video_path.relative_to(input_root)
        output_dir = output_root / rel_path.parent / video_path.stem

        # Skip / overwrite logic
        if output_dir.exists() and any(output_dir.glob(f"*.{ext}")):
            if not overwrite:
                print(f"[SKIP]    {rel_path}  (output exists, use --overwrite to re-extract)")
                continue
            else:
                print(f"[OVERWRITE] {rel_path}")
                for f in output_dir.glob(f"*.{ext}"):
                    f.unlink()
                for f in output_dir.glob(f"_tmp_*.{ext}"):
                    f.unlink()
                old_csv = output_dir.parent / f"{video_path.stem}.csv"
                if old_csv.exists():
                    old_csv.unlink()
        else:
            print(f"[PROCESS] {rel_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        start_dt = parse_stem_timestamp(video_path.stem, assume_utc)
        fps: Optional[float] = None
        if start_dt is not None:
            fps = get_video_fps(video_path)

        try:
            if start_dt is not None and fps is not None:
                print(f"  Timestamp: {start_dt.isoformat()}, FPS: {fps:.4f} → epoch-ms filenames")
                rows = extract_frames_epoch_ms(video_path, output_dir, start_dt, fps, ext)
            else:
                reason = "no timestamp in filename" if start_dt is None else "FPS unknown"
                print(f"  {reason} → sequential frame numbers")
                rows = extract_frames_sequential(video_path, output_dir, ext)

            csv_path = output_dir.parent / f"{video_path.stem}.csv"
            write_csv(csv_path, rows)
            print(f"  {len(rows)} frames → {output_dir}")
            print(f"  CSV        → {csv_path}\n")

        except subprocess.CalledProcessError as exc:
            print(
                f"  ERROR: ffmpeg failed for {video_path}\n"
                f"  stderr: {exc.stderr[-500:] if exc.stderr else '(none)'}",
                file=sys.stderr,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Extract MP4 videos to image folders, preserving directory structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir", type=Path,
                        help="Root directory containing .mp4 files")
    parser.add_argument("output_dir", type=Path,
                        help="Root directory for output image folders")
    parser.add_argument("--ext", default="jpg", choices=["jpg", "png"],
                        help="Output image format (default: jpg)")
    parser.add_argument("--utc", action="store_true",
                        help="Treat timestamps in filenames as UTC (default: local timezone)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract frames even if output folder already has images")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: input directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    convert_videos(
        input_root=args.input_dir,
        output_root=args.output_dir,
        ext=args.ext,
        assume_utc=args.utc,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
