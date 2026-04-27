#!/usr/bin/env python3
"""
convert_hd_epic.py

Copies rectified HD-EPIC frames into the semantic persistence labeler format:

  <output_root>/<participant_id>/<sequence_id>/rgb/<timestamp_us>.png

Source structure (hd-epic-processed):
  <frames_root>/<participant_id>/<sequence_id>/frames_rectified/<N>.jpg
  <frames_root>/<participant_id>/<sequence_id>/timestamps_ms.txt

Each line in timestamps_ms.txt is the epoch timestamp in milliseconds for
frame N (1-indexed filenames: 000001.jpg = line 1).

Timestamps are converted ms → us for filenames to match labeler convention.
Images are re-saved as PNG.

Usage:
  python3 convert_hd_epic.py --participants P01

  # All participants
  python3 convert_hd_epic.py

  # Dry run
  python3 convert_hd_epic.py --participants P01 --dry-run

Defaults:
  --frames-root  /robodata/tarunrav/hd-epic-processed
  --output-root  /robodata/tarunrav/hd-epic-converted
  --workers      8
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import cv2

DEFAULT_FRAMES_ROOT = "/robodata/tarunrav/hd-epic-processed"
DEFAULT_OUTPUT_ROOT = "/robodata/tarunrav/hd-epic-converted"


# ---------------------------------------------------------------------------
# Per-image worker
# ---------------------------------------------------------------------------

def _copy_one(args: Tuple[Path, Path]) -> Tuple[str, bool, str]:
    """Read jpg, write png. Returns (src_name, success, error)."""
    src, dst = args
    try:
        img = cv2.imread(str(src))
        if img is None:
            return (src.name, False, "cv2.imread returned None")
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), img)
        return (src.name, True, "")
    except Exception as e:
        return (src.name, False, str(e))


# ---------------------------------------------------------------------------
# Per-sequence conversion
# ---------------------------------------------------------------------------

def convert_sequence(
    seq_dir: Path,
    output_seq_dir: Path,
    workers: int,
    dry_run: bool,
) -> bool:
    rect_dir = seq_dir / "frames_rectified"
    ts_file = seq_dir / "timestamps_ms.txt"

    if not rect_dir.exists():
        print(f"    [SKIP] No frames_rectified/ in {seq_dir.name}")
        return False
    if not ts_file.exists():
        print(f"    [SKIP] No timestamps_ms.txt in {seq_dir.name}")
        return False

    # Load timestamps — one per line, 1-indexed to match 000001.jpg etc.
    timestamps_ms = []
    with open(ts_file) as f:
        for line in f:
            line = line.strip()
            if line:
                timestamps_ms.append(int(line))

    frame_files = sorted(f for f in rect_dir.glob("*.jpg"))

    if len(frame_files) != len(timestamps_ms):
        print(f"    [WARN] {seq_dir.name}: {len(frame_files)} frames but "
              f"{len(timestamps_ms)} timestamps — using min of the two")

    n = min(len(frame_files), len(timestamps_ms))
    rgb_out = output_seq_dir / "rgb"

    print(f"    {seq_dir.name}: {n} frames → {rgb_out}")

    if dry_run:
        return True

    rgb_out.mkdir(parents=True, exist_ok=True)

    work, skipped = [], 0
    for i in range(n):
        src = frame_files[i]
        ts_us = timestamps_ms[i] * 1000  # ms → us
        dst = rgb_out / f"{ts_us}.png"
        if dst.exists():
            skipped += 1
        else:
            work.append((src, dst))

    if skipped:
        print(f"    {skipped} already converted, skipping.")
    if not work:
        print(f"    Nothing new to do.")
        return True

    failures = []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_copy_one, w): w for w in work}
        for future in as_completed(futures):
            name, ok, err = future.result()
            done += 1
            if not ok:
                failures.append((name, err))
            if done % 100 == 0 or done == len(work):
                print(f"    {done}/{len(work)} ...", end="\r")
    print()

    if failures:
        print(f"    {len(failures)} failures:")
        for name, err in failures[:5]:
            print(f"      {name}: {err}")
    else:
        print(f"    Done. {len(work)} frames written.")

    return True


# ---------------------------------------------------------------------------
# Per-participant conversion
# ---------------------------------------------------------------------------

def convert_participant(
    participant: str,
    frames_root: Path,
    output_root: Path,
    workers: int,
    dry_run: bool,
) -> None:
    participant_dir = frames_root / participant
    if not participant_dir.exists():
        print(f"  [SKIP] Participant dir not found: {participant_dir}")
        return

    sequences = sorted([d for d in participant_dir.iterdir() if d.is_dir()])
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Participant: {participant} "
          f"({len(sequences)} sequences)")

    for seq_dir in sequences:
        output_seq_dir = output_root / participant / seq_dir.name
        convert_sequence(seq_dir, output_seq_dir, workers, dry_run)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert rectified HD-EPIC frames to labeler format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--frames-root", default=DEFAULT_FRAMES_ROOT,
                   help="Root of hd-epic-processed directory.")
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                   help="Output root. Participant IDs become location tags.")
    p.add_argument("--participants", default=None,
                   help="Comma-separated participant IDs (e.g. P01,P02). "
                        "Omit to convert all.")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel worker processes.")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview without writing files.")
    return p.parse_args()


def main():
    args = parse_args()

    frames_root = Path(args.frames_root)
    output_root = Path(args.output_root)

    if not frames_root.exists():
        print(f"ERROR: frames-root not found: {frames_root}")
        sys.exit(1)

    if args.participants:
        participants = [p.strip() for p in args.participants.split(",")]
    else:
        participants = sorted([
            d.name for d in frames_root.iterdir()
            if d.is_dir() and d.name.startswith("P")
        ])
        print(f"Auto-discovered {len(participants)} participants: {participants}")

    print(f"Frames root: {frames_root}")
    print(f"Output root: {output_root}")
    print(f"Workers:     {args.workers}")
    if args.dry_run:
        print("*** DRY RUN — no files will be written ***")

    for participant in participants:
        convert_participant(participant, frames_root, output_root,
                            args.workers, args.dry_run)

    print("\nAll done.")
    if not args.dry_run:
        print(f"\nAnnotation tool location_root per participant:")
        for p in participants:
            print(f"  {p}  →  {output_root / p}")


if __name__ == "__main__":
    main()
