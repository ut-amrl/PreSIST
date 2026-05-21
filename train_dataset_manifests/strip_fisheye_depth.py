#!/usr/bin/env python3
"""Strip fisheye and depth image paths from train manifest files in-place.

Run AFTER manifest construction so the sort+shuffle order in build_manifest.py
is not affected by the removal.

Filters applied to each line:
  - openloris_scene/<scene>/fisheye1/... or fisheye2/...
  - any path containing 'depth' (case-insensitive)

Usage:
    python strip_fisheye_depth.py train_mix_11.txt train_mix_12.txt ...
    python strip_fisheye_depth.py train_mix_*.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FISHEYE_RE = re.compile(r"(^|/)openloris_scene/[^/]+/fisheye[12]/")
DEPTH_RE = re.compile(r"depth", re.IGNORECASE)


def should_drop(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if FISHEYE_RE.search(s):
        return True
    if DEPTH_RE.search(s):
        return True
    return False


def filter_file(path: Path) -> tuple[int, int]:
    original = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [ln for ln in original if not should_drop(ln)]
    dropped = len(original) - len(kept)
    if dropped:
        path.write_text("".join(kept), encoding="utf-8")
    return len(original), dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path, help="Manifest files to filter in-place")
    args = ap.parse_args()

    total_in = total_dropped = 0
    for f in args.files:
        if not f.is_file():
            print(f"skip (not a file): {f}", file=sys.stderr)
            continue
        n_in, n_drop = filter_file(f)
        total_in += n_in
        total_dropped += n_drop
        print(f"{f}: {n_in} -> {n_in - n_drop} (dropped {n_drop})")

    print(f"TOTAL: {total_in} -> {total_in - total_dropped} (dropped {total_dropped})")


if __name__ == "__main__":
    main()
