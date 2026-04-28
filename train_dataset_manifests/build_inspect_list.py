#!/usr/bin/env python3
"""
Build a curated list of files to manually inspect, based on rotation audit.

Produces train_dataset_manifests/rotation_audit/inspect_list.txt with:
  - all EXIF-non-trivial files (from exif_nontrivial.txt)
  - up to N portrait samples per dataset for mixed-orientation datasets
    (sku110k, coco, sun397)

Read-only: only opens images lazily (no pixel decode).
"""
from __future__ import annotations

import argparse
import glob
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

MIXED_DATASETS = ("sku110k", "coco", "sun397")
PORTRAIT_PER_DATASET = 10
SCAN_CAP_PER_DATASET = 4000  # enough to find 10 portraits with high probability
SEED = 1234


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-glob", default="train_dataset_manifests/train_mix_*.txt")
    p.add_argument("--dataset-root", default="/workspace/data/input")
    p.add_argument(
        "--exif-file",
        default="train_dataset_manifests/rotation_audit/exif_nontrivial.txt",
    )
    p.add_argument(
        "--output",
        default="train_dataset_manifests/rotation_audit/inspect_list.txt",
    )
    p.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    return p.parse_args()


def load_unique_paths(manifest_glob: str) -> List[str]:
    seen, out = set(), []
    for mf in sorted(glob.glob(manifest_glob)):
        with open(mf, "r", encoding="utf-8") as fh:
            for line in fh:
                rel = line.strip()
                if rel and rel not in seen:
                    seen.add(rel)
                    out.append(rel)
    return out


def _is_portrait(args: Tuple[str, str]) -> Tuple[str, bool]:
    rel, root = args
    try:
        with Image.open(os.path.join(root, rel)) as img:
            w, h = img.size
        return rel, h > w
    except Exception:
        return rel, False


def find_portraits(
    candidates: List[str],
    dataset_root: str,
    num_workers: int,
    target: int,
) -> List[str]:
    """Stream candidates through workers, stop once `target` portraits found."""
    work = [(p, dataset_root) for p in candidates]
    found: List[str] = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for rel, is_p in tqdm(
            ex.map(_is_portrait, work, chunksize=32),
            total=len(work),
            desc="Scanning",
            unit="img",
            leave=False,
        ):
            if is_p:
                found.append(rel)
                if len(found) >= target:
                    break
    return found


def load_exif_paths(exif_file: str) -> List[Tuple[str, str, str]]:
    """Returns list of (rel_path, orientation, size)."""
    rows: List[Tuple[str, str, str]] = []
    if not os.path.exists(exif_file):
        return rows
    with open(exif_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))
    return rows


def main() -> None:
    args = parse_args()
    rng = random.Random(SEED)

    paths = load_unique_paths(args.manifest_glob)
    print(f"Loaded {len(paths)} unique manifest paths")

    by_ds: Dict[str, List[str]] = defaultdict(list)
    for p in paths:
        ds = p.split("/", 1)[0]
        by_ds[ds].append(p)

    portraits_by_ds: Dict[str, List[str]] = {}
    for ds in MIXED_DATASETS:
        pool = list(by_ds.get(ds, []))
        rng.shuffle(pool)
        pool = pool[:SCAN_CAP_PER_DATASET]
        print(f"[{ds}] scanning up to {len(pool)} files for {PORTRAIT_PER_DATASET} portraits")
        portraits_by_ds[ds] = find_portraits(
            pool, args.dataset_root, args.num_workers, PORTRAIT_PER_DATASET
        )
        print(f"  -> found {len(portraits_by_ds[ds])} portrait file(s)")

    exif_rows = load_exif_paths(args.exif_file)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("# Files to manually inspect (open in an image viewer).\n")
        fh.write("# Paths are relative to dataset_root (default /workspace/data/input).\n\n")

        fh.write(f"## EXIF non-trivial ({len(exif_rows)} files)\n")
        fh.write("# These have an EXIF orientation tag != 1. The LLM-labeling loader\n")
        fh.write("# does NOT call exif_transpose, so they are fed sideways/rotated.\n")
        fh.write("# orientation=6 means stored 90 CW (needs 90 CCW to upright).\n")
        fh.write("# orientation=0 is invalid/unset (most decoders ignore).\n")
        for rel, ori, size in exif_rows:
            fh.write(f"{rel}\t# orientation={ori} {size}\n")
        fh.write("\n")

        for ds in MIXED_DATASETS:
            files = portraits_by_ds.get(ds, [])
            fh.write(f"## Portrait samples from {ds} ({len(files)} files)\n")
            fh.write(f"# {ds} is mixed-orientation; verify these are natively portrait\n")
            fh.write("# vs. landscape stored sideways.\n")
            for rel in files:
                fh.write(f"{rel}\n")
            fh.write("\n")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
