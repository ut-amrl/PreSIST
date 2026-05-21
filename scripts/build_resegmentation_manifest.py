#!/usr/bin/env python3
"""Build a re-segmentation manifest of relpaths drawn from the train_mix
sources that either (a) have no segmentation_results.npz at all, or
(b) have a seg npz but it contains zero masks.

Writes a single combined manifest plus a small report.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import numpy.lib.format as npy_fmt


REPO_ROOT = Path(__file__).resolve().parent.parent


def n_masks_in_seg_npz(path: Path) -> int:
    """Read just the masks.npy header inside the .npz to get the mask
    count without decompressing the full payload (~10x faster than np.load).
    """
    with zipfile.ZipFile(str(path)) as z:
        if "masks.npy" not in z.namelist():
            return 0
        with z.open("masks.npy") as f:
            version = npy_fmt.read_magic(f)
            shape, _fortran, _dtype = npy_fmt._read_array_header(f, version)
            return int(shape[0]) if len(shape) >= 1 else 0


def read_train_mix_relpaths(train_mix_dir: Path, glob: str) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for f in sorted(train_mix_dir.glob(glob)):
        if f.name.startswith("backup_"):
            continue
        for ln in f.read_text().splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--train-mix-dir",
        type=Path,
        default=REPO_ROOT / "train_dataset_manifests",
    )
    ap.add_argument(
        "--train-mix-glob",
        default="train_mix_[0-9]*.txt",
        help="Only the numbered train_mix_<N>.txt files by default — excludes "
             "train_mix_new_pklot.txt, train_mix_empty_seg.txt, etc.",
    )
    ap.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("/workspace/data/output/registry/exports/path_to_id.csv"),
    )
    ap.add_argument(
        "--registry-root",
        default="/workspace/data/input",
    )
    ap.add_argument(
        "--seg-dir",
        type=Path,
        default=Path(
            "/workspace/data/output/segmentation_results/"
            "langGround_p_qwenvl37bT_disc_maxObj30_u3_s_sam3b32__s0p35_mA0p001_dedupe0p9"
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "train_dataset_manifests" / "train_mix_needs_segmentation.txt",
    )
    args = ap.parse_args()

    print(f"[1/4] Reading train_mix files from {args.train_mix_dir}", flush=True)
    relpaths = read_train_mix_relpaths(args.train_mix_dir, args.train_mix_glob)
    print(f"   unique train_mix relpaths: {len(relpaths)}")

    print(f"[2/4] Loading relpath→image_id from {args.registry_csv}", flush=True)
    rel2id: Dict[str, str] = {}
    with args.registry_csv.open() as f:
        for row in csv.DictReader(f):
            if row["root"] == args.registry_root:
                rel2id.setdefault(row["relpath"], row["image_id"])
    print(f"   registry rows (root={args.registry_root}): {len(rel2id)}")

    print(f"[3/4] Checking seg_npz existence + content under {args.seg_dir}", flush=True)
    no_image_id: List[str] = []
    no_seg_file: List[str] = []
    empty_seg: List[str] = []
    ok = 0

    for i, rp in enumerate(relpaths):
        iid = rel2id.get(rp)
        if iid is None:
            no_image_id.append(rp)
            continue
        p = args.seg_dir / iid / "segmentation_results.npz"
        if not p.exists():
            no_seg_file.append(rp)
            continue
        try:
            n = n_masks_in_seg_npz(p)
        except Exception as e:
            print(f"   WARN: failed to load {p}: {e}", file=sys.stderr)
            empty_seg.append(rp)
            continue
        if n == 0:
            empty_seg.append(rp)
        else:
            ok += 1
        if (i + 1) % 5000 == 0:
            print(f"   ... {i+1}/{len(relpaths)}")

    print()
    print(f"summary over {len(relpaths)} train_mix relpaths:")
    print(f"  not in registry (no image_id):   {len(no_image_id)}")
    print(f"  no seg npz file (never run):     {len(no_seg_file)}")
    print(f"  seg npz exists but is empty:     {len(empty_seg)}")
    print(f"  seg has masks (ok):              {ok}")

    combined = sorted(set(no_seg_file) | set(empty_seg))
    print(f"\n  → re-segmentation candidates (no-file ∪ empty): {len(combined)}")

    print(f"\n[4/4] Writing {args.output}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(combined) + ("\n" if combined else ""))
    print(f"   wrote {len(combined)} relpaths")

    if no_image_id:
        print(f"\nfirst {min(5, len(no_image_id))} no-image_id (likely renamed/moved):")
        for r in no_image_id[:5]:
            print(f"  {r}")


if __name__ == "__main__":
    main()
