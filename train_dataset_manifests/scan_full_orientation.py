#!/usr/bin/env python3
"""
Full (non-sampled) EXIF orientation scan over a subtree of dataset_root.

Walks every image file under one or more subtrees and records its EXIF
orientation tag. Use this when the per-folder sampler in
audit_manifest_rotation.py might have missed an isolated bad file.

Output: one CSV-like text file with rel_path, orientation, WxH for files
whose orientation tag is anything other than 1 (the upright / no-op value).

Read-only.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

EXIF_ORIENTATION_TAG = 0x0112
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jp2"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-root",
        default="/workspace/data/input",
        help="Dataset root. Subtrees are interpreted relative to this.",
    )
    p.add_argument(
        "--subtree",
        action="append",
        required=True,
        help="One or more subtrees (relative to --dataset-root) to scan. "
        "Repeat the flag for multiple. Example: --subtree openloris_location",
    )
    p.add_argument(
        "--output",
        default="train_dataset_manifests/rotation_audit/full_scan_nontrivial.txt",
    )
    p.add_argument(
        "--summary",
        default="train_dataset_manifests/rotation_audit/full_scan_summary.txt",
    )
    p.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    return p.parse_args()


def walk_images(root: Path, subtrees: Iterable[str]) -> List[str]:
    """Return rel-paths (relative to root) of all images under each subtree."""
    rels: List[str] = []
    for sub in subtrees:
        base = (root / sub).resolve()
        if not base.is_dir():
            print(f"WARNING: subtree not a dir: {base}")
            continue
        for dirpath, _dirnames, filenames in os.walk(base):
            for fn in filenames:
                if Path(fn).suffix.lower() in IMAGE_EXTS:
                    abs_p = Path(dirpath) / fn
                    rel = abs_p.resolve().relative_to(root).as_posix()
                    rels.append(rel)
    return rels


def _probe(args: Tuple[str, str]) -> Tuple[str, int, int, Optional[int], Optional[str]]:
    rel, root = args
    abs_p = os.path.join(root, rel)
    try:
        with Image.open(abs_p) as img:
            w, h = img.size
            try:
                exif = img.getexif()
                ori = exif.get(EXIF_ORIENTATION_TAG) if exif else None
            except Exception:
                ori = None
        return rel, int(w), int(h), int(ori) if ori is not None else None, None
    except Exception as e:
        return rel, 0, 0, None, f"{type(e).__name__}: {e}"


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    rels = walk_images(root, args.subtree)
    print(f"Found {len(rels)} image file(s) under {args.subtree}")

    work = [(r, str(root)) for r in rels]
    rows: List[Tuple[str, int, int, Optional[int], Optional[str]]] = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        for row in tqdm(
            ex.map(_probe, work, chunksize=64),
            total=len(work),
            desc="Probing",
            unit="img",
        ):
            rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nontrivial = [r for r in rows if r[4] is None and r[3] not in (None, 1)]
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("# rel_path\torientation\tWxH\n")
        for rel, w, h, ori, _err in nontrivial:
            fh.write(f"{rel}\t{ori}\t{w}x{h}\n")

    summary_path = Path(args.summary)
    by_folder: dict = defaultdict(Counter)
    err_rows = [r for r in rows if r[4] is not None]
    for rel, _w, _h, ori, _err in rows:
        folder = Path(rel).parent.as_posix()
        by_folder[folder][ori if ori is not None else "none"] += 1

    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# full scan summary, total={len(rows)}, errors={len(err_rows)}\n")
        fh.write(f"# nontrivial (ori not in {{1, None}}): {len(nontrivial)}\n\n")
        fh.write("== per-folder orientation histogram ==\n")
        for folder in sorted(by_folder):
            hist = dict(by_folder[folder])
            nt = sum(c for k, c in hist.items() if k not in (1, "none"))
            tag = " *** NONTRIVIAL ***" if nt > 0 else ""
            fh.write(f"  {folder}: {hist}{tag}\n")
        if err_rows:
            fh.write(f"\n== errors ({len(err_rows)}) ==\n")
            for rel, _w, _h, _o, err in err_rows[:200]:
                fh.write(f"  {rel}\t{err}\n")

    print(f"Wrote: {out_path} ({len(nontrivial)} nontrivial files)")
    print(f"Wrote: {summary_path}")
    # Also echo folders with nontrivial orientation values to stdout.
    nontrivial_folders = sorted({Path(r[0]).parent.as_posix() for r in nontrivial})
    if nontrivial_folders:
        print("\nFolders with at least one nontrivial-orientation file:")
        for f in nontrivial_folders:
            print(f"  {f}")


if __name__ == "__main__":
    main()
