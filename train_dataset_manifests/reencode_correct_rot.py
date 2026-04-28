#!/usr/bin/env python3
"""
Re-encode every image under each `old_subdir` listed in a rewrites module
into its `new_subdir` counterpart, with EXIF orientation baked into the
pixels (and the EXIF tag stripped). Idempotent.

Designed to be the data-side complement of rewrite_correct_rot.py.

Run from inside the dataset container.

Example:
  python train_dataset_manifests/reencode_correct_rot.py \\
    --rewrites-module openloris_rewrites
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jp2"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rewrites-module",
        required=True,
        help="Module name (importable from train_dataset_manifests/) that "
        "exposes PREFIX and SUBDIR_REWRITES, e.g. openloris_rewrites.",
    )
    p.add_argument("--dataset-root", default="/workspace/data/input")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written without touching disk.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the new dir. Default: skip if "
        "destination already exists.",
    )
    p.add_argument(
        "--new-files-out",
        default=None,
        help="Optional output text file listing the new (rewritten) paths "
        "(rel to dataset_root). Useful for downstream zipping.",
    )
    return p.parse_args()


def load_rewrites(module_name: str):
    # Allow imports like `openloris_rewrites` when run from repo root.
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "PREFIX") or not hasattr(mod, "SUBDIR_REWRITES"):
        raise SystemExit(
            f"ERROR: {module_name} must expose both PREFIX and SUBDIR_REWRITES"
        )
    return mod.PREFIX, mod.SUBDIR_REWRITES


def list_images(folder: Path) -> List[Path]:
    out: List[Path] = []
    if not folder.is_dir():
        return out
    for dirpath, _dirs, files in os.walk(folder):
        for fn in files:
            if Path(fn).suffix.lower() in IMAGE_EXTS:
                out.append(Path(dirpath) / fn)
    return out


def reencode_one(src: Path, dst: Path) -> None:
    """Open src, apply EXIF transpose, save dst with no EXIF metadata."""
    with Image.open(src) as img:
        oriented = ImageOps.exif_transpose(img)
        # Drop any inherited info dict so EXIF isn't carried into the new file.
        oriented.info.pop("exif", None)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Preserve original suffix (and thus format) by letting PIL infer.
        oriented.save(dst)


def main() -> int:
    args = parse_args()
    prefix, rewrites = load_rewrites(args.rewrites_module)
    root = Path(args.dataset_root).resolve()

    new_rels: List[str] = []
    n_total = n_written = n_skipped = n_missing = 0

    for old_sub, new_sub in rewrites.items():
        old_dir = (root / prefix / old_sub).resolve()
        new_dir = (root / prefix / new_sub).resolve()

        files = list_images(old_dir)
        if not files:
            print(f"  WARNING: no images under {old_dir}")
            n_missing += 1
            continue

        print(
            f"\n[{prefix}{old_sub}] -> [{prefix}{new_sub}]: "
            f"{len(files)} image(s)"
        )

        for src in tqdm(files, desc=f"  reencoding {old_sub}", unit="img", leave=False):
            n_total += 1
            rel_inside = src.resolve().relative_to(old_dir).as_posix()
            dst = new_dir / rel_inside

            new_rel = f"{prefix}{new_sub}/{rel_inside}"
            new_rels.append(new_rel)

            if dst.exists() and not args.overwrite:
                n_skipped += 1
                continue

            if args.dry_run:
                continue

            reencode_one(src, dst)
            n_written += 1

    print(
        f"\nTotal {n_total} | wrote {n_written} | skipped (already exist) {n_skipped}"
        f" | missing source dirs {n_missing}"
    )
    if args.dry_run:
        print("(dry run — no files written)")

    if args.new_files_out:
        out = Path(args.new_files_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(sorted(set(new_rels))) + "\n", encoding="utf-8")
        print(f"Wrote new-files manifest: {out} ({len(set(new_rels))} path(s))")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
