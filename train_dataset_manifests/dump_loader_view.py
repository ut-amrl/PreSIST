#!/usr/bin/env python3
"""
Dump images the way the LLM-labeling loader sees them.

The loader at utils/image_utils.py:loadImageForDatasetGeneration does
`Image.open(p).convert("RGB")` with NO ImageOps.exif_transpose. Image
viewers and IDEs honor EXIF orientation, so they show files upright even
when the on-disk pixels are sideways. This script bypasses that: it loads
each path the loader's way and saves the result as a fresh PNG (no EXIF),
so what you see in the dump is exactly what the code feeds to the model.

Compare each dumped *_loader.png next to the original in your viewer:
  - same orientation -> code sees it correctly
  - rotated relative to original -> file needs re-encoding

Read-only on the inputs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input_list",
        help="Text file with one rel_path per line (e.g. rotation_audit/inspect_list.txt). "
        "Lines starting with '#' or blank are skipped. Trailing '#' comments are stripped.",
    )
    p.add_argument("--dataset-root", default="/workspace/data/input")
    p.add_argument(
        "--output-dir",
        default="train_dataset_manifests/rotation_audit/loader_view",
        help="Output directory for *_loader.png dumps (mirrors source structure).",
    )
    return p.parse_args()


def iter_paths(list_file: str):
    with open(list_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                yield line


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    n_ok = n_err = 0
    for rel in iter_paths(args.input_list):
        src = os.path.join(args.dataset_root, rel)
        dst = out_root / (Path(rel).with_suffix("").as_posix() + "_loader.png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(src) as img:
                img.convert("RGB").save(dst)
            n_ok += 1
            print(f"OK  {rel}\n -> {dst}")
        except Exception as e:
            n_err += 1
            print(f"ERR {rel}: {type(e).__name__}: {e}")

    print(f"\nDone: {n_ok} dumped, {n_err} errors. Output in {out_root}")


if __name__ == "__main__":
    main()
