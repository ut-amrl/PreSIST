#!/usr/bin/env python3
"""
Audit images referenced by train_mix_*.txt manifests for rotation issues.

Two classes of issues are detected:

1. EXIF orientation tag != 1 -- the on-disk pixels look one way but the
   tag asks viewers to re-orient. Loaders that don't call
   `ImageOps.exif_transpose` (e.g. utils/image_utils.py
   loadImageForDatasetGeneration) feed these to the model in their
   un-transposed orientation.

2. "Natively sideways" pixels -- no EXIF, but the file is stored in the
   wrong orientation relative to its capture session. Detected by
   per-deepest-folder aspect-ratio sampling.

Read-only: opens images lazily (no pixel decode) and writes a few report
files to the chosen output directory.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

EXIF_ORIENTATION_TAG = 0x0112
SAMPLE_MIN = 5
SAMPLE_FRACTION = 0.05
UNIFORM_THRESHOLD = 0.95
SAMPLING_SEED = 1234


@dataclass
class ImageRecord:
    rel_path: str
    folder: str
    dataset: str
    width: int
    height: int
    exif_orientation: Optional[int]
    error: Optional[str] = None

    @property
    def is_landscape(self) -> bool:
        return self.width >= self.height


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest-glob",
        default="train_dataset_manifests/train_mix_*.txt",
        help="Glob (relative to CWD) for manifest files to audit.",
    )
    p.add_argument(
        "--dataset-root",
        default="/workspace/data/input",
        help="Root that manifest paths are relative to.",
    )
    p.add_argument(
        "--output-dir",
        default="train_dataset_manifests/rotation_audit",
        help="Directory to write audit reports.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
    )
    return p.parse_args()


def load_manifest_paths(manifest_glob: str) -> List[str]:
    files = sorted(glob.glob(manifest_glob))
    if not files:
        raise FileNotFoundError(f"No manifests matched: {manifest_glob}")
    seen: set = set()
    paths: List[str] = []
    for mf in files:
        with open(mf, "r", encoding="utf-8") as fh:
            for line in fh:
                rel = line.strip()
                if not rel or rel in seen:
                    continue
                seen.add(rel)
                paths.append(rel)
    return paths


def sample_per_folder(paths: List[str]) -> List[str]:
    """Sample max(SAMPLE_MIN, ceil(SAMPLE_FRACTION * N)) per deepest folder.

    Deterministic given SAMPLING_SEED.
    """
    rng = random.Random(SAMPLING_SEED)
    by_folder: Dict[str, List[str]] = defaultdict(list)
    for p in paths:
        by_folder[Path(p).parent.as_posix()].append(p)

    sampled: List[str] = []
    for folder, members in by_folder.items():
        members.sort()
        n = len(members)
        target = max(SAMPLE_MIN, math.ceil(SAMPLE_FRACTION * n))
        if target >= n:
            sampled.extend(members)
        else:
            sampled.extend(rng.sample(members, target))
    return sampled


def _inspect_one(args: Tuple[str, str]) -> ImageRecord:
    rel_path, dataset_root = args
    dataset = rel_path.split("/", 1)[0] if "/" in rel_path else rel_path
    folder = Path(rel_path).parent.as_posix()
    abs_path = os.path.join(dataset_root, rel_path)
    try:
        with Image.open(abs_path) as img:
            w, h = img.size
            try:
                exif = img.getexif()
                ori = exif.get(EXIF_ORIENTATION_TAG) if exif else None
            except Exception:
                ori = None
        return ImageRecord(
            rel_path=rel_path,
            folder=folder,
            dataset=dataset,
            width=int(w),
            height=int(h),
            exif_orientation=int(ori) if ori is not None else None,
        )
    except Exception as e:
        return ImageRecord(
            rel_path=rel_path,
            folder=folder,
            dataset=dataset,
            width=0,
            height=0,
            exif_orientation=None,
            error=f"{type(e).__name__}: {e}",
        )


def inspect_all(
    sampled: List[str],
    dataset_root: str,
    num_workers: int,
) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    work = [(p, dataset_root) for p in sampled]
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for rec in tqdm(
            ex.map(_inspect_one, work, chunksize=32),
            total=len(work),
            desc="Inspecting images",
            unit="img",
        ):
            records.append(rec)
    return records


def classify_folder(records: List[ImageRecord]) -> str:
    valid = [r for r in records if r.error is None]
    if not valid:
        return "no-data"
    n_land = sum(1 for r in valid if r.is_landscape)
    frac = n_land / len(valid)
    if frac >= UNIFORM_THRESHOLD:
        return "uniform-landscape"
    if 1.0 - frac >= UNIFORM_THRESHOLD:
        return "uniform-portrait"
    return "mixed"


def write_reports(
    records: List[ImageRecord],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. EXIF non-trivial.
    exif_path = output_dir / "exif_nontrivial.txt"
    with exif_path.open("w", encoding="utf-8") as fh:
        fh.write("# rel_path<TAB>orientation<TAB>WxH\n")
        for r in records:
            if r.error is not None:
                continue
            if r.exif_orientation in (None, 1):
                continue
            fh.write(f"{r.rel_path}\t{r.exif_orientation}\t{r.width}x{r.height}\n")

    # Group by folder and dataset for the next two reports.
    by_folder: Dict[str, List[ImageRecord]] = defaultdict(list)
    for r in records:
        by_folder[r.folder].append(r)

    folder_class: Dict[str, str] = {f: classify_folder(rs) for f, rs in by_folder.items()}

    by_dataset: Dict[str, List[ImageRecord]] = defaultdict(list)
    for r in records:
        by_dataset[r.dataset].append(r)
    dataset_class: Dict[str, str] = {d: classify_folder(rs) for d, rs in by_dataset.items()}

    # 2. Native sideways candidates: whole-folder flags + in-folder outliers.
    native_path = output_dir / "native_sideways_candidates.txt"
    with native_path.open("w", encoding="utf-8") as fh:
        fh.write("# kind<TAB>rel_path_or_folder<TAB>WxH<TAB>note\n")

        # Whole-folder flags.
        for folder, fcls in sorted(folder_class.items()):
            ds = folder.split("/", 1)[0] if "/" in folder else folder
            ds_cls = dataset_class.get(ds, "no-data")
            if fcls == "uniform-portrait" and ds_cls == "uniform-landscape":
                # Pick a representative size.
                rep = next((r for r in by_folder[folder] if r.error is None), None)
                size = f"{rep.width}x{rep.height}" if rep else "?"
                fh.write(
                    f"folder\t{folder}/\t{size}\t"
                    f"folder uniform-portrait under landscape dataset {ds!r}\n"
                )

        # In-folder outliers within uniform folders.
        for folder, rs in sorted(by_folder.items()):
            fcls = folder_class[folder]
            if fcls == "uniform-landscape":
                for r in rs:
                    if r.error is None and not r.is_landscape:
                        fh.write(
                            f"file\t{r.rel_path}\t{r.width}x{r.height}\t"
                            f"portrait outlier in uniform-landscape folder\n"
                        )
            elif fcls == "uniform-portrait":
                for r in rs:
                    if r.error is None and r.is_landscape:
                        fh.write(
                            f"file\t{r.rel_path}\t{r.width}x{r.height}\t"
                            f"landscape outlier in uniform-portrait folder\n"
                        )

    # 3. Summary.
    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(
            f"# rotation audit summary\n"
            f"# sampling: max({SAMPLE_MIN}, ceil({SAMPLE_FRACTION:.2f} * N)) per deepest folder, seed={SAMPLING_SEED}\n"
            f"# uniform threshold: {UNIFORM_THRESHOLD:.2f}\n"
            f"# total sampled: {len(records)}, errors: {sum(1 for r in records if r.error)}\n\n"
        )

        fh.write("== per-dataset ==\n")
        for ds in sorted(by_dataset):
            rs = by_dataset[ds]
            valid = [r for r in rs if r.error is None]
            n = len(valid)
            n_land = sum(1 for r in valid if r.is_landscape)
            ori_hist = Counter(
                (r.exif_orientation if r.exif_orientation is not None else "none")
                for r in valid
            )
            pct_land = (100.0 * n_land / n) if n else 0.0
            fh.write(
                f"  {ds}: sampled={len(rs)} valid={n} "
                f"%landscape={pct_land:.1f} class={dataset_class[ds]} "
                f"exif={dict(ori_hist)}\n"
            )

        fh.write("\n== per-deepest-folder ==\n")
        for folder in sorted(by_folder):
            rs = by_folder[folder]
            valid = [r for r in rs if r.error is None]
            n = len(valid)
            n_land = sum(1 for r in valid if r.is_landscape)
            ori_hist = Counter(
                (r.exif_orientation if r.exif_orientation is not None else "none")
                for r in valid
            )
            pct_land = (100.0 * n_land / n) if n else 0.0
            fh.write(
                f"  {folder}: sampled={len(rs)} valid={n} "
                f"%landscape={pct_land:.1f} class={folder_class[folder]} "
                f"exif={dict(ori_hist)}\n"
            )

        # Errors section.
        errs = [r for r in records if r.error]
        if errs:
            fh.write(f"\n== errors ({len(errs)}) ==\n")
            for r in errs[:200]:
                fh.write(f"  {r.rel_path}\t{r.error}\n")
            if len(errs) > 200:
                fh.write(f"  ... and {len(errs) - 200} more\n")

    print(f"Wrote: {exif_path}")
    print(f"Wrote: {native_path}")
    print(f"Wrote: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as fh:
        print(fh.read())


def main() -> None:
    args = parse_args()
    paths = load_manifest_paths(args.manifest_glob)
    print(f"Loaded {len(paths)} unique paths from {args.manifest_glob}")

    sampled = sample_per_folder(paths)
    print(
        f"Sampled {len(sampled)} paths across "
        f"{len({Path(p).parent.as_posix() for p in paths})} deepest folders"
    )

    records = inspect_all(sampled, args.dataset_root, args.num_workers)
    write_reports(records, Path(args.output_dir))


if __name__ == "__main__":
    main()
