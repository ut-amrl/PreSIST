#!/usr/bin/env python3
"""Generate a single pseudo-labeling-pipeline manifest with new_pklot relpaths.

For every line in train_mix_{0..15}.txt that starts with `pklot/` (Roboflow
PKLot at 640x640), derive the equivalent new_pklot relpath (original 1280x720
PKLot release, organized by lot/weather/date) and write the deduped result
to a brand-new manifest file. The source train_mix_*.txt files are read only;
they are NEVER modified.

Output: train_dataset_manifests/train_mix_new_pklot.txt

The output is meant to be fed to the pseudo-labeling pipeline so we can run
segmentation/captioning/persistence-querying on the higher-resolution PKLot
images later.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parent.parent

# Roboflow filenames: 2012-09-11_15_16_58_jpg.rf.<hex>.jpg
ROBOFLOW_STEM_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})_jpg\.rf\.[0-9a-f]+\.jpg$"
)
# Original PKLot filenames: 2012-09-11_15_16_58.jpg
ORIG_STEM_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})\.jpg$")


def build_stem_to_new_pklot_relpath(orig_root: Path) -> Dict[str, str]:
    """Walk <orig_root>/<LOT>/<weather>/<date>/<stem>.jpg → {stem -> new_pklot relpath}.

    Output relpath is rooted at the input_root_dir layout, i.e.:
        new_pklot/PKLot/PKLot/<LOT>/<weather>/<date>/<stem>.jpg
    """
    out: Dict[str, str] = {}
    # Layout on disk is: <orig_root>/<LOT>/<weather>/<date>/<file>.jpg
    # The relpath we want under input_root is
    #   new_pklot/PKLot/PKLot/<LOT>/<weather>/<date>/<file>.jpg
    # so the prefix that precedes <LOT> in the relpath is "new_pklot/PKLot/PKLot".
    relpath_prefix = "new_pklot/PKLot/PKLot"
    for jpg in orig_root.rglob("*.jpg"):
        m = ORIG_STEM_RE.match(jpg.name)
        if not m:
            continue
        stem = m.group(1)
        rel_under_root = jpg.relative_to(orig_root).as_posix()
        out[stem] = f"{relpath_prefix}/{rel_under_root}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--train-mix-dir",
        type=Path,
        default=REPO_ROOT / "train_dataset_manifests",
        help="Directory containing train_mix_*.txt (read-only).",
    )
    ap.add_argument(
        "--train-mix-glob",
        default="train_mix_*.txt",
        help="Glob pattern for source manifests (default: train_mix_*.txt). "
             "Files named backup_* are always skipped.",
    )
    ap.add_argument(
        "--orig-pklot-root",
        type=Path,
        default=Path("/workspace/data/input/new_pklot/PKLot/PKLot"),
        help="Root of the original PKLot tree (contains PUCPR/UFPR04/UFPR05). "
             "Default is the in-container mount path; override with the host path "
             "(/robodata/.../new_pklot/PKLot/PKLot) when running outside the container.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "train_dataset_manifests" / "train_mix_new_pklot.txt",
        help="Output manifest path.",
    )
    args = ap.parse_args()

    print(f"[1/3] Building stem→new_pklot map from {args.orig_pklot_root}", flush=True)
    stem_to_new_pklot = build_stem_to_new_pklot_relpath(args.orig_pklot_root)
    print(f"   unique stems found: {len(stem_to_new_pklot)}")

    print(f"[2/3] Scanning train_mix files in {args.train_mix_dir}", flush=True)
    pklot_lines_total = 0
    unknown_format = 0
    no_counterpart = 0
    seen_stems: set[str] = set()
    new_pklot_relpaths: list[str] = []

    for f in sorted(args.train_mix_dir.glob(args.train_mix_glob)):
        if f.name.startswith("backup_"):
            continue
        for ln in f.read_text().splitlines():
            s = ln.strip()
            if not s or s.startswith("#") or not s.startswith("pklot/"):
                continue
            pklot_lines_total += 1
            m = ROBOFLOW_STEM_RE.match(Path(s).name)
            if not m:
                unknown_format += 1
                continue
            stem = m.group(1)
            new_rel = stem_to_new_pklot.get(stem)
            if new_rel is None:
                no_counterpart += 1
                continue
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            new_pklot_relpaths.append(new_rel)

    new_pklot_relpaths.sort()
    print(f"   pklot/ lines read: {pklot_lines_total}")
    print(f"   unique stems mapped: {len(new_pklot_relpaths)}")
    print(f"   unknown filename format (skipped): {unknown_format}")
    print(f"   no new_pklot counterpart (skipped): {no_counterpart}")

    print(f"[3/3] Writing {args.output}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(new_pklot_relpaths) + "\n")
    print(f"   wrote {len(new_pklot_relpaths)} lines")


if __name__ == "__main__":
    main()
