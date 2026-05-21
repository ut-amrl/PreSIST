#!/usr/bin/env python3
"""Audit: for every image with persistence labels, check that every
labeled instance_id is also present in the corresponding seg npz's ids.

Labels SHOULD be a subset of seg ids — a label referencing a non-existent
seg id implies upstream pipeline drift (seg results regenerated/dedupe'd
without re-running persistence). Prints offending examples + summaries
without modifying anything.

Usage (defaults match the seg_langSam3Q8T_…_behav4_mO20 setup):
    python3 scripts/audit_label_vs_seg_ids.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from spf_io.persistence_io import readPersistenceData


def label_ids_from_file(p: Path) -> set[int]:
    obj = readPersistenceData(str(p))
    out: set[int] = set()
    if isinstance(obj, list):
        for entry in obj:
            if isinstance(entry, dict) and "id" in entry:
                try:
                    out.add(int(entry["id"]))
                except Exception:
                    pass
    elif isinstance(obj, dict):
        for k in obj.keys():
            try:
                out.add(int(k))
            except Exception:
                pass
    return out


def seg_ids_from_npz(p: Path) -> set[int]:
    with np.load(str(p), allow_pickle=False) as data:
        if "ids" not in data.files:
            return set()
        return {int(x) for x in data["ids"].tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--full-results-dir",
        type=Path,
        default=Path(
            "/workspace/data/output/full_results/"
            "seg_langSam3Q8T_capG31bp7_idG31b3imgL_pG31bv3iqr_only_behav4_mO20"
        ),
    )
    ap.add_argument(
        "--seg-dir",
        type=Path,
        default=Path(
            "/workspace/data/output/segmentation_results/"
            "langGround_p_qwenvl37bT_disc_maxObj30_u3_s_sam3b32__s0p35_mA0p001_dedupe0p9"
        ),
    )
    ap.add_argument("--max-examples", type=int, default=10)
    args = ap.parse_args()

    image_ids = sorted(os.listdir(args.full_results_dir))
    print(f"scanning {len(image_ids)} image_ids")

    n_label_missing_file = 0
    n_seg_missing_file = 0
    n_seg_empty = 0
    n_ok = 0
    n_labels_subset = 0
    n_labels_extra = 0
    examples_extra: list[tuple[str, list[int], int, int]] = []
    diff_size_hist: Counter = Counter()

    for i, iid in enumerate(image_ids):
        label_path = args.full_results_dir / iid / "persistence_parameters.json"
        seg_path = args.seg_dir / iid / "segmentation_results.npz"
        if not label_path.exists():
            n_label_missing_file += 1
            continue
        if not seg_path.exists():
            n_seg_missing_file += 1
            continue
        try:
            seg_ids = seg_ids_from_npz(seg_path)
            label_ids = label_ids_from_file(label_path)
        except Exception as e:
            print(f"  ERROR reading {iid}: {e}", file=sys.stderr)
            continue
        if not seg_ids:
            n_seg_empty += 1
        if not label_ids:
            continue
        extra = label_ids - seg_ids
        if not extra:
            n_labels_subset += 1
            n_ok += 1
        else:
            n_labels_extra += 1
            diff_size_hist[len(extra)] += 1
            if len(examples_extra) < args.max_examples:
                examples_extra.append((iid, sorted(extra), len(seg_ids), len(label_ids)))
        if (i + 1) % 5000 == 0:
            print(f"  ... scanned {i+1}/{len(image_ids)}, mismatch so far: {n_labels_extra}")

    print()
    print(f"summary over {len(image_ids)} image_ids:")
    print(f"  labels file missing:  {n_label_missing_file}")
    print(f"  seg file missing:     {n_seg_missing_file}")
    print(f"  seg npz had no ids:   {n_seg_empty}")
    print(f"  labels ⊆ seg (ok):    {n_labels_subset}")
    print(f"  labels ⊄ seg (BAD):   {n_labels_extra}")
    print(f"  extra-id count hist (n_extra: n_images): {dict(sorted(diff_size_hist.items()))}")

    if examples_extra:
        print(f"\nfirst {len(examples_extra)} bad examples (image_id, extra ids in labels, |seg|, |labels|):")
        for iid, extra, ns, nl in examples_extra:
            print(f"  {iid}  extra={extra}  |seg|={ns}  |labels|={nl}")


if __name__ == "__main__":
    main()
