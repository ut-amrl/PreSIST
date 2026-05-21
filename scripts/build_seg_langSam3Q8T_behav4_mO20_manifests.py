#!/usr/bin/env python3
"""Build train/val manifests for the cross-attention persistence trainer
against the `seg_langSam3Q8T_capG31bp7_idG31b3imgL_pG31bv3iqr_only_behav4_mO20`
results.

Algorithm:
  1. Cross-reference labeled image_ids (under full_results_dir) with the
     registry path map to get the relpath universe.
  2. Hard-drop fisheye/depth lines.
  3. Random 85/15 split (seeded).
  4. Pairwise swap A: train PUCPR ↔ val UFPR04+UFPR05.
     Move min(N_train_PUCPR, N_val_UFPR*) each way; drop surplus train PUCPR.
  5. Pairwise swap B: train exmaps ↔ val sku110k.
     Move min(N_train_exmaps, N_val_sku) each way; drop surplus train exmaps.
  6. Restrict val PUCPR to the 50/20/30 train window
     (2012-09-11..2012-10-15); drop out-of-window from val.
  7. Restrict val exmaps to non-June; drop June from val.
  8. Assert disjointness. Write manifests. Print per-source histograms.

Outputs:
  train_dataset_manifests/seg_langSam3Q8T_behav4_mO20/train.txt
  train_dataset_manifests/seg_langSam3Q8T_behav4_mO20/val.txt
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Set


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "train_dataset_manifests"))
from strip_fisheye_depth import FISHEYE_RE, DEPTH_RE  # noqa: E402


# Roboflow filenames: YYYY-MM-DD_HH_MM_SS_jpg.rf.<hex>.jpg
ROBOFLOW_STEM_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})_jpg\.rf\.[0-9a-f]+\.jpg$"
)
# Original PKLot filenames: YYYY-MM-DD_HH_MM_SS.jpg
ORIG_STEM_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})\.jpg$")
# exmaps date dir: exmaps/YYYY-MM-DD/...
EXMAPS_DATE_RE = re.compile(r"^exmaps/(\d{4})-(\d{2})-\d{2}/")

# PUCPR 50/20/30 train-split window (inclusive) — matches the existing
# new_pklot/postprocessed/.../PUCPR_repaired_split_flanks_fixed/splits/50_20_30/
# train_through_2012-10-15.json (split_metadata: min_date 2012-09-11,
# train_end_date 2012-10-15).
PUCPR_WINDOW_START = date(2012, 9, 11)
PUCPR_WINDOW_END = date(2012, 10, 15)

JUNE_MONTH = 6


def build_pklot_stem_to_lot(orig_root: Path) -> Dict[str, str]:
    """Walk <orig_root>/<LOT>/<weather>/<date>/<stem>.jpg → {stem: lot}."""
    out: Dict[str, str] = {}
    for jpg in orig_root.rglob("*.jpg"):
        m = ORIG_STEM_RE.match(jpg.name)
        if not m:
            continue
        lot = jpg.relative_to(orig_root).parts[0]
        out[m.group(1)] = lot
    return out


def pklot_stem_and_date(relpath: str) -> tuple[str, date] | None:
    """Parse Roboflow `pklot/<...>/<YYYY-MM-DD_HH_MM_SS>_jpg.rf.<hex>.jpg`."""
    name = Path(relpath).name
    m = ROBOFLOW_STEM_RE.match(name)
    if not m:
        return None
    stem = m.group(1)
    y, mo, d = int(stem[0:4]), int(stem[5:7]), int(stem[8:10])
    return stem, date(y, mo, d)


def classify(
    relpath: str,
    pklot_stem_to_lot: Dict[str, str],
) -> str:
    """Return a category tag for splitting/swap purposes.

    Categories:
      'fisheye_depth'      - drop entirely
      'pklot_pucpr_in'     - PUCPR pklot inside 50/20/30 train window
      'pklot_pucpr_out'    - PUCPR pklot outside that window
      'pklot_ufpr'         - UFPR04 or UFPR05 pklot
      'pklot_unknown'      - pklot but stem not found in original-PKLot index
      'exmaps_june'        - exmaps in June (any year)
      'exmaps_non_june'    - exmaps any other month
      'sku110k'            - sku110k
      'other'              - everything else (CODa, kitti, EPIC-KITCHENS, ...)
    """
    if FISHEYE_RE.search(relpath) or DEPTH_RE.search(relpath):
        return "fisheye_depth"
    if relpath.startswith("pklot/"):
        parsed = pklot_stem_and_date(relpath)
        if parsed is None:
            return "pklot_unknown"
        stem, d = parsed
        lot = pklot_stem_to_lot.get(stem)
        if lot == "PUCPR":
            if PUCPR_WINDOW_START <= d <= PUCPR_WINDOW_END:
                return "pklot_pucpr_in"
            return "pklot_pucpr_out"
        if lot in ("UFPR04", "UFPR05"):
            return "pklot_ufpr"
        return "pklot_unknown"
    m = EXMAPS_DATE_RE.match(relpath)
    if m:
        if int(m.group(2)) == JUNE_MONTH:
            return "exmaps_june"
        return "exmaps_non_june"
    if relpath.startswith("sku110k/"):
        return "sku110k"
    return "other"


def source_prefix(relpath: str) -> str:
    return relpath.split("/", 1)[0]


def read_relpath_to_image_id(csv_path: Path, registry_root: str) -> Dict[str, str]:
    """Load relpath -> image_id from path_to_id.csv, filtered to registry_root."""
    out: Dict[str, str] = {}
    with csv_path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row["root"] != registry_root:
                continue
            # If a relpath appears multiple times, keep the first (stable).
            out.setdefault(row["relpath"], row["image_id"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("/workspace/data/output/registry/exports/path_to_id.csv"),
    )
    ap.add_argument(
        "--registry-root",
        default="/workspace/data/input",
        help="Filter path_to_id.csv to rows whose `root` matches this.",
    )
    ap.add_argument(
        "--full-results-dir",
        type=Path,
        default=Path(
            "/workspace/data/output/full_results/"
            "seg_langSam3Q8T_capG31bp7_idG31b3imgL_pG31bv3iqr_only_behav4_mO20"
        ),
    )
    ap.add_argument(
        "--pklot-orig-root",
        type=Path,
        default=Path("/workspace/data/input/new_pklot/PKLot/PKLot"),
        help="Root of original PKLot tree (contains PUCPR/UFPR04/UFPR05).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "train_dataset_manifests" / "seg_langSam3Q8T_behav4_mO20",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Initial random val fraction before swaps (default 0.10 = 90/10).",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f"[1/6] Building pklot stem→lot map from {args.pklot_orig_root}", flush=True)
    stem_to_lot = build_pklot_stem_to_lot(args.pklot_orig_root)
    print(f"   unique stems: {len(stem_to_lot)} "
          f"(lots: {dict(Counter(stem_to_lot.values()))})")

    print(f"[2/6] Reading relpath→image_id from {args.registry_csv}", flush=True)
    relpath_to_id = read_relpath_to_image_id(args.registry_csv, args.registry_root)
    print(f"   registry rows (root={args.registry_root}): {len(relpath_to_id)}")

    print(f"[3/6] Listing labeled image_ids under {args.full_results_dir}", flush=True)
    labeled_ids: Set[str] = set(os.listdir(args.full_results_dir))
    print(f"   labeled image_ids: {len(labeled_ids)}")

    # Universe = labeled images that we can resolve to a relpath.
    universe: List[str] = sorted(
        rp for rp, iid in relpath_to_id.items() if iid in labeled_ids
    )
    print(f"   labeled relpaths resolved: {len(universe)}")

    print("[4/6] Classifying and applying hard drops", flush=True)
    categories: Dict[str, List[str]] = defaultdict(list)
    for rp in universe:
        categories[classify(rp, stem_to_lot)].append(rp)
    cat_counts = {k: len(v) for k, v in categories.items()}
    print(f"   per-category counts: {cat_counts}")

    # Hard drop fisheye/depth.
    dropped_fisheye_depth = categories.pop("fisheye_depth", [])
    # pklot_unknown is unexpected; keep but warn.
    if categories.get("pklot_unknown"):
        print(f"   WARNING: {len(categories['pklot_unknown'])} pklot lines with "
              f"unknown stems (treating as 'other')")
        categories["other"].extend(categories.pop("pklot_unknown"))

    # Pool of items that participate in random 85/15:
    all_items: List[str] = []
    for v in categories.values():
        all_items.extend(v)

    train_pct = int(round((1.0 - args.val_ratio) * 100))
    val_pct = 100 - train_pct
    print(f"[5/6] Random {train_pct}/{val_pct} split and applying swap rules", flush=True)
    rng.shuffle(all_items)
    n_val_initial = int(round(args.val_ratio * len(all_items)))
    val_set: Set[str] = set(all_items[:n_val_initial])
    train_set: Set[str] = set(all_items[n_val_initial:])
    assert val_set.isdisjoint(train_set)
    print(f"   initial: train={len(train_set)} val={len(val_set)}")

    def by_cat(items: Set[str], cat: str) -> List[str]:
        return [rp for rp in items if classify(rp, stem_to_lot) == cat]

    # ---- Swap A: train PUCPR (in OR out of window) ↔ val UFPR* ----
    train_pucpr = by_cat(train_set, "pklot_pucpr_in") + by_cat(train_set, "pklot_pucpr_out")
    val_ufpr = by_cat(val_set, "pklot_ufpr")
    rng.shuffle(train_pucpr)
    rng.shuffle(val_ufpr)
    n_swap_a = min(len(train_pucpr), len(val_ufpr))

    moved_pucpr_to_val = train_pucpr[:n_swap_a]
    moved_ufpr_to_train = val_ufpr[:n_swap_a]
    dropped_train_pucpr_surplus = train_pucpr[n_swap_a:]

    train_set.difference_update(moved_pucpr_to_val)
    train_set.difference_update(dropped_train_pucpr_surplus)
    val_set.difference_update(moved_ufpr_to_train)
    val_set.update(moved_pucpr_to_val)
    train_set.update(moved_ufpr_to_train)
    print(f"   swap A: moved {n_swap_a} PUCPR→val, {n_swap_a} UFPR*→train, "
          f"dropped {len(dropped_train_pucpr_surplus)} surplus train PUCPR")

    # ---- Swap B: train exmaps (any month) ↔ val sku110k ----
    train_exmaps = by_cat(train_set, "exmaps_non_june") + by_cat(train_set, "exmaps_june")
    val_sku = by_cat(val_set, "sku110k")
    rng.shuffle(train_exmaps)
    rng.shuffle(val_sku)
    n_swap_b = min(len(train_exmaps), len(val_sku))

    moved_exmaps_to_val = train_exmaps[:n_swap_b]
    moved_sku_to_train = val_sku[:n_swap_b]
    dropped_train_exmaps_surplus = train_exmaps[n_swap_b:]

    train_set.difference_update(moved_exmaps_to_val)
    train_set.difference_update(dropped_train_exmaps_surplus)
    val_set.difference_update(moved_sku_to_train)
    val_set.update(moved_exmaps_to_val)
    train_set.update(moved_sku_to_train)
    print(f"   swap B: moved {n_swap_b} exmaps→val, {n_swap_b} sku110k→train, "
          f"dropped {len(dropped_train_exmaps_surplus)} surplus train exmaps")

    # ---- Val restrictions ----
    val_pucpr_out = by_cat(val_set, "pklot_pucpr_out")
    val_set.difference_update(val_pucpr_out)
    print(f"   dropped {len(val_pucpr_out)} val PUCPR-out-of-window")

    val_exmaps_june = by_cat(val_set, "exmaps_june")
    val_set.difference_update(val_exmaps_june)
    print(f"   dropped {len(val_exmaps_june)} val exmaps-June")

    # ---- Sanity checks ----
    assert val_set.isdisjoint(train_set), "train and val overlap"
    train_cats = Counter(classify(rp, stem_to_lot) for rp in train_set)
    val_cats = Counter(classify(rp, stem_to_lot) for rp in val_set)
    forbidden_train = ("pklot_pucpr_in", "pklot_pucpr_out", "exmaps_non_june", "exmaps_june")
    forbidden_val = ("pklot_pucpr_out", "exmaps_june")
    for c in forbidden_train:
        assert train_cats[c] == 0, f"train has {train_cats[c]} of forbidden cat {c}"
    for c in forbidden_val:
        assert val_cats[c] == 0, f"val has {val_cats[c]} of forbidden cat {c}"

    train_final = sorted(train_set)
    val_final = sorted(val_set)
    rng.shuffle(train_final)
    rng.shuffle(val_final)

    print(f"[6/6] Writing manifests to {args.output_dir}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.output_dir / "train.txt"
    val_out = args.output_dir / "val.txt"
    train_out.write_text("\n".join(train_final) + "\n")
    val_out.write_text("\n".join(val_final) + "\n")

    def hist(name: str, items) -> None:
        src_ctr = Counter(source_prefix(x) for x in items)
        print(f"   {name} ({len(items)}) by source: {dict(src_ctr.most_common())}")

    hist("train.txt", train_final)
    hist("val.txt  ", val_final)
    print(f"   train per-category: {dict(train_cats)}")
    print(f"   val   per-category: {dict(val_cats)}")
    print(f"   wrote {train_out}")
    print(f"   wrote {val_out}")
    print(f"   dropped fisheye/depth: {len(dropped_fisheye_depth)}")


if __name__ == "__main__":
    main()
