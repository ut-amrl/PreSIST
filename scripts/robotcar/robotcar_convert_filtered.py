#!/usr/bin/env python3
"""
robotcar_convert_filtered.py

Drives RobotCar frame conversion from query_results.csv files (location-query
output from build_gps_index.py). For each row in the CSV, places the target
frame at <output_root>/<traversal>/<ms>.png using a 3-priority source-resolution
strategy:

  1. Existing canonical PNG at <existing_location_root>/<traversal>/<ms>.png
  2. Full-conversion tree PNG at <full_conversion_root>/<traversal>/rgb/<ms>.png
  3. Fresh conversion via process_image() from robotcar_convert.py

Reuses these helpers from robotcar_convert.py (no edits to that file):
  process_image, load_timestamps, us_to_ms

Writes an updated copy of query_results.csv to <output_root>/query_results.csv
with `converted` and `image_path` columns reflecting frames that landed.

Layout produced (flat — matches existing robotcar_queries/<location>/):
  <output_root>/<traversal>/<ms>.png
  <output_root>/query_results.csv
"""

import argparse
import csv
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Reuse helpers from robotcar_convert.py without modifying it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from robotcar_convert import process_image, load_timestamps, us_to_ms  # noqa: E402


CSV_COLUMNS = [
    "dist_m", "traversal", "image_timestamp_ms",
    "latitude", "longitude", "northing", "easting",
    "image_path", "converted",
]

OUTCOMES = ("copied_canonical", "copied_full", "converted", "already_present", "missing")


def read_query_csv(csv_path: Path) -> list[dict]:
    """Parse query_results.csv (CRLF-tolerant). Returns list of row dicts."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize whitespace/CR on every value
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def write_query_csv(csv_path: Path, rows: list[dict]) -> None:
    """Write rows back as LF-terminated CSV with the canonical column order."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in CSV_COLUMNS})


def resolve_and_place(
    traversal: str,
    ms: int,
    out_dir: Path,
    existing_location_root: Path | None,
    full_conversion_root: Path | None,
    downloads_root: Path,
    ms_to_us_path: dict[int, Path] | None,
    regenerate: bool,
) -> str:
    """
    Resolve the source PNG for one (traversal, ms) and place it at out_dir/<ms>.png.
    Returns one of OUTCOMES.

    `ms_to_us_path` is a precomputed lookup for the traversal: ms -> source raw image
    path under stereo/centre/. It is None for traversals where no fresh conversion
    will ever be needed (callers can pass {} or None).
    """
    dst = out_dir / f"{ms}.png"
    if dst.exists():
        return "already_present"

    # Priority 1: existing canonical
    if not regenerate and existing_location_root is not None:
        candidate = existing_location_root / traversal / f"{ms}.png"
        if candidate.exists():
            shutil.copy2(candidate, dst)
            return "copied_canonical"

    # Priority 2: full-conversion tree
    if not regenerate and full_conversion_root is not None:
        candidate = full_conversion_root / traversal / "rgb" / f"{ms}.png"
        if candidate.exists():
            shutil.copy2(candidate, dst)
            return "copied_full"

    # Priority 3: fresh conversion from raw
    if ms_to_us_path is None:
        return "missing"
    src = ms_to_us_path.get(ms)
    if src is None or not src.exists():
        return "missing"
    process_image(src, dst)
    return "converted"


def build_ms_to_us_lookup(traversal_dir: Path) -> dict[int, Path]:
    """Map us_to_ms(ts_us) -> raw image Path, only for entries whose PNG exists."""
    timestamps_file = traversal_dir / "stereo.timestamps"
    images_dir = traversal_dir / "stereo" / "centre"
    if not timestamps_file.exists() or not images_dir.exists():
        return {}
    available = {p.stem: p for p in images_dir.glob("*.png")}
    out: dict[int, Path] = {}
    for ts_us, _chunk in load_timestamps(timestamps_file):
        if str(ts_us) in available:
            ms = us_to_ms(ts_us)
            # Multiple us could collapse to the same ms; first one wins (matches
            # robotcar_convert.py's behavior — it processes in timestamp order).
            out.setdefault(ms, available[str(ts_us)])
    return out


def process_traversal(
    traversal: str,
    targets_ms: list[int],
    output_root: Path,
    existing_location_root: Path | None,
    full_conversion_root: Path | None,
    downloads_root: Path,
    regenerate: bool,
    dry_run: bool,
) -> tuple[str, Counter, dict[int, str], float]:
    """Process all target frames for one traversal. Returns (traversal, counts, per-ms outcomes, elapsed_s)."""
    start = time.time()
    out_dir = output_root / traversal
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Decide whether we need the raw-image lookup (only if any target might fall to priority 3)
    need_raw_lookup = regenerate or full_conversion_root is None or existing_location_root is None
    if not need_raw_lookup:
        # Even if both higher priorities exist, we still need raw lookup for any ms
        # that's missing from both. Just always build it; it's cheap.
        pass

    traversal_dir = downloads_root / traversal
    ms_to_us_path = build_ms_to_us_lookup(traversal_dir) if traversal_dir.exists() else {}

    counts: Counter = Counter()
    per_ms: dict[int, str] = {}
    for ms in targets_ms:
        if dry_run:
            # Predict outcome without writing
            outcome = predict_outcome(
                traversal=traversal,
                ms=ms,
                out_dir=out_dir,
                existing_location_root=existing_location_root,
                full_conversion_root=full_conversion_root,
                ms_to_us_path=ms_to_us_path,
                regenerate=regenerate,
            )
        else:
            outcome = resolve_and_place(
                traversal=traversal,
                ms=ms,
                out_dir=out_dir,
                existing_location_root=existing_location_root,
                full_conversion_root=full_conversion_root,
                downloads_root=downloads_root,
                ms_to_us_path=ms_to_us_path,
                regenerate=regenerate,
            )
        counts[outcome] += 1
        per_ms[ms] = outcome

    elapsed = time.time() - start
    return traversal, counts, per_ms, elapsed


def predict_outcome(
    traversal: str,
    ms: int,
    out_dir: Path,
    existing_location_root: Path | None,
    full_conversion_root: Path | None,
    ms_to_us_path: dict[int, Path],
    regenerate: bool,
) -> str:
    """Same priority logic as resolve_and_place but read-only."""
    if (out_dir / f"{ms}.png").exists():
        return "already_present"
    if not regenerate and existing_location_root is not None:
        if (existing_location_root / traversal / f"{ms}.png").exists():
            return "copied_canonical"
    if not regenerate and full_conversion_root is not None:
        if (full_conversion_root / traversal / "rgb" / f"{ms}.png").exists():
            return "copied_full"
    if ms in ms_to_us_path:
        return "converted"
    return "missing"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", nargs="+", required=True, help="One or more query_results.csv paths")
    ap.add_argument("--downloads_root", required=True, type=Path)
    ap.add_argument("--existing_location_root", type=Path, default=None,
                    help="Priority-1 source: existing flat <traversal>/<ms>.png tree")
    ap.add_argument("--full_conversion_root", type=Path, default=None,
                    help="Priority-2 source: full <traversal>/rgb/<ms>.png tree")
    ap.add_argument("--output_root", required=True, type=Path)
    ap.add_argument("--regenerate", action="store_true",
                    help="Force priority-3 (fresh conversion); skip copy from priorities 1 and 2")
    ap.add_argument("--only_no", action="store_true",
                    help="Only process rows where converted==no in the CSV")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    # Aggregate rows from all input CSVs
    all_rows: list[dict] = []
    for q in args.queries:
        path = Path(q)
        rows = read_query_csv(path)
        for r in rows:
            r["_source_csv"] = str(path)
        all_rows.extend(rows)

    if args.only_no:
        all_rows = [r for r in all_rows if r.get("converted", "").lower() == "no"]

    # Group by traversal
    by_trav: dict[str, list[int]] = defaultdict(list)
    for r in all_rows:
        try:
            ms = int(r["image_timestamp_ms"])
        except (KeyError, ValueError):
            print(f"[WARN] skipping row with bad image_timestamp_ms: {r}")
            continue
        by_trav[r["traversal"]].append(ms)

    # Deduplicate ms within each traversal but preserve order
    for trav in by_trav:
        seen = set()
        deduped = []
        for ms in by_trav[trav]:
            if ms not in seen:
                seen.add(ms)
                deduped.append(ms)
        by_trav[trav] = deduped

    print(f"Loaded {len(all_rows)} rows across {len(by_trav)} traversals from {len(args.queries)} CSV(s).")
    if args.dry_run:
        print("[dry-run] No files will be written.")

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)

    # Process traversals
    total_counts: Counter = Counter()
    per_ms_outcomes: dict[tuple[str, int], str] = {}

    if args.workers > 1 and not args.dry_run:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    process_traversal,
                    trav, msl, args.output_root,
                    args.existing_location_root, args.full_conversion_root,
                    args.downloads_root, args.regenerate, False,
                ): trav
                for trav, msl in by_trav.items()
            }
            for fut in as_completed(futures):
                trav, counts, per_ms, elapsed = fut.result()
                total_counts.update(counts)
                for ms, outcome in per_ms.items():
                    per_ms_outcomes[(trav, ms)] = outcome
                _print_traversal_summary(trav, len(by_trav[trav]), counts, elapsed)
    else:
        for trav, msl in by_trav.items():
            trav_, counts, per_ms, elapsed = process_traversal(
                trav, msl, args.output_root,
                args.existing_location_root, args.full_conversion_root,
                args.downloads_root, args.regenerate, args.dry_run,
            )
            total_counts.update(counts)
            for ms, outcome in per_ms.items():
                per_ms_outcomes[(trav, ms)] = outcome
            _print_traversal_summary(trav, len(msl), counts, elapsed)

    # Final summary
    print("\n=== Final summary ===")
    grand = sum(total_counts.values())
    for k in OUTCOMES:
        print(f"  {k:20s}: {total_counts.get(k, 0)}")
    print(f"  {'total':20s}: {grand}")

    # Update and write CSV copy (only if not dry-run)
    if not args.dry_run:
        updated_rows = []
        for r in all_rows:
            r = {k: v for k, v in r.items() if k != "_source_csv"}
            try:
                ms = int(r["image_timestamp_ms"])
            except (KeyError, ValueError):
                updated_rows.append(r)
                continue
            outcome = per_ms_outcomes.get((r["traversal"], ms))
            placed = outcome in ("copied_canonical", "copied_full", "converted", "already_present")
            if placed:
                r["converted"] = "yes"
                r["image_path"] = str(args.output_root / r["traversal"] / f"{ms}.png")
            updated_rows.append(r)
        out_csv = args.output_root / "query_results.csv"
        write_query_csv(out_csv, updated_rows)
        print(f"\nWrote updated CSV: {out_csv}")


def _print_traversal_summary(trav: str, n_targets: int, counts: Counter, elapsed: float) -> None:
    parts = " ".join(f"{k}={counts.get(k,0)}" for k in OUTCOMES if counts.get(k, 0))
    print(f"  [{trav}] targets={n_targets} {parts}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
