"""
Merge a secondary image registry into a primary one.

Both registries are SQLite DBs produced by precache_registry.py, living at
    <registry_root>/registry.sqlite3
with an `images` table keyed by (root, relpath).

Example (copy/paste, swap the paths):

    python merge_registry.py \
        --primary   /scratch/aaadkins/data/semantic-persistence-filter/registries/train_mix_main \
        --secondary /scratch/aaadkins/data/semantic-persistence-filter/registries/train_mix_extra \
        --on-conflict prefer-newer \
        --export-human-readable

Arguments:
    --primary <dir>
        Path to the primary registry_root. This is the directory that contains
        registry.sqlite3 (NOT the .sqlite3 file itself). It will be modified
        in place — new rows from --secondary are merged into it.

    --secondary <dir>
        Path to the secondary registry_root (also a directory containing
        registry.sqlite3). Opened read-only; never modified.

    --on-conflict {keep-primary,prefer-secondary,prefer-newer}
        What to do when the same (root, relpath) exists in both registries:
            keep-primary      (default) leave the primary row untouched
            prefer-secondary  overwrite primary with secondary's row
            prefer-newer      pick whichever row has the larger indexed_at_ns
        Rows whose payload (image_id, abspath, size, mtime) is identical in
        both registries are always counted as "skipped_same" — no write.
        image_id mismatches at the same (root, relpath) are logged to stderr.

    --export-human-readable
        After merging, re-run RegistryExporter on the primary so that
        <primary>/exports/path_to_id.csv and id_to_paths.jsonl reflect the
        merged state. Omit if you only need the SQLite DB updated.

    --dry-run
        Open the primary read-only and report what would change (insert /
        overwrite / skip counts) without writing anything.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from dataset_generation.images_dataset.exporter import RegistryExporter
from dataset_generation.images_dataset.registry import ImageRegistry

ConflictPolicy = str  # "keep-primary" | "prefer-secondary" | "prefer-newer"


def _registry_for(root: Path, *, read_only: bool) -> ImageRegistry:
    db_path = root / ImageRegistry.DEFAULT_DB_NAME
    if not db_path.exists():
        raise FileNotFoundError(f"No registry DB at {db_path}")
    return ImageRegistry.from_db_path(db_path, read_only=read_only)


def _fmt_diff(existing: dict, incoming: dict) -> str:
    fields = ("image_id", "abspath", "file_size", "mtime_ns", "indexed_at_ns")
    diffs = [f"{f}: {existing[f]!r} -> {incoming[f]!r}"
             for f in fields if existing[f] != incoming[f]]
    return "; ".join(diffs) if diffs else "(no field differences)"


def merge_registries(
    primary_root: Path,
    secondary_root: Path,
    *,
    on_conflict: ConflictPolicy = "keep-primary",
    export_human_readable: bool = False,
    dry_run: bool = False,
    sample: int = 0,
    sample_random: bool = False,
    sample_seed: int = 0,
    require_safe: bool = False,
    safe_fields: Tuple[str, ...] = ("mtime_ns", "indexed_at_ns"),
) -> dict:
    if on_conflict not in {"keep-primary", "prefer-secondary", "prefer-newer"}:
        raise ValueError(f"Unknown --on-conflict policy: {on_conflict}")
    safe_field_set = set(safe_fields)

    primary = _registry_for(primary_root, read_only=dry_run)
    secondary = _registry_for(secondary_root, read_only=True)

    try:
        secondary_rows = secondary.all_rows()

        to_write: List[Tuple[str, str, str, str, int, int, int]] = []
        stats = {
            "secondary_total": len(secondary_rows),
            "inserted": 0,
            "overwritten": 0,
            "skipped_same": 0,
            "skipped_conflict": 0,
            "id_mismatches": 0,
        }
        # For deterministic ordered sampling we keep at most N per bucket.
        # For random sampling we use reservoir sampling so we don't have to hold
        # everything in memory.
        samples = {"inserted": [], "overwritten": [], "skipped_same": [], "skipped_conflict": []}
        seen_per_bucket: Counter = Counter()
        rng = random.Random(sample_seed) if sample_random else None

        # Track field-level diffs among conflicts (overwritten + skipped_conflict)
        # so we can tell metadata drift from real content/id problems.
        conflict_field_diffs: Counter = Counter()
        conflict_image_id_matches = 0
        conflict_image_id_mismatches = 0
        unsafe_rows: List[dict] = []  # rows whose diffs exceed safe_field_set

        def _maybe_sample(bucket: str, payload):
            if sample <= 0:
                return
            seen_per_bucket[bucket] += 1
            if rng is None:
                if len(samples[bucket]) < sample:
                    samples[bucket].append(payload)
            else:
                # Reservoir sampling: keeps a uniform random sample of size `sample`.
                n = seen_per_bucket[bucket]
                if n <= sample:
                    samples[bucket].append(payload)
                else:
                    j = rng.randrange(n)
                    if j < sample:
                        samples[bucket][j] = payload

        for r in secondary_rows:
            existing = primary.get_record(r["root"], r["relpath"])

            if existing is None:
                stats["inserted"] += 1
                _maybe_sample("inserted", {"row": dict(r)})
                to_write.append((
                    r["root"], r["relpath"], r["abspath"], r["image_id"],
                    r["file_size"], r["mtime_ns"], r["indexed_at_ns"],
                ))
                continue

            # Conflict on (root, relpath).
            same_payload = (
                existing["image_id"] == r["image_id"]
                and existing["abspath"] == r["abspath"]
                and existing["file_size"] == r["file_size"]
                and existing["mtime_ns"] == r["mtime_ns"]
            )
            if same_payload:
                stats["skipped_same"] += 1
                _maybe_sample("skipped_same", {"row": dict(r)})
                continue

            # Per-field diff tally — useful to distinguish metadata drift
            # (abspath/mtime only) from real problems (image_id/file_size).
            differing_fields = [f for f in ("image_id", "abspath", "file_size", "mtime_ns",
                                            "indexed_at_ns")
                                if existing[f] != r[f]]
            for f in differing_fields:
                conflict_field_diffs[f] += 1

            unsafe_diffs = [f for f in differing_fields if f not in safe_field_set]
            if unsafe_diffs:
                unsafe_rows.append({
                    "root": r["root"],
                    "relpath": r["relpath"],
                    "unsafe_fields": unsafe_diffs,
                    "diff": _fmt_diff(dict(existing), dict(r)),
                })

            if existing["image_id"] != r["image_id"]:
                conflict_image_id_mismatches += 1
                stats["id_mismatches"] += 1
                print(
                    f"[warn] image_id mismatch for {r['root']} :: {r['relpath']}: "
                    f"primary={existing['image_id']} secondary={r['image_id']}",
                    file=sys.stderr,
                )
            else:
                conflict_image_id_matches += 1

            take_secondary = False
            if on_conflict == "prefer-secondary":
                take_secondary = True
            elif on_conflict == "prefer-newer":
                take_secondary = r["indexed_at_ns"] > existing["indexed_at_ns"]

            bucket = "overwritten" if take_secondary else "skipped_conflict"
            _maybe_sample(bucket, {
                "key": {"root": r["root"], "relpath": r["relpath"]},
                "diff": _fmt_diff(dict(existing), dict(r)),
                "primary": dict(existing),
                "secondary": dict(r),
            })

            if take_secondary:
                stats["overwritten"] += 1
                to_write.append((
                    r["root"], r["relpath"], r["abspath"], r["image_id"],
                    r["file_size"], r["mtime_ns"], r["indexed_at_ns"],
                ))
            else:
                stats["skipped_conflict"] += 1

        if require_safe and unsafe_rows:
            print(
                f"\n[require-safe] REFUSING to merge: {len(unsafe_rows)} conflicts have "
                f"differences outside the safe field set {sorted(safe_field_set)}.",
                file=sys.stderr,
            )
            print("First few unsafe rows:", file=sys.stderr)
            for u in unsafe_rows[:10]:
                print(f"  {u['root']} :: {u['relpath']}", file=sys.stderr)
                print(f"    unsafe fields: {u['unsafe_fields']}", file=sys.stderr)
                print(f"    diff: {u['diff']}", file=sys.stderr)
            raise SystemExit(2)

        if dry_run:
            print("[dry-run] no changes written")
            if require_safe:
                print(f"[require-safe] OK: 0 unsafe rows "
                      f"(safe fields = {sorted(safe_field_set)})")
        else:
            primary.bulk_upsert(to_write)
            if export_human_readable:
                RegistryExporter(primary).export(primary_root / "exports")
            if require_safe:
                print(f"[require-safe] OK: 0 unsafe rows "
                      f"(safe fields = {sorted(safe_field_set)})")

        total_conflicts = conflict_image_id_matches + conflict_image_id_mismatches
        if total_conflicts > 0:
            print("\nConflict breakdown (rows where (root, relpath) existed in both):")
            print(f"  total conflicts:        {total_conflicts}")
            print(f"  image_id matched:       {conflict_image_id_matches}  "
                  "(metadata drift only — expected for cross-system merges)")
            print(f"  image_id MISMATCHED:    {conflict_image_id_mismatches}  "
                  "(suspicious — different content or different id-computer)")
            print("  fields that differed (per-row tally):")
            for f, n in sorted(conflict_field_diffs.items(), key=lambda kv: -kv[1]):
                print(f"    {f:12s} {n}")

        if sample > 0:
            for bucket, items in samples.items():
                if not items:
                    continue
                print(f"\n--- sample of {bucket} (up to {sample}) ---")
                for i, item in enumerate(items, 1):
                    if bucket in ("inserted", "skipped_same"):
                        row = item["row"]
                        print(f"  [{i}] {row['root']} :: {row['relpath']}  "
                              f"image_id={row['image_id']}")
                    else:
                        print(f"  [{i}] {item['key']['root']} :: {item['key']['relpath']}")
                        print(f"       {item['diff']}")

        return stats
    finally:
        secondary.close()
        primary.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--primary", required=True, type=Path, help="Primary registry_root (will be modified)")
    p.add_argument("--secondary", required=True, type=Path, help="Secondary registry_root (read-only source)")
    p.add_argument(
        "--on-conflict",
        default="keep-primary",
        choices=["keep-primary", "prefer-secondary", "prefer-newer"],
    )
    p.add_argument("--export-human-readable", action="store_true",
                   help="Re-export <primary>/exports/ after merge")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change without writing")
    p.add_argument("--sample", type=int, default=0,
                   help="Print up to N sample rows per bucket "
                        "(inserted/overwritten/skipped_same/skipped_conflict). "
                        "For overwritten/skipped_conflict, prints the field diff.")
    p.add_argument("--sample-random", action="store_true",
                   help="Pick samples uniformly at random (reservoir sampling) "
                        "instead of the first N by sort order.")
    p.add_argument("--sample-seed", type=int, default=0,
                   help="Seed for --sample-random (default 0 for reproducibility).")
    p.add_argument("--require-safe", action="store_true",
                   help="Refuse to merge (exit 2) if any conflict has differences "
                        "outside --safe-fields. Use this to certify a cross-system "
                        "merge as 'metadata drift only'.")
    p.add_argument("--safe-fields", default="mtime_ns,indexed_at_ns",
                   help="Comma-separated list of fields whose differences are "
                        "considered safe. Default: mtime_ns,indexed_at_ns. Other "
                        "fields are image_id, abspath, file_size.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    stats = merge_registries(
        primary_root=args.primary,
        secondary_root=args.secondary,
        on_conflict=args.on_conflict,
        export_human_readable=args.export_human_readable,
        dry_run=args.dry_run,
        sample=args.sample,
        sample_random=args.sample_random,
        sample_seed=args.sample_seed,
        require_safe=args.require_safe,
        safe_fields=tuple(f.strip() for f in args.safe_fields.split(",") if f.strip()),
    )
    print("Merge summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
