#!/usr/bin/env python3
"""
Rewrite exmaps paths in train_mix_*.txt manifests to point at the
*_correct_rot variants for a fixed set of (date, run) subdirectories,
back up the originals as backup_<name>, and emit a manifest of the new
files for downstream zipping with zip_manifest.py.

Run from inside the container started by docker_start_datasets_plus_spf.sh.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path
from typing import List, Set, Tuple

from exmaps_rewrites import EXMAPS_PREFIX, SUBDIR_REWRITES


def rewrite_line(line: str) -> Tuple[str, bool, bool]:
    """
    Returns (new_line, was_rewritten, is_partial_match).

    - was_rewritten: strict (exmaps/<dir>/<basename> with <dir> exactly a key).
    - is_partial_match: line contains a key as a /-bounded substring but did
      NOT satisfy the strict rule (e.g. nested subdir under the key).
    """
    if not line.startswith(EXMAPS_PREFIX):
        return line, False, False

    rel = line[len(EXMAPS_PREFIX):]  # e.g. "2020-03-22/run_2/ar_core_data/frame_x.jpg"
    last_slash = rel.rfind("/")
    if last_slash == -1:
        return line, False, False

    dir_part = rel[:last_slash]
    basename = rel[last_slash + 1:]

    if dir_part in SUBDIR_REWRITES:
        new_line = f"{EXMAPS_PREFIX}{SUBDIR_REWRITES[dir_part]}/{basename}"
        return new_line, True, False

    # Partial-match probe: does any key appear as a /-bounded substring?
    padded = "/" + rel + "/"
    for key in SUBDIR_REWRITES:
        if f"/{key}/" in padded:
            return line, False, True

    return line, False, False


def process_manifest(
    manifest_path: Path,
    dataset_root: Path,
    dry_run: bool,
) -> Tuple[int, List[str], List[str]]:
    """
    Returns (num_changed_lines, new_relpaths, partial_match_lines).
    Writes backup + rewritten manifest if not dry_run and any change occurred.
    """
    original_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    new_lines: List[str] = []
    new_relpaths: List[str] = []
    partial_matches: List[str] = []
    changed = 0

    for raw in original_lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(raw)
            continue

        new_line, was_rewritten, is_partial = rewrite_line(stripped)
        if was_rewritten:
            changed += 1
            new_lines.append(new_line)
            new_relpaths.append(new_line)
        else:
            new_lines.append(raw)
            if is_partial:
                partial_matches.append(f"{manifest_path.name}: {stripped}")

    if changed == 0 or dry_run:
        return changed, new_relpaths, partial_matches

    backup_path = manifest_path.with_name(f"backup_{manifest_path.name}")
    if backup_path.exists():
        if not filecmp.cmp(backup_path, manifest_path, shallow=False):
            raise SystemExit(
                f"ERROR: backup {backup_path} already exists and differs from "
                f"current {manifest_path}; refusing to overwrite. Inspect "
                f"manually before re-running."
            )
        # backup matches current — safe to leave it as-is.
    else:
        shutil.copy2(manifest_path, backup_path)

    # Preserve trailing newline if original had one.
    trailing_nl = "\n" if original_lines and manifest_path.read_text(encoding="utf-8").endswith("\n") else ""
    manifest_path.write_text("\n".join(new_lines) + trailing_nl, encoding="utf-8")

    return changed, new_relpaths, partial_matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=script_dir,
        help="Directory containing train_mix_*.txt manifests (default: this script's dir).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/data/input"),
        help="Dataset root the manifest paths are relative to (default: /workspace/data/input).",
    )
    parser.add_argument(
        "--new-files-out",
        type=Path,
        default=script_dir / "correct_rot_new_files.txt",
        help="Output manifest listing the new (rewritten) exmaps paths.",
    )
    parser.add_argument(
        "--partial-log",
        type=Path,
        default=script_dir / "partial_matches.log",
        help="Where to write partial-match diagnostics.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without touching any files.",
    )
    args = parser.parse_args()

    manifest_dir: Path = args.manifest_dir.resolve()
    if not manifest_dir.is_dir():
        print(f"ERROR: manifest dir not found: {manifest_dir}", file=sys.stderr)
        return 1

    manifests = sorted(
        p for p in manifest_dir.glob("train_mix_*.txt") if not p.name.startswith("backup_")
    )
    if not manifests:
        print(f"ERROR: no train_mix_*.txt manifests found in {manifest_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(manifests)} manifest(s) in {manifest_dir}")
    if args.dry_run:
        print("(dry run — no files will be written)")

    all_new: Set[str] = set()
    all_partials: List[str] = []
    changed_manifests: List[str] = []
    skipped_manifests: List[str] = []

    for m in manifests:
        changed, new_relpaths, partials = process_manifest(
            m, args.dataset_root.resolve(), args.dry_run
        )
        all_partials.extend(partials)
        if changed > 0:
            changed_manifests.append(f"{m.name}: {changed} line(s) rewritten")
            all_new.update(new_relpaths)
        else:
            skipped_manifests.append(m.name)

    print("\n=== Per-manifest changes ===")
    for line in changed_manifests:
        print(f"  {line}")
    if skipped_manifests:
        print("\n=== Manifests with no changes ===")
        for name in skipped_manifests:
            print(f"  {name}")

    print(f"\nTotal unique new exmaps paths: {len(all_new)}")

    # Existence check on new files (warn only).
    missing = [p for p in sorted(all_new) if not (args.dataset_root / p).is_file()]
    if missing:
        print(
            f"\nWARNING: {len(missing)} new path(s) do not exist on disk under "
            f"{args.dataset_root}:",
            file=sys.stderr,
        )
        for p in missing[:20]:
            print(f"  {p}", file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... ({len(missing) - 20} more)", file=sys.stderr)

    # Partial matches.
    if all_partials:
        print(
            f"\n=== Partial matches (key appears in path but strict rule did "
            f"not fire): {len(all_partials)} line(s) ==="
        )
        for line in all_partials[:20]:
            print(f"  {line}")
        if len(all_partials) > 20:
            print(f"  ... ({len(all_partials) - 20} more, see {args.partial_log})")
        if not args.dry_run:
            args.partial_log.write_text("\n".join(all_partials) + "\n", encoding="utf-8")
            print(f"Wrote partial-match log: {args.partial_log}")
    else:
        print("\nNo partial matches detected.")

    if not args.dry_run and all_new:
        out = args.new_files_out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(sorted(all_new)) + "\n", encoding="utf-8")
        print(f"\nWrote new-files manifest: {out} ({len(all_new)} path(s))")
    elif args.dry_run and all_new:
        print(f"\n(dry run) would write {len(all_new)} path(s) to {args.new_files_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
