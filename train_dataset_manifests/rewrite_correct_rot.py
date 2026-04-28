#!/usr/bin/env python3
"""
Generic version of rewrite_exmaps_correct_rot.py.

Rewrites paths in train_mix_*.txt manifests from <prefix><old_subdir>/...
to <prefix><new_subdir>/... using a rewrites module (e.g. exmaps_rewrites,
openloris_rewrites). Backs up each modified manifest as backup_<name> and
emits a manifest of the new paths.

Run from inside the dataset container.

Example:
  python train_dataset_manifests/rewrite_correct_rot.py \\
    --rewrites-module openloris_rewrites
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_rewrites(module_name: str) -> Tuple[str, Dict[str, str]]:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "PREFIX") or not hasattr(mod, "SUBDIR_REWRITES"):
        raise SystemExit(
            f"ERROR: {module_name} must expose both PREFIX and SUBDIR_REWRITES"
        )
    return mod.PREFIX, mod.SUBDIR_REWRITES


def rewrite_line(line: str, prefix: str, rewrites: Dict[str, str]) -> Tuple[str, bool, bool]:
    """
    Returns (new_line, was_rewritten, is_partial_match).

    - was_rewritten: strict (<prefix><dir>/<basename> with <dir> exactly a key).
    - is_partial_match: line contains a key as a /-bounded substring but did
      NOT satisfy the strict rule (e.g. nested subdir under the key).
    """
    if not line.startswith(prefix):
        return line, False, False

    rel = line[len(prefix):]
    last_slash = rel.rfind("/")
    if last_slash == -1:
        return line, False, False

    dir_part = rel[:last_slash]
    basename = rel[last_slash + 1:]

    if dir_part in rewrites:
        return f"{prefix}{rewrites[dir_part]}/{basename}", True, False

    padded = "/" + rel + "/"
    for key in rewrites:
        if f"/{key}/" in padded:
            return line, False, True

    return line, False, False


def process_manifest(
    manifest_path: Path,
    prefix: str,
    rewrites: Dict[str, str],
    dry_run: bool,
) -> Tuple[int, List[str], List[str]]:
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

        new_line, was_rewritten, is_partial = rewrite_line(stripped, prefix, rewrites)
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
    if not backup_path.exists():
        shutil.copy2(manifest_path, backup_path)
    # If backup already exists (e.g. from a prior rewrite pass), leave it
    # alone — it represents the true original, and overwriting it with the
    # current already-rewritten state would lose that history.

    trailing_nl = (
        "\n" if original_lines and manifest_path.read_text(encoding="utf-8").endswith("\n") else ""
    )
    manifest_path.write_text("\n".join(new_lines) + trailing_nl, encoding="utf-8")

    return changed, new_relpaths, partial_matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--rewrites-module", required=True)
    parser.add_argument("--manifest-dir", type=Path, default=script_dir)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/data/input"),
    )
    parser.add_argument(
        "--new-files-out",
        type=Path,
        default=None,
        help="Where to write the list of new paths. "
        "Default: <manifest-dir>/<rewrites-module>_new_files.txt",
    )
    parser.add_argument("--partial-log", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prefix, rewrites = load_rewrites(args.rewrites_module)

    manifest_dir = args.manifest_dir.resolve()
    if not manifest_dir.is_dir():
        print(f"ERROR: manifest dir not found: {manifest_dir}", file=sys.stderr)
        return 1

    manifests = sorted(
        p for p in manifest_dir.glob("train_mix_*.txt") if not p.name.startswith("backup_")
    )
    if not manifests:
        print(f"ERROR: no train_mix_*.txt manifests found in {manifest_dir}", file=sys.stderr)
        return 1

    print(f"Module: {args.rewrites_module} (prefix={prefix!r}, {len(rewrites)} rewrite(s))")
    print(f"Found {len(manifests)} manifest(s) in {manifest_dir}")
    if args.dry_run:
        print("(dry run — no files will be written)")

    new_files_out = args.new_files_out or (
        manifest_dir / f"{args.rewrites_module}_new_files.txt"
    )
    partial_log = args.partial_log or (
        manifest_dir / f"{args.rewrites_module}_partial_matches.log"
    )

    all_new: Set[str] = set()
    all_partials: List[str] = []
    changed_manifests: List[str] = []
    skipped_manifests: List[str] = []

    for m in manifests:
        changed, new_relpaths, partials = process_manifest(
            m, prefix, rewrites, args.dry_run
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

    print(f"\nTotal unique new paths: {len(all_new)}")

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

    if all_partials:
        print(
            f"\n=== Partial matches (key appears in path but strict rule did "
            f"not fire): {len(all_partials)} line(s) ==="
        )
        for line in all_partials[:20]:
            print(f"  {line}")
        if len(all_partials) > 20:
            print(f"  ... ({len(all_partials) - 20} more, see {partial_log})")
        if not args.dry_run:
            partial_log.write_text("\n".join(all_partials) + "\n", encoding="utf-8")
            print(f"Wrote partial-match log: {partial_log}")
    else:
        print("\nNo partial matches detected.")

    if not args.dry_run and all_new:
        new_files_out.parent.mkdir(parents=True, exist_ok=True)
        new_files_out.write_text("\n".join(sorted(all_new)) + "\n", encoding="utf-8")
        print(f"\nWrote new-files manifest: {new_files_out} ({len(all_new)} path(s))")
    elif args.dry_run and all_new:
        print(f"\n(dry run) would write {len(all_new)} path(s) to {new_files_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
