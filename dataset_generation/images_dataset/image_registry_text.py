#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# ----------------------------
# Hashing
# ----------------------------

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ----------------------------
# Config parsing
# ----------------------------

def read_relpaths_config(config_path: Path) -> List[str]:
    relpaths: List[str] = []
    for raw in config_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        relpaths.append(line.replace("\\", "/"))
    return relpaths


# ----------------------------
# Registry (text-based)
# ----------------------------

@dataclass(frozen=True)
class TextRegistryPaths:
    registry_root: Path

    @property
    def meta(self) -> Path:
        return self.registry_root / "meta.json"

    @property
    def ledger(self) -> Path:
        return self.registry_root / "ledger.jsonl"

    @property
    def path_to_id(self) -> Path:
        return self.registry_root / "path_to_id.csv"

    @property
    def id_to_paths(self) -> Path:
        return self.registry_root / "id_to_paths.jsonl"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_relpath(relpath: str) -> str:
    # Normalize and ensure it's relative-ish (no leading /)
    rel = relpath.replace("\\", "/").lstrip("/")
    # Disallow parent traversal
    if rel.startswith("../") or "/../" in rel or rel == "..":
        raise ValueError(f"Refusing relpath with parent traversal: {relpath}")
    return rel


def write_meta(meta_path: Path, root: Path, hash_alg: str) -> None:
    meta = {
        "root": str(root.resolve()),
        "hash_alg": hash_alg,
        "schema_version": 1,
        "created_at_ns": time.time_ns(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def append_ledger(ledger_path: Path, record: dict) -> None:
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def rebuild_indexes(paths: TextRegistryPaths) -> Tuple[int, int]:
    """
    Rebuild:
      - path_to_id.csv   (unique latest mapping per path)
      - id_to_paths.jsonl (image_id -> list of paths)
    from ledger.jsonl.
    Returns (num_paths, num_ids)
    """
    if not paths.ledger.exists():
        raise FileNotFoundError(f"Missing ledger: {paths.ledger}")

    # Latest record per (root, relpath)
    latest_by_path: Dict[Tuple[str, str], dict] = {}
    # image_id -> set of (root, relpath)
    id_to_paths: Dict[str, List[Tuple[str, str]]] = {}

    with paths.ledger.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            root = rec["root"]
            rel = rec["relpath"]
            key = (root, rel)

            prev = latest_by_path.get(key)
            if prev is None or rec["indexed_at_ns"] >= prev["indexed_at_ns"]:
                latest_by_path[key] = rec

    # Build reverse mapping from latest_by_path only (keeps it sane)
    for rec in latest_by_path.values():
        image_id = rec["image_id"]
        id_to_paths.setdefault(image_id, []).append((rec["root"], rec["relpath"]))

    # Write path_to_id.csv
    with paths.path_to_id.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["root", "relpath", "image_id", "file_size", "mtime_ns", "hash_alg", "indexed_at_ns"])
        for rec in sorted(latest_by_path.values(), key=lambda r: (r["root"], r["relpath"])):
            w.writerow([
                rec["root"], rec["relpath"], rec["image_id"],
                rec["file_size"], rec["mtime_ns"], rec["hash_alg"], rec["indexed_at_ns"]
            ])

    # Write id_to_paths.jsonl
    with paths.id_to_paths.open("w", encoding="utf-8") as f:
        for image_id in sorted(id_to_paths.keys()):
            entries = [{"root": root, "relpath": rel} for (root, rel) in sorted(id_to_paths[image_id])]
            f.write(json.dumps({"image_id": image_id, "paths": entries}, ensure_ascii=False) + "\n")

    return len(latest_by_path), len(id_to_paths)


def lookup_path_to_id(path_to_id_csv: Path, root: Path, relpath: str) -> str | None:
    root_s = str(root.resolve())
    rel = safe_relpath(relpath)

    if not path_to_id_csv.exists():
        return None

    with path_to_id_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["root"] == root_s and row["relpath"] == rel:
                return row["image_id"]
    return None


def lookup_id_to_paths(id_to_paths_jsonl: Path, image_id: str) -> List[dict]:
    if not id_to_paths_jsonl.exists():
        return []
    with id_to_paths_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("image_id") == image_id:
                return rec.get("paths", [])
    return []


def index_from_config(
    registry_root: Path,
    image_root: Path,
    config_file: Path,
    hash_alg: str = "sha256",
    strict: bool = True,
    rebuild: bool = True,
) -> Tuple[int, int]:
    if hash_alg != "sha256":
        raise ValueError("This text registry script currently supports only sha256 (full-file).")

    ensure_dir(registry_root)
    paths = TextRegistryPaths(registry_root=registry_root)

    # Write meta once (or keep existing)
    if not paths.meta.exists():
        write_meta(paths.meta, image_root, hash_alg)

    relpaths = read_relpaths_config(config_file)
    ok = 0
    failed = 0

    root = image_root.resolve()
    for rel in relpaths:
        rel_norm = safe_relpath(rel)
        abs_path = (root / rel_norm).resolve()

        if strict:
            if not abs_path.exists():
                print(f"[ERROR] Missing file: {rel_norm}", file=sys.stderr)
                failed += 1
                continue
            if not abs_path.is_file():
                print(f"[ERROR] Not a file: {rel_norm}", file=sys.stderr)
                failed += 1
                continue
        else:
            if not abs_path.exists() or not abs_path.is_file():
                failed += 1
                continue

        st = abs_path.stat()
        rec = {
            "root": str(root),
            "relpath": rel_norm,
            "abspath": str(abs_path),          # helpful for humans, okay if local-only
            "image_id": sha256_file(abs_path),
            "hash_alg": hash_alg,
            "file_size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
            "indexed_at_ns": time.time_ns(),
        }

        append_ledger(paths.ledger, rec)
        ok += 1

    if rebuild:
        rebuild_indexes(paths)

    return ok, failed


# ----------------------------
# CLI
# ----------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Text-based image registry (JSONL + CSV indexes).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Index images listed in a config file.")
    p_index.add_argument("--image-root", required=True, help="Root directory containing images.")
    p_index.add_argument("--config", required=True, help="Config file (relative paths, one per line).")
    p_index.add_argument("--registry-root", required=True, help="Directory to store registry files.")
    p_index.add_argument("--non-strict", action="store_true", help="Skip missing files instead of erroring.")
    p_index.add_argument("--no-rebuild", action="store_true", help="Do not rebuild indexes after indexing.")

    p_rebuild = sub.add_parser("rebuild", help="Rebuild CSV/JSONL indexes from ledger.jsonl.")
    p_rebuild.add_argument("--registry-root", required=True)

    p_lookup_path = sub.add_parser("lookup-path", help="Find image_id for (image_root, relpath).")
    p_lookup_path.add_argument("--image-root", required=True)
    p_lookup_path.add_argument("--relpath", required=True)
    p_lookup_path.add_argument("--registry-root", required=True)

    p_lookup_id = sub.add_parser("lookup-id", help="Find file(s) for an image_id.")
    p_lookup_id.add_argument("--image-id", required=True)
    p_lookup_id.add_argument("--registry-root", required=True)

    args = p.parse_args()

    if args.cmd == "index":
        ok, failed = index_from_config(
            registry_root=Path(args.registry_root),
            image_root=Path(args.image_root),
            config_file=Path(args.config),
            strict=not args.non_strict,
            rebuild=not args.no_rebuild,
        )
        print(f"Indexed OK: {ok}, Failed: {failed}")
        if args.no_rebuild:
            print("Note: indexes not rebuilt (--no-rebuild). Run `rebuild` later.")
        return 0

    if args.cmd == "rebuild":
        paths = TextRegistryPaths(registry_root=Path(args.registry_root))
        num_paths, num_ids = rebuild_indexes(paths)
        print(f"Rebuilt indexes: {num_paths} paths, {num_ids} unique image_ids")
        return 0

    if args.cmd == "lookup-path":
        rr = TextRegistryPaths(registry_root=Path(args.registry_root))
        image_id = lookup_path_to_id(rr.path_to_id, Path(args.image_root), args.relpath)
        if image_id is None:
            print("NOT FOUND")
            return 2
        print(image_id)
        return 0

    if args.cmd == "lookup-id":
        rr = TextRegistryPaths(registry_root=Path(args.registry_root))
        paths = lookup_id_to_paths(rr.id_to_paths, args.image_id)
        if not paths:
            print("NOT FOUND")
            return 2
        for pth in paths:
            print(f"{pth['root']}/{pth['relpath']}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())


# python image_registry_text.py index \
#   --image-root /work/datasets \
#   --config images_to_process.txt \
#   --registry-root /work/registry_text

# python image_registry_text.py lookup-path \
#   --image-root /work/datasets \
#   --relpath dataset1/subset2/image3.png \
#   --registry-root /work/registry_text

# python image_registry_text.py lookup-id \
#   --image-id <sha256hex> \
#   --registry-root /work/registry_text
