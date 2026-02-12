# image_id_pipeline/preprocess.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Dict, Any, List, Literal

import torch
from torch.utils.data import DataLoader

from .hashing import ImageIdComputer, HashSpec
from .registry import ImageRegistry, RegistryConfig
from .exporter import RegistryExporter
from .resolver import ResolvedItem, resolve_paths
from .dataset import RegistryBackedImageDataset, make_collate_fn

from dataclasses import dataclass


def _dedupe_by_path(items: Sequence[ResolvedItem]) -> List[ResolvedItem]:
    seen = set()
    out: List[ResolvedItem] = []
    for it in items:
        if it.relpath in seen:
            continue
        seen.add(it.relpath)
        out.append(it)
    return out



def preprocess_and_build_dataloader(
    *,
    image_root: str,
    spec_path: str,                 # .txt OR .json/.yaml
    registry_root: str,
    image_loader: Callable[[str], torch.Tensor],
    lists_root: Optional[str] = None,  # required for config mode
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = True,
    strict: bool = True,
    export_human_readable: bool = True,
) -> Tuple[RegistryBackedImageDataset, DataLoader, Path, Dict[str, Any]]:
    """
    spec_path:
      - .txt: plain list of relpaths under image_root
      - .json/.yaml: dataset config referencing per-root named lists under lists_root

    Returns: (dataset, dataloader, registry_db_path, resolved_meta)
    """
    root = Path(image_root).resolve()

    # 1) Resolve paths
    items, meta = resolve_paths(image_root=image_root, spec_path=spec_path, lists_root=lists_root)

    # Dedupe policy from config (text-list defaults to by_path)
    dedupe = meta.get("dedupe", "by_path")
    if dedupe == "by_path":
        items = _dedupe_by_path(items)
    elif dedupe == "by_image_id":
        # We'll compute ids during indexing and then dedupe.
        # For simplicity, we dedupe after indexing by keeping the first encountered per image_id.
        pass
    else:
        raise ValueError(f"Unknown dedupe policy: {dedupe}")

    # 2) Build / open registry
    registry = ImageRegistry(RegistryConfig(registry_root=Path(registry_root)))

    print("registry_root arg:", registry_root)
    print("resolved registry root:", Path(registry_root).resolve())
    print("registry db_path:", registry.db_path)
    print("cwd:", Path.cwd())

    id_computer = ImageIdComputer(HashSpec(alg="sha256"))

    try:
        # 3) Bulk index pass (compute missing/stale IDs)
        now_ns = time.time_ns()
        to_upsert = []
        computed_ids: Dict[str, str] = {}  # relpath -> image_id

        for it in items:
            abspath = (root / it.relpath).resolve()
            if strict and (not abspath.exists() or not abspath.is_file()):
                raise FileNotFoundError(f"Missing file: {it.relpath}")

            st = abspath.stat()
            file_size = int(st.st_size)
            mtime_ns = int(st.st_mtime_ns)

            row = registry.get_record(root, it.relpath)
            if row and int(row["file_size"]) == file_size and int(row["mtime_ns"]) == mtime_ns:
                computed_ids[it.relpath] = row["image_id"]
                continue

            image_id = id_computer.compute(abspath)
            computed_ids[it.relpath] = image_id
            to_upsert.append((str(root), it.relpath, str(abspath), image_id, file_size, mtime_ns, now_ns))

        if to_upsert:
            registry.bulk_upsert(to_upsert)

        # 4) If dedupe by_image_id, reduce items now that ids are known
        if dedupe == "by_image_id":
            seen_ids = set()
            deduped: List[ResolvedItem] = []
            for it in items:
                iid = computed_ids.get(it.relpath) or registry.path_to_id(root, it.relpath)
                if iid is None:
                    # should not happen if strict and hashing succeeded
                    continue
                if iid in seen_ids:
                    continue
                seen_ids.add(iid)
                deduped.append(it)
            items = deduped

        # 5) Export readable formats
        if export_human_readable:
            RegistryExporter(registry).export(Path(registry_root) / "exports")

        # 6) Build dataset/dataloader (read-only registry use)
        dataset = RegistryBackedImageDataset(
            image_root=image_root,
            # registry=registry,
            registry_db_path=str(registry.db_path),
            image_loader=image_loader,
            resolved_items=items,
            id_computer=id_computer,
            strict=strict,
            compute_missing_ids=False,  # important: avoid DB writes in workers
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            collate_fn=make_collate_fn(),
        )

        return dataset, loader, registry.db_path, meta

    finally:
        # If you want the registry connection open during iteration, manage it externally.
        registry.close()




SpecKind = Literal["list", "config"]


@dataclass
class DatasetSpecConfig:
    """
    Hydra/OmegaConf-friendly dataset spec.

    - kind can be omitted and inferred from spec_path suffix.
    - lists_root is required only for config mode.
    """
    image_root: str
    spec_path: str
    lists_root: Optional[str] = None
    kind: Optional[SpecKind] = None  # "list" | "config" | None (infer)

    def inferred_kind(self) -> SpecKind:
        if self.kind is not None:
            return self.kind

        suffix = Path(self.spec_path).suffix.lower()
        if suffix == ".txt":
            return "list"
        if suffix in (".yaml", ".yml", ".json"):
            return "config"
        raise ValueError(f"Cannot infer dataset spec kind from spec_path: {self.spec_path}")

    def validate(self) -> None:
        k = self.inferred_kind()
        if k == "config" and not self.lists_root:
            raise ValueError("lists_root is required when using config mode (.yaml/.yml/.json spec_path).")

    def to_preprocess_kwargs(self) -> Dict[str, Any]:
        """
        Only the kwargs relevant to preprocess_and_build_dataloader().
        """
        self.validate()
        k = self.inferred_kind()
        out: Dict[str, Any] = {
            "image_root": self.image_root,
            "spec_path": self.spec_path,
        }
        if k == "config":
            out["lists_root"] = self.lists_root
        return out


# @dataclass
# class RegistryConfig:
#     registry_root: str
#     export_human_readable: bool = True
#     strict: bool = True



# How you call it
# Mode A: plain text list
# dataset, loader, db_path, meta = preprocess_and_build_dataloader(
#     image_root="/work/datasets",
#     spec_path="/work/configs/images_to_process.txt",
#     registry_root="/work/registry",
#     image_loader=loadImageForDatasetGeneration,
#     batch_size=8,
#     num_workers=4,
# )
# Mode B: composed dataset config (per-root named lists)
# dataset, loader, db_path, meta = preprocess_and_build_dataloader(
#     image_root="/work/datasets",                 # same style as text mode
#     spec_path="/work/dataset_configs/ds_AB.yaml",
#     lists_root="/work/lists",                    # contains A/subset1.txt etc.
#     registry_root="/work/registry",
#     image_loader=loadImageForDatasetGeneration,
#     batch_size=8,
#     num_workers=4,
# )

# Config format for the “complex” version (relative-path based)
# Per-root named list files
# If you have roots A, B, C as subdirectories under image_root, you keep list files like:
# <lists_root>/
#   A/
#     subset1.txt
#     subset2.txt
#   B/
#     train.txt
#   C/
#     all.txt
# Each list file contains relpaths within that root subdir:
# <lists_root>/A/subset1.txt
# subset2/image3.png
# subset2/image4.png
# Top-level dataset config (JSON or YAML)
# Example (ds_AB.yaml or ds_AB.json):
# dataset_name: ds_AB
# components:
#   - name: A_subset1
#     root_subdir: A
#     list: subset1
#   - name: B_train
#     root_subdir: B
#     list: train

# dedupe: by_path   # or: by_image_id
# Resulting resolved relpaths under image_root become:
# A/subset2/image3.png
# B/<whatever>/...
# So you still pass image_root exactly like the text-file mode.