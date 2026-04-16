# image_id_pipeline/dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .hashing import ImageIdComputer
from .registry import ImageRegistry
from .resolver import ResolvedItem, resolve_paths
from utils.image_utils import LazyImage


class RegistryBackedImageDataset(Dataset):
    """
    Returns per item:
      - path (abs str)
      - image (tensor)
      - image_id (str | None)
      - relpath (relative to image_root)
      - component (provenance label)
      - index, file_size, mtime_ns
    """

    def __init__(
        self,
        image_root: str,
        registry_db_path: str,
        image_loader: Callable[[str], torch.Tensor],
        *,
        # Either provide resolved_items OR a spec_path (+ optional lists_root)
        resolved_items: Optional[Sequence[ResolvedItem]] = None,
        spec_path: Optional[str] = None,
        lists_root: Optional[str] = None,
        id_computer: Optional[ImageIdComputer] = None,
        strict: bool = True,
        compute_missing_ids: bool = False,  # recommend False when you preprocess
        lazy_image_loading: bool = False,  # if True, skip image load in __getitem__; "image" will be None
    ):
        self.root = Path(image_root).resolve()
        self.registry_db_path = str(registry_db_path)
        self.image_loader = image_loader
        self.id_computer = id_computer or ImageIdComputer()
        self.strict = strict
        self.compute_missing_ids = compute_missing_ids
        self.lazy_image_loading = lazy_image_loading

        # Registry is created lazily in each worker process
        self._registry: Optional[ImageRegistry] = None

        if resolved_items is None:
            if spec_path is None:
                raise ValueError("Provide either resolved_items or spec_path")
            resolved_items, _meta = resolve_paths(image_root=image_root, spec_path=spec_path, lists_root=lists_root)

        self.items: List[ResolvedItem] = list(resolved_items)

        if self.strict:
            missing = []
            for it in self.items:
                p = self.root / it.relpath
                if not p.exists() or not p.is_file():
                    missing.append(it.relpath)
            if missing:
                raise FileNotFoundError(f"{len(missing)} missing files. First few: {missing[:5]}")

    def _get_registry(self) -> ImageRegistry:
        if self._registry is None:
            # read-only is what you want inside workers
            self._registry = ImageRegistry.from_db_path(self.registry_db_path, read_only=not self.compute_missing_ids)
        return self._registry

    def __len__(self) -> int:
        return len(self.items)

    def get_metadata(self, index: int) -> Dict[str, Any]:
        """
        Cheap metadata-only lookup.
        Does NOT load the image tensor.
        Mirrors the logic in __getitem__ except it skips the image_loader.
        """
        it = self.items[index]
        abspath = (self.root / it.relpath).resolve()

        if self.strict and (not abspath.exists() or not abspath.is_file()):
            raise FileNotFoundError(f"Missing file: {it.relpath}")

        st = abspath.stat()
        file_size = int(st.st_size)
        mtime_ns = int(st.st_mtime_ns)

        registry = self._get_registry()
        row = registry.get_record(self.root, it.relpath)

        # Same logic as __getitem__ for determining image_id
        if row and int(row["file_size"]) == file_size and int(row["mtime_ns"]) == mtime_ns:
            image_id = row["image_id"]
        else:
            if not self.compute_missing_ids:
                image_id = None
            else:
                # Do NOT compute missing IDs here — metadata calls should stay cheap.
                # So we return None rather than computing & updating registry.
                image_id = None

        return {
            "index": index,
            "component": it.component,
            "relpath": it.relpath,
            "path": str(abspath),
            "image_id": image_id,
            "file_size": file_size,
            "mtime_ns": mtime_ns,
        }


    def __getitem__(self, index: int) -> Dict[str, Any]:
        it = self.items[index]
        abspath = (self.root / it.relpath).resolve()

        if self.strict and (not abspath.exists() or not abspath.is_file()):
            raise FileNotFoundError(f"Missing file: {it.relpath}")

        st = abspath.stat()
        file_size = int(st.st_size)
        mtime_ns = int(st.st_mtime_ns)

        # Return a LazyImage: deferred when lazy_image_loading=True, preloaded otherwise
        if self.lazy_image_loading:
            image = LazyImage(str(abspath), loader=self.image_loader)
        else:
            image = LazyImage(str(abspath), preloaded=self.image_loader(str(abspath)), loader=self.image_loader)

        registry = self._get_registry()

        # Lookup id from registry (preferred)
        row = registry.get_record(self.root, it.relpath)
        if row and int(row["file_size"]) == file_size and int(row["mtime_ns"]) == mtime_ns:
            image_id = row["image_id"]
        else:
            if not self.compute_missing_ids:
                image_id = None
            else:
                # If you enable this, be careful with num_workers>0 (DB writes).
                import time
                image_id = self.id_computer.compute(abspath)
                registry.upsert_record(
                    root=self.root,
                    relpath=it.relpath,
                    abspath=abspath,
                    image_id=image_id,
                    file_size=file_size,
                    mtime_ns=mtime_ns,
                    indexed_at_ns=time.time_ns(),
                )

        return {
            "index": index,
            "component": it.component,
            "relpath": it.relpath,
            "path": str(abspath),
            "image": image,
            "image_id": image_id,
            "file_size": file_size,
            "mtime_ns": mtime_ns,
        }


def make_collate_fn():
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
            "component": [b["component"] for b in batch],
            "relpath": [b["relpath"] for b in batch],
            "path": [b["path"] for b in batch],
            "image": [b["image"] for b in batch],
            "image_id": [b["image_id"] for b in batch],
            "file_size": torch.tensor([b["file_size"] for b in batch], dtype=torch.long),
            "mtime_ns": torch.tensor([b["mtime_ns"] for b in batch], dtype=torch.long),
        }
    return collate

