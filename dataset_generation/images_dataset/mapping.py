# image_id_pipeline/mapping.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from .registry import ImageRegistry


def _normalize_relpath_user(relpath: str) -> str:
    """
    Normalize like ImageRegistry._normalize_relpath, but keep it local
    (don't import private helpers).
    """
    rel = relpath.replace("\\", "/").lstrip("/")
    if rel == ".." or rel.startswith("../") or "/../" in rel:
        raise ValueError(f"Refusing relpath with parent traversal: {relpath}")
    if rel == "":
        raise ValueError("relpath cannot be empty")
    return rel


@dataclass
class RegistryMapping:
    """
    Read-only mapping interface backed by ImageRegistry SQLite DB.

    Safe usage patterns:
      - main process: keep one instance open
      - DataLoader workers: construct one per worker (don't share across fork)
    """
    image_root: Path
    registry_db_path: Path

    def __post_init__(self) -> None:
        self.image_root = Path(self.image_root).resolve()
        self.registry_db_path = Path(self.registry_db_path)

        # Enforce read-only DB access
        self._registry = ImageRegistry.from_db_path(self.registry_db_path, read_only=True)

        # Root string must match what was stored during indexing: str(Path(image_root).resolve())
        self._root_s = str(self.image_root)

    def close(self) -> None:
        self._registry.close()

    def __enter__(self) -> "RegistryMapping":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def relpath_to_image_id(self, relpath: str) -> str:
        rp = _normalize_relpath_user(relpath)
        image_id = self._registry.path_to_id(self.image_root, rp)
        if image_id is None:
            raise KeyError(f"relpath not found in registry: {rp!r} (root={self._root_s})")
        return image_id

    def image_id_to_relpath(self, image_id: str) -> str:
        """
        Reverse mapping scoped to this image_root.
        If the same image_id exists under multiple roots, we pick the one matching this root.
        """
        rows = self._registry.id_to_paths(image_id)
        for r in rows:
            if str(r.get("root")) == self._root_s:
                return str(r["relpath"])
        raise KeyError(f"image_id not found for root={self._root_s}: {image_id!r}")


def make_relpath_to_image_id(
    *,
    image_root: str,
    registry_root: str,
) -> Tuple[Callable[[str], str], Callable[[], None]]:
    """
    Factory returning:
      - relpath_to_image_id: Callable[[str], str]
      - close(): cleanup function

    Uses <registry_root>/registry.sqlite3 (your ImageRegistry.DEFAULT_DB_NAME).
    """
    image_root_p = Path(image_root).resolve()
    db_path = Path(registry_root).resolve() / ImageRegistry.DEFAULT_DB_NAME

    mapping = RegistryMapping(image_root=image_root_p, registry_db_path=db_path)
    return mapping.relpath_to_image_id, mapping.close


def make_mapping(
    *,
    image_root: str,
    registry_root: str,
) -> RegistryMapping:
    """
    Convenience factory when you want both directions and context manager support.
    """
    db_path = Path(registry_root).resolve() / ImageRegistry.DEFAULT_DB_NAME
    return RegistryMapping(image_root=Path(image_root), registry_db_path=db_path)
