# image_id_pipeline/image_id_fn.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .registry import ImageRegistry, RegistryConfig


@dataclass
class ImageIdToRelpath:
    """
    A picklable, worker-safe Callable[[str], str].

    Properties:
      - No DB connection opened in parent process.
      - Each worker lazily opens its own read-only sqlite connection on first call.
      - If image_id is non-unique, returns the first (deterministic) and logs a warning.
    """
    image_root: str
    registry_cfg: RegistryConfig
    warn: Optional[Callable[[str], None]] = None  # e.g., print or logging.getLogger(...).warning

    # Lazily initialized per-process
    _root: Optional[Path] = None
    _registry: Optional[ImageRegistry] = None

    def _get(self) -> tuple[Path, ImageRegistry]:
        if self._registry is None:
            self._root = Path(self.image_root).resolve()
            self._registry = ImageRegistry(self.registry_cfg, read_only=True)
        return self._root, self._registry

    def __call__(self, image_id: str) -> str:
        root, reg = self._get()

        rows = reg.id_to_paths(str(image_id))  # [{'root': ..., 'relpath': ..., 'abspath': ...}, ...]
        if not rows:
            raise KeyError(f"image_id not found in registry: {image_id!r} (root={root})")

        # Prefer entries matching *this* image_root if registry contains multiple roots.
        root_s = str(root)
        matching = [r for r in rows if str(r.get("root")) == root_s]

        chosen_list = matching if matching else rows

        if len(chosen_list) > 1:
            warn_fn = self.warn or print
            warn_fn(
                f"[ImageIdToRelpath] image_id={image_id!r} has {len(chosen_list)} paths "
                f"{'(filtered to current root)' if matching else '(across all roots)'}; "
                f"returning the first: root={chosen_list[0].get('root')!r} relpath={chosen_list[0].get('relpath')!r}"
            )

        relpath = chosen_list[0].get("relpath")
        if not isinstance(relpath, str) or not relpath:
            raise TypeError(f"Bad registry row for image_id={image_id!r}: {chosen_list[0]!r}")

        return relpath

    def close(self) -> None:
        if self._registry is not None:
            self._registry.close()
            self._registry = None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_registry"] = None
        d["_root"] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
