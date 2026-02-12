# image_id_pipeline/relpath_id_fn.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .registry import ImageRegistry, RegistryConfig


@dataclass
class RelpathToImageId:
    """
    A picklable, worker-safe Callable[[str], str].

    Key properties:
      - No DB connection is opened in the parent process.
      - Each worker lazily opens its own read-only sqlite connection on first call.
      - Works with fork and spawn DataLoader start methods.
    """
    image_root: str
    registry_cfg: RegistryConfig

    # Lazily initialized per-process
    _root: Optional[Path] = None
    _registry: Optional[ImageRegistry] = None

    def _get(self) -> tuple[Path, ImageRegistry]:
        if self._registry is None:
            self._root = Path(self.image_root).resolve()
            # Force read-only in workers; ignore cfg.strict/export_human_readable here
            self._registry = ImageRegistry(self.registry_cfg, read_only=True)
        return self._root, self._registry

    def __call__(self, relpath: str) -> str:
        root, reg = self._get()
        image_id = reg.path_to_id(root, relpath)
        if image_id is None:
            raise KeyError(f"relpath not found in registry: {relpath!r} (root={root})")
        return image_id

    def close(self) -> None:
        # Optional explicit cleanup (usually not required)
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

