"""On-disk cache for rasterized dense supervision tensors.

For each ``(image_id, rasterizer, normalizer, label_transform)`` tuple we
produce a fixed set of ``(target_map, loss_mask, pixel_weights)``. The
inputs are deterministic functions of the dataset, so we save the outputs
to disk keyed by a fingerprint of the configuration and reload them on
subsequent epochs.

The cache also maintains a ``manifest.json`` mapping each fingerprint to
the configs that produced it so users can audit what's cached and detect
orphaned entries.

Cache key composition (sha256, hex, truncated to the first 24 chars for
filesystem-friendly filenames):

    sha256( str(image_id) || rasterizer.cache_fingerprint() ||
            label_transform fingerprint )

The rasterizer fingerprint already includes the normalizer fingerprint
and both class versions.

Filesystem layout:

    <root>/
        manifest.json
        <fingerprint>/<image_id_safe>.pt

Each ``.pt`` file is a dict with keys ``target_map``, ``loss_mask``,
``pixel_weights``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import torch

from persistence_parameter_networks.data.dense_target_rasterizer import (
    TargetRasterizer,
)


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(image_id: str) -> str:
    return _SAFE_NAME_RE.sub("_", image_id)[:200]


def _label_transform_fingerprint(label_transform: Callable) -> str:
    """Deterministic-ish identifier for a label-transform callable.

    Functions don't carry a meaningful runtime version, so we hash the
    qualified name plus an optional ``__version__`` attribute. If you
    change a label_transform's semantics, *either* rename it *or* bump
    its ``__version__``."""
    inner = getattr(label_transform, "__wrapped__", label_transform)
    name = getattr(inner, "__qualname__", None) or getattr(inner, "__name__", repr(inner))
    version = int(getattr(inner, "__version__", 1))
    payload = {"name": str(name), "version": version}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


class DenseSupervisionCache:
    """Disk-backed cache for rasterized dense supervision tensors."""

    def __init__(
        self,
        root_dir: Optional[Path],
        rasterizer: TargetRasterizer,
        label_transform: Callable,
        *,
        enable: bool = True,
    ):
        self.rasterizer = rasterizer
        self.label_transform = label_transform
        self.enable = enable and root_dir is not None
        self.root_dir = Path(root_dir) if root_dir is not None else None

        self._fingerprint = self._compute_fingerprint()
        self._stage_dir = (
            self.root_dir / self._fingerprint if self.root_dir is not None else None
        )
        self._manifest_path = (
            self.root_dir / "manifest.json" if self.root_dir is not None else None
        )
        self._manifest_lock = threading.Lock()

        if self.enable:
            assert self._stage_dir is not None  # for type checker
            self._stage_dir.mkdir(parents=True, exist_ok=True)
            self._ensure_manifest_entry()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _compute_fingerprint(self) -> str:
        ras_fp = self.rasterizer.cache_fingerprint()
        lt_fp = _label_transform_fingerprint(self.label_transform)
        payload = {"rasterizer": ras_fp, "label_transform": lt_fp}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]

    def _ensure_manifest_entry(self) -> None:
        if self._manifest_path is None:
            return
        with self._manifest_lock:
            try:
                if self._manifest_path.exists():
                    manifest = json.loads(self._manifest_path.read_text())
                else:
                    manifest = {}
            except json.JSONDecodeError:
                manifest = {}

            if self._fingerprint not in manifest:
                manifest[self._fingerprint] = {
                    "rasterizer_class": type(self.rasterizer).__name__,
                    "rasterizer_version": int(type(self.rasterizer).__version__),
                    "rasterizer_config": self.rasterizer._config_dict(),
                    "normalizer_class": type(self.rasterizer.normalizer).__name__,
                    "normalizer_version": int(
                        type(self.rasterizer.normalizer).__version__
                    ),
                    "normalizer_config": self.rasterizer.normalizer._config_dict(),
                    "label_transform": getattr(
                        getattr(self.label_transform, "__wrapped__", self.label_transform),
                        "__qualname__",
                        None,
                    )
                    or getattr(
                        getattr(self.label_transform, "__wrapped__", self.label_transform),
                        "__name__",
                        repr(self.label_transform),
                    ),
                }
                tmp = self._manifest_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
                os.replace(tmp, self._manifest_path)

    def _entry_path(self, image_id: str, instance_ids: Sequence[int]) -> Optional[Path]:
        if self._stage_dir is None:
            return None
        # Per-entry hash binds the cached tensors to both the image and
        # the exact set of instances used to build them, so a val cap (or
        # any other instance filter) doesn't accidentally serve stale
        # tensors built from a different instance set.
        sorted_ids = tuple(int(i) for i in sorted(instance_ids))
        per_entry = hashlib.sha256(
            json.dumps({"image_id": str(image_id), "instance_ids": sorted_ids}, sort_keys=True)
            .encode("utf-8")
        ).hexdigest()[:16]
        return self._stage_dir / f"{_safe_filename(image_id)}__{per_entry}.pt"

    def get(
        self,
        image_id: str,
        instance_ids: Sequence[int],
        instance_masks: torch.Tensor,
        instance_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(target_map, loss_mask, pixel_weights)``.

        Loads from disk if a cached entry exists; otherwise computes via
        the rasterizer and writes the result back. When ``enable=False``
        always recomputes (no disk I/O)."""
        if self.enable:
            path = self._entry_path(image_id, instance_ids)
            if path is not None and path.exists():
                try:
                    blob = torch.load(path, map_location="cpu")
                    return (
                        blob["target_map"],
                        blob["loss_mask"],
                        blob["pixel_weights"],
                    )
                except Exception as exc:
                    print(
                        f"[DenseSupervisionCache] WARN: failed to read {path}: "
                        f"{exc!r}; recomputing."
                    )

        target, loss_mask, pixel_weights = self.rasterizer.rasterize(
            instance_masks, instance_labels
        )

        if self.enable:
            path = self._entry_path(image_id, instance_ids)
            if path is not None:
                tmp = path.with_suffix(".pt.tmp")
                torch.save(
                    {
                        "target_map": target,
                        "loss_mask": loss_mask,
                        "pixel_weights": pixel_weights,
                    },
                    tmp,
                )
                os.replace(tmp, path)

        return target, loss_mask, pixel_weights
