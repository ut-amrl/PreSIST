"""Per-anchor-observation torch Dataset for vision persistence training.

For each `TrainingTrack` with a `first_missing_time`, defines a per-track
pseudo end-time `t_end = last_observed + (first_missing - last_observed)/2`
and emits one sample per positive observation that has both an image and a
mask path. Each sample's supervision is `log(t_end - anchor_time)`.

Tracks without a first-absent-after observation are skipped entirely (no
`t_end` can be defined).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from persistence_eval.methods.training.base import TrainingTrack


def _make_image_transform(size: int):
    return T.Compose([
        T.Resize((size, size), interpolation=InterpolationMode.BILINEAR),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@dataclass
class _AnchorRecord:
    track_id: str
    image_path: str
    mask_path: str
    anchor_time: float
    target_seconds: float


class AnchorMidpointDataset(Dataset):
    """Per-anchor (image, mask, log-target) samples derived from TrainingTracks."""

    def __init__(
        self,
        tracks: Iterable[TrainingTrack],
        *,
        manifest_dir: Path | None,
        image_size: int = 518,
    ) -> None:
        self._manifest_dir = Path(manifest_dir) if manifest_dir is not None else None
        self._image_size = image_size
        self._image_tx = _make_image_transform(image_size)
        self._records: list[_AnchorRecord] = []

        for track in tracks:
            tm = track.bounds.first_missing_time
            tl = track.bounds.last_observed_time
            if tm is None:
                continue
            t_end = tl + (tm - tl) / 2.0
            for i, t in enumerate(track.observation_times):
                if not track.is_positive[i]:
                    continue
                ip = track.image_paths[i] if i < len(track.image_paths) else None
                mp = track.mask_paths[i] if i < len(track.mask_paths) else None
                if ip is None or mp is None:
                    continue
                target_s = t_end - t
                if target_s <= 0.0:
                    continue
                self._records.append(_AnchorRecord(
                    track_id=track.track_id,
                    image_path=ip,
                    mask_path=mp,
                    anchor_time=t,
                    target_seconds=target_s,
                ))

    def __len__(self) -> int:
        return len(self._records)

    @property
    def track_ids(self) -> list[str]:
        return [r.track_id for r in self._records]

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute() or self._manifest_dir is None:
            return path
        return self._manifest_dir / path

    def __getitem__(self, idx: int) -> dict:
        rec = self._records[idx]

        img_path = self._resolve(rec.image_path)
        mask_path = self._resolve(rec.mask_path)

        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            arr = np.array(im, dtype=np.uint8)
        image = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W) uint8
        image = self._image_tx(image)

        with Image.open(mask_path) as mk:
            mk = mk.convert("L")
            marr = np.array(mk)
        mask_bool = torch.from_numpy(marr > 0).to(torch.float32).unsqueeze(0)  # (1,H,W)
        # Resize mask to image_size with nearest to preserve binary edges.
        mask_resized = torch.nn.functional.interpolate(
            mask_bool.unsqueeze(0),
            size=(self._image_size, self._image_size),
            mode="nearest",
        ).squeeze(0)

        target_seconds = torch.tensor(rec.target_seconds, dtype=torch.float32)
        target = torch.tensor(math.log(rec.target_seconds), dtype=torch.float32)

        return {
            "image": image,
            "mask": mask_resized,
            "target": target,                # supervision: log(t_end - anchor_time)
            "target_seconds": target_seconds,  # raw seconds, for original-space metrics
        }
