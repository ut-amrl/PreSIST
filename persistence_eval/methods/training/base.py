"""Trainer interface and training dataset wrapper.

Each `AbsTrainer.train(dataset)` owns its own iteration loop so closed-form
fits, EM, and SGD trainers can each pull data on their own terms. Returns a
`{key -> ParamEntry}` dict that the orchestrator serializes to JSON.

The orchestrator wraps an annotation-tool BackingDataset in `TrainingDataset`
which surfaces:
  - iter_tracks(): per-track durations + per-track observation sequences,
    grouped under their bucket key (class_id or location_key).
  - as_torch_dataset(include_images): torch Dataset of per-frame samples
    for vision trainers (loads images on demand).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from persistence_eval.core.domain import BinaryObservationEvent, TrackTemporalBounds


# A single bucket-keyed entry in the params JSON, matching the disk schema:
#   {"params": {...}, "confidence": float|null, "metadata": {...}}
ParamEntry = dict


@dataclass
class TrainingTrack:
    """Per-track sample bundle consumed by training code.

    `observation_times` and `is_positive` parallel each other and are sorted
    by time; they're the inputs upstream EM expects (observation_times,
    y_bool). `duration_s` is the convenience scalar most native trainers use.

    `image_paths` and `mask_paths` parallel `observation_times` and carry the
    per-observation file paths (relative to the manifest dir) when available.
    Vision trainers consume them; distribution-fit trainers ignore them.
    """
    track_id: str
    class_id: int
    location_key: str
    observation_times: list[float]   # seconds
    is_positive: list[bool]
    bounds: TrackTemporalBounds
    image_paths: list[str | None] = field(default_factory=list)
    mask_paths: list[str | None] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.bounds.last_observed_time - self.bounds.first_observed_time


class TrainingDataset:
    """Owns the train manifest and the bucket-keying logic for trainers.

    Wraps a list of raw annotation-tool track dicts plus a `cell_size_px`
    location-keying parameter. Subsamples deterministically under
    `subsample_seed` and `subsample_fraction`.

    When `val_track_fraction > 0`, a deterministic fraction of the raw tracks
    is held out for validation: `iter_tracks` yields only the train remainder
    (then subsampled), while `iter_val_tracks` yields the held-out portion
    untouched by `subsample_fraction`. Distribution-fit trainers leave
    `val_track_fraction=0` and see the original behavior.
    """

    def __init__(
        self,
        raw_tracks: list[dict],
        *,
        cell_size_px: int,
        class_id: int = 0,
        subsample_fraction: float = 1.0,
        subsample_seed: int = 0,
        val_track_fraction: float = 0.0,
        val_split_seed: int = 0,
        manifest_dir: Path | None = None,
        external_val_raw_tracks: list[dict] | None = None,
    ) -> None:
        if not 0.0 < subsample_fraction <= 1.0:
            raise ValueError(
                f"subsample_fraction must be in (0, 1]; got {subsample_fraction}"
            )
        if not 0.0 <= val_track_fraction < 1.0:
            raise ValueError(
                f"val_track_fraction must be in [0, 1); got {val_track_fraction}"
            )
        if external_val_raw_tracks is not None and val_track_fraction > 0.0:
            raise ValueError(
                "Pass either external_val_raw_tracks or val_track_fraction>0, not both."
            )
        self._raw_tracks = list(raw_tracks)
        self._cell_size_px = cell_size_px
        self._class_id = class_id
        self._subsample_fraction = subsample_fraction
        self._subsample_seed = subsample_seed
        self._val_track_fraction = val_track_fraction
        self._val_split_seed = val_split_seed
        self._manifest_dir = Path(manifest_dir) if manifest_dir is not None else None
        self._external_val_raw_tracks = (
            list(external_val_raw_tracks) if external_val_raw_tracks is not None else None
        )

    def _split_raw_tracks(self) -> tuple[list[dict], list[dict]]:
        """Return (train_pool, val_pool) by deterministic split. Subsampling
        is NOT applied here — only the train/val partition."""
        if self._external_val_raw_tracks is not None:
            return list(self._raw_tracks), list(self._external_val_raw_tracks)
        if self._val_track_fraction <= 0.0:
            return list(self._raw_tracks), []
        shuffled = list(self._raw_tracks)
        random.Random(self._val_split_seed).shuffle(shuffled)
        n_val = max(1, int(round(self._val_track_fraction * len(shuffled))))
        val = shuffled[:n_val]
        train = shuffled[n_val:]
        return train, val

    def _subsampled_raw_tracks(self) -> list[dict]:
        train, _ = self._split_raw_tracks()
        if self._subsample_fraction >= 1.0:
            return train
        shuffled = list(train)
        random.Random(self._subsample_seed).shuffle(shuffled)
        n = max(1, int(self._subsample_fraction * len(shuffled)))
        return shuffled[:n]

    def _val_raw_tracks(self) -> list[dict]:
        _, val = self._split_raw_tracks()
        return val

    def _make_track(self, track: dict) -> TrainingTrack:
        from persistence_eval.methods.training.location_keying import (
            location_key_for_track,
        )

        _US_TO_S = 1_000_000.0
        obs = sorted(track["observations"], key=lambda o: o["timestamp"])
        obs = [o for o in obs if o["state"] != "NotVisible"]
        obs_times = [o["timestamp"] / _US_TO_S for o in obs]
        is_pos = [o["state"] == "Positive" for o in obs]
        image_paths = [o.get("image_path") for o in obs]
        mask_paths = [o.get("mask_path") for o in obs]
        b = track["boundaries"]
        bounds = TrackTemporalBounds(
            first_observed_time=b["first_observed_ts"] / _US_TO_S,
            last_observed_time=b["last_observed_ts"] / _US_TO_S,
            first_absent_before_observation=(
                b["last_absent_before_ts"] / _US_TO_S
                if b.get("last_absent_before_ts") is not None else None
            ),
            first_missing_time=(
                b["first_absent_after_ts"] / _US_TO_S
                if b.get("first_absent_after_ts") is not None else None
            ),
        )
        return TrainingTrack(
            track_id=track["track_id"],
            class_id=self._class_id,
            location_key=location_key_for_track(track, self._cell_size_px),
            observation_times=obs_times,
            is_positive=is_pos,
            bounds=bounds,
            image_paths=image_paths,
            mask_paths=mask_paths,
        )

    def iter_tracks(self) -> Iterator[TrainingTrack]:
        for track in self._subsampled_raw_tracks():
            yield self._make_track(track)

    def iter_val_tracks(self) -> Iterator[TrainingTrack]:
        for track in self._val_raw_tracks():
            yield self._make_track(track)

    def n_tracks(self) -> int:
        return len(self._subsampled_raw_tracks())

    def n_val_tracks(self) -> int:
        return len(self._val_raw_tracks())

    @property
    def subsample_fraction(self) -> float:
        return self._subsample_fraction

    @property
    def subsample_seed(self) -> int:
        return self._subsample_seed

    @property
    def val_track_fraction(self) -> float:
        return self._val_track_fraction

    @property
    def val_split_seed(self) -> int:
        return self._val_split_seed

    @property
    def manifest_dir(self) -> Path | None:
        return self._manifest_dir


class AbsTrainer(Protocol):
    """Trainers own their iteration over the dataset.

    `grouping` is "class" or "location"; trainers bucket TrainingTracks
    accordingly before fitting. The returned dict is keyed by the bucket
    identifier (str(class_id) or location_key) and includes a "__global__"
    entry built by pooling all tracks for use as a fallback at eval time.
    """

    grouping: str  # "class" | "location"
    name: str      # variant identifier; appears in metadata

    def train(self, dataset: TrainingDataset) -> dict[str, ParamEntry]: ...


def load_raw_tracks(manifest_path: str | Path) -> list[dict]:
    """Read an annotation-tool manifest and return the list of raw track dicts."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    manifest_dir = manifest_path.parent
    return [
        json.loads((manifest_dir / rel).read_text())
        for rel in manifest["track_files"]
    ]


def _remap_path(path_str: str, host_root: str | None, container_root: str | None) -> str:
    if host_root is not None and container_root is not None and path_str.startswith(host_root):
        return container_root + path_str[len(host_root):]
    return path_str


def load_manifest_with_resolved_paths(
    manifest_path: str | Path,
    *,
    host_root: str | None = None,
    container_root: str | None = None,
) -> list[dict]:
    """Load an annotation-tool manifest and return raw tracks with image_path
    and mask_path resolved to absolute container-side paths.

    Image paths in observations are relative to `manifest["location_root"]`;
    mask paths are absolute host-side paths. Both are remapped to the
    container filesystem when host_root/container_root are provided.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    manifest_dir = manifest_path.parent
    location_root_str = _remap_path(manifest["location_root"], host_root, container_root)
    location_root = Path(location_root_str)
    raw_tracks = [
        json.loads((manifest_dir / rel).read_text())
        for rel in manifest["track_files"]
    ]
    resolved: list[dict] = []
    for track in raw_tracks:
        new_obs = []
        for obs in track["observations"]:
            new = dict(obs)
            ip = new.get("image_path")
            if ip is not None:
                new["image_path"] = str(location_root / ip)
            mp = new.get("mask_path")
            if mp is not None:
                new["mask_path"] = _remap_path(mp, host_root, container_root)
            new_obs.append(new)
        resolved.append({**track, "observations": new_obs})
    return resolved


def bucket_tracks(
    tracks: Iterable[TrainingTrack], grouping: str
) -> dict[str, list[TrainingTrack]]:
    """Group tracks by bucket key (class_id stringified, or location_key)."""
    buckets: dict[str, list[TrainingTrack]] = {}
    for t in tracks:
        if grouping == "class":
            key = str(t.class_id)
        elif grouping == "location":
            key = t.location_key
        else:
            raise ValueError(f"grouping must be 'class' or 'location'; got {grouping!r}")
        buckets.setdefault(key, []).append(t)
    return buckets
