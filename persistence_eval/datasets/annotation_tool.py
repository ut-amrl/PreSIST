from __future__ import annotations

import json
import random
from itertools import groupby
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from persistence_eval.core.domain import (
    AnchorObservation,
    BinaryObservationEvent,
    ImageMaskGroup,
    TrackTemporalBounds,
)
from persistence_eval.datasets.backing import AbsBackingDataset
from persistence_eval.datasets.image_mask_view import ImageMaskView
from persistence_eval.datasets.track_view import TrackView

_US_TO_S = 1_000_000.0
_POSITIVE = "Positive"


def _remap_path(path_str: str, host_root: str | None, container_root: str | None) -> str:
    if host_root is not None and container_root is not None and path_str.startswith(host_root):
        return container_root + path_str[len(host_root):]
    return path_str


def _remap_track_mask_paths(
    raw_tracks: list[dict], host_root: str, container_root: str
) -> list[dict]:
    result = []
    for track in raw_tracks:
        new_obs = []
        for obs in track["observations"]:
            mask = obs.get("mask_path")
            if mask is not None:
                obs = {**obs, "mask_path": _remap_path(mask, host_root, container_root)}
            new_obs.append(obs)
        result.append({**track, "observations": new_obs})
    return result


def _has_positive_on_both_sides(track: dict, boundary_us: int) -> bool:
    has_before = any(
        obs["state"] == _POSITIVE and obs["timestamp"] < boundary_us
        for obs in track["observations"]
    )
    has_after = any(
        obs["state"] == _POSITIVE and obs["timestamp"] >= boundary_us
        for obs in track["observations"]
    )
    return has_before and has_after


def _filter_track_observations(track: dict, boundary_us: int, before: bool) -> dict:
    """Return a shallow copy of track with observations restricted to one side of boundary_us."""
    if before:
        kept = [obs for obs in track["observations"] if obs["timestamp"] < boundary_us]
    else:
        kept = [obs for obs in track["observations"] if obs["timestamp"] >= boundary_us]
    return {**track, "observations": kept}


class AnnotationToolBackingDataset(AbsBackingDataset):
    """
    Backing dataset for annotation-tool export format.
    Loads a manifest JSON plus one JSON file per track. Both views share the
    parsed track data; per-view caches are populated lazily on first access.
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        name: str | None = None,
        image_mask_view=None,
        track_view=None,
        boundary_timestamp: float | None = None,
        before: bool | None = None,
        max_anchor_observations: int | None = None,
        shuffle_seed: int | None = None,
        host_root: str | None = None,
        container_root: str | None = None,
    ) -> None:
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        self._run_id: str = manifest["run_id"]
        self._location_root = Path(
            _remap_path(manifest["location_root"], host_root, container_root)
        )
        manifest_dir = manifest_path.parent
        raw_tracks = [
            json.loads((manifest_dir / rel).read_text())
            for rel in manifest["track_files"]
        ]
        if host_root is not None and container_root is not None:
            raw_tracks = _remap_track_mask_paths(raw_tracks, host_root, container_root)
        self._raw_tracks = self._apply_boundary(raw_tracks, boundary_timestamp, before)
        self._max_anchor_observations = max_anchor_observations
        self._shuffle_seed = shuffle_seed
        self._track_obs_cache: list[tuple[str, list[dict], dict]] | None = None
        self._image_obs_cache: list[tuple[str, dict]] | None = None

    @classmethod
    def _from_raw(
        cls,
        run_id: str,
        location_root: Path,
        raw_tracks: list[dict],
        boundary_timestamp: float | None = None,
        before: bool | None = None,
        max_anchor_observations: int | None = None,
        shuffle_seed: int | None = None,
    ) -> AnnotationToolBackingDataset:
        """Construct from already-loaded data, bypassing file I/O."""
        obj = object.__new__(cls)
        obj._run_id = run_id
        obj._location_root = location_root
        obj._raw_tracks = cls._apply_boundary(raw_tracks, boundary_timestamp, before)
        obj._max_anchor_observations = max_anchor_observations
        obj._shuffle_seed = shuffle_seed
        obj._track_obs_cache = None
        obj._image_obs_cache = None
        return obj

    @staticmethod
    def _apply_boundary(
        raw_tracks: list[dict],
        boundary_timestamp: float | None,
        before: bool | None,
    ) -> list[dict]:
        if boundary_timestamp is None:
            return raw_tracks
        boundary_us = int(boundary_timestamp * _US_TO_S)
        filtered = [t for t in raw_tracks if not _has_positive_on_both_sides(t, boundary_us)]
        if before is None:
            return filtered
        return [_filter_track_observations(t, boundary_us, before) for t in filtered]

    @property
    def name(self) -> str:
        return self._run_id

    def _get_track_observations(self) -> list[tuple[str, list[dict], dict]]:
        if self._track_obs_cache is None:
            self._track_obs_cache = [
                (t["track_id"], sorted(t["observations"], key=lambda o: o["timestamp"]), t["boundaries"])
                for t in self._raw_tracks
            ]
        return self._track_obs_cache

    def _get_image_observations(self) -> list[tuple[str, dict]]:
        if self._image_obs_cache is None:
            obs_list = [
                (t["track_id"], obs)
                for t in self._raw_tracks
                for obs in t["observations"]
                if obs["state"] == _POSITIVE and obs.get("mask_path") is not None
            ]
            if self._shuffle_seed is not None:
                random.Random(self._shuffle_seed).shuffle(obs_list)
            if self._max_anchor_observations is not None:
                obs_list = obs_list[: self._max_anchor_observations]
            self._image_obs_cache = obs_list
        return self._image_obs_cache


class AnnotationToolImageMaskView(ImageMaskView):
    _backing: AnnotationToolBackingDataset

    def iter_anchor_observations(self) -> Iterable[AnchorObservation]:
        root = self._backing._location_root
        for track_id, obs in self._backing._get_image_observations():
            image = torch.from_numpy(
                np.array(Image.open(root / obs["image_path"]).convert("RGB"), dtype=np.uint8)
            ).permute(2, 0, 1)  # HWC → CHW
            mask = torch.from_numpy(
                np.array(Image.open(root / obs["mask_path"]).convert("L")) > 0
            )
            anchor_time = obs["timestamp"] / _US_TO_S
            yield AnchorObservation(image=image, mask=mask, anchor_time=anchor_time, track_id=track_id)

    def iter_image_groups(self) -> Iterable[ImageMaskGroup]:
        """Group all masks sharing an image into one batch, loading each image once."""
        root = self._backing._location_root
        sorted_obs = sorted(self._backing._get_image_observations(), key=lambda x: x[1]["image_path"])
        for image_path, group in groupby(sorted_obs, key=lambda x: x[1]["image_path"]):
            group_list = list(group)
            image = torch.from_numpy(
                np.array(Image.open(root / image_path).convert("RGB"), dtype=np.uint8)
            ).permute(2, 0, 1)  # HWC → CHW
            mask_list = [
                torch.from_numpy(np.array(Image.open(root / obs["mask_path"]).convert("L")) > 0)
                for _, obs in group_list
            ]
            mask_paths = [str(root / obs["mask_path"]) for _, obs in group_list]
            track_ids = [track_id for track_id, _ in group_list]
            anchor_time = group_list[0][1]["timestamp"] / _US_TO_S
            yield ImageMaskGroup(
                image=image,
                masks=torch.stack(mask_list, dim=0),  # (K, H, W)
                track_ids=track_ids,
                anchor_time=anchor_time,
                image_path=str(root / image_path),
                mask_paths=mask_paths,
            )


class AnnotationToolTrackView(TrackView):
    _backing: AnnotationToolBackingDataset

    def iter_tracks(self) -> Iterable[tuple[str, list[BinaryObservationEvent], TrackTemporalBounds]]:
        for track_id, obs_list, boundaries in self._backing._get_track_observations():
            events = [
                BinaryObservationEvent(
                    timestamp=obs["timestamp"] / _US_TO_S,
                    is_positive=obs["state"] == _POSITIVE,
                )
                for obs in obs_list
                if obs["state"] != "NotVisible"
            ]
            b = boundaries
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
            yield track_id, events, bounds


def split_at_boundary(
    backing: AnnotationToolBackingDataset,
    boundary_timestamp: float,
) -> tuple[AnnotationToolBackingDataset, AnnotationToolBackingDataset]:
    """
    Split a backing dataset into before/after halves at boundary_timestamp (seconds).
    Tracks with Positive observations on both sides of the boundary are dropped from both halves.
    Reuses already-loaded track data — no file I/O.
    """
    before_ds = AnnotationToolBackingDataset._from_raw(
        backing._run_id,
        backing._location_root,
        backing._raw_tracks,
        boundary_timestamp=boundary_timestamp,
        before=True,
        max_anchor_observations=backing._max_anchor_observations,
        shuffle_seed=backing._shuffle_seed,
    )
    after_ds = AnnotationToolBackingDataset._from_raw(
        backing._run_id,
        backing._location_root,
        backing._raw_tracks,
        boundary_timestamp=boundary_timestamp,
        before=False,
        max_anchor_observations=backing._max_anchor_observations,
        shuffle_seed=backing._shuffle_seed,
    )
    return before_ds, after_ds
