from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_UNDER_CAP_WARN_LIMIT = 10


def _per_track_balanced_sample(
    obs_list: list[tuple[str, dict]],
    target_n: int,
    seed: int | None,
) -> list[tuple[str, dict]]:
    """Subsample ``obs_list`` so each track contributes roughly equally.

    Picks the smallest per-track cap ``K`` such that ``sum(min(c_i, K))`` is at
    least ``target_n``, randomly subsamples each track to that cap, then shuffles
    the concatenated result and slices to exactly ``target_n``. Tracks with
    fewer than ``K`` observations are warned about (rate-limited) and counted
    in a final summary log.
    """
    if target_n <= 0 or not obs_list:
        return []

    rng = random.Random(seed)

    by_track: dict[str, list[dict]] = defaultdict(list)
    for track_id, obs in obs_list:
        by_track[track_id].append(obs)

    counts = sorted((len(v) for v in by_track.values()), reverse=False)
    total_obs = sum(counts)
    if target_n >= total_obs:
        # Budget exceeds available data: keep everything, shuffle for consistency.
        flat = [(tid, obs) for tid, obs_l in by_track.items() for obs in obs_l]
        rng.shuffle(flat)
        return flat

    # Find smallest integer K with sum(min(c_i, K)) >= target_n via binary
    # search (monotone non-decreasing in K). The unique solution gives the
    # tightest per-track balance — fewest observations to trim post-shuffle.
    lo, hi = 1, max(counts)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(min(c, mid) for c in counts) >= target_n:
            hi = mid
        else:
            lo = mid + 1
    chosen_k = lo

    under_cap_tracks: list[tuple[str, int]] = []
    sampled: list[tuple[str, dict]] = []
    for track_id, obs_l in by_track.items():
        if len(obs_l) <= chosen_k:
            if len(obs_l) < chosen_k:
                under_cap_tracks.append((track_id, len(obs_l)))
            for obs in obs_l:
                sampled.append((track_id, obs))
        else:
            picked = rng.sample(obs_l, chosen_k)
            for obs in picked:
                sampled.append((track_id, obs))

    rng.shuffle(sampled)
    trimmed = sampled[:target_n]

    total_tracks = len(by_track)
    n_under = len(under_cap_tracks)
    for track_id, count in under_cap_tracks[:_UNDER_CAP_WARN_LIMIT]:
        logger.warning(
            "per_track_balanced: track %s has %d observations (< cap K=%d)",
            track_id, count, chosen_k,
        )
    if n_under > _UNDER_CAP_WARN_LIMIT:
        logger.warning(
            "per_track_balanced: ... %d additional under-cap tracks suppressed.",
            n_under - _UNDER_CAP_WARN_LIMIT,
        )
    frac = n_under / total_tracks if total_tracks else 0.0
    logger.warning(
        "per_track_balanced summary: K=%d, %d/%d tracks under cap (%.1f%%); "
        "kept %d/%d observations (target %d).",
        chosen_k, n_under, total_tracks, 100.0 * frac,
        len(trimmed), total_obs, target_n,
    )
    return trimmed

from persistence_eval.core.domain import (
    AnchorObservation,
    BinaryObservationEvent,
    ImageMaskGroup,
    TrackTemporalBounds,
)
from persistence_eval.datasets.backing import AbsBackingDataset
from persistence_eval.datasets.image_mask_view import ImageMaskView
from persistence_eval.datasets._mask_io import load_mask_bool
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
        per_track_balanced: bool = False,
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
        self._per_track_balanced = per_track_balanced
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
        per_track_balanced: bool = False,
    ) -> AnnotationToolBackingDataset:
        """Construct from already-loaded data, bypassing file I/O."""
        obj = object.__new__(cls)
        obj._run_id = run_id
        obj._location_root = location_root
        obj._raw_tracks = cls._apply_boundary(raw_tracks, boundary_timestamp, before)
        obj._max_anchor_observations = max_anchor_observations
        obj._shuffle_seed = shuffle_seed
        obj._per_track_balanced = per_track_balanced
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
            if self._per_track_balanced and self._max_anchor_observations is not None:
                obs_list = _per_track_balanced_sample(
                    obs_list,
                    target_n=self._max_anchor_observations,
                    seed=self._shuffle_seed,
                )
            else:
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
            mask = load_mask_bool(root / obs["mask_path"])
            anchor_time = obs["timestamp"] / _US_TO_S
            yield AnchorObservation(image=image, mask=mask, anchor_time=anchor_time, track_id=track_id)

    def num_anchor_observations(self) -> int:
        return len(self._backing._get_image_observations())

    def num_image_groups(self) -> int:
        return len({obs["image_path"] for _, obs in self._backing._get_image_observations()})

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
                load_mask_bool(root / obs["mask_path"])
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


def MultiAnnotationToolBackingDataset(
    sources: list[dict],
    *,
    name: str | None = None,
    image_mask_view=None,
    track_view=None,
    max_anchor_observations: int | None = None,
    shuffle_seed: int | None = None,
    per_track_balanced: bool = False,
) -> AnnotationToolBackingDataset:
    """Combine multiple annotation-tool manifest exports into one backing dataset.

    Each entry in ``sources`` is a dict with keys ``manifest_path`` (required),
    ``host_root`` / ``container_root`` (optional, for paths needing prefix swap),
    and ``source_name`` (optional, used to namespace track_id / space_id). We
    load each source via a temporary ``AnnotationToolBackingDataset`` (so the
    existing parse + remap logic is reused), then rewrite every observation's
    ``image_path`` to be absolute by prepending the per-source ``location_root``.
    Mask paths are already absolute. With both paths absolute, the views' usual
    ``root / path`` joins become no-ops (absolute right-hand side wins under
    ``pathlib``), so we pass ``Path("/")`` as the combined ``location_root``.

    Globally applied ``max_anchor_observations`` / ``shuffle_seed`` /
    ``per_track_balanced`` mean balancing happens across the namespaced track ids
    of all sources — each (source, track) pair contributes roughly equally.
    """
    if not sources:
        raise ValueError("MultiAnnotationToolBackingDataset: sources is empty")

    combined_raw: list[dict] = []
    combined_run_id_parts: list[str] = []
    for i, src in enumerate(sources):
        manifest_path = src["manifest_path"]
        source_name = src.get("source_name") or f"source_{i}"
        child = AnnotationToolBackingDataset(
            manifest_path=manifest_path,
            host_root=src.get("host_root"),
            container_root=src.get("container_root"),
        )
        combined_run_id_parts.append(f"{source_name}={child._run_id}")
        child_root = child._location_root
        for track in child._raw_tracks:
            new_obs = []
            for obs in track["observations"]:
                if "image_path" in obs:
                    obs = {**obs, "image_path": str(child_root / obs["image_path"])}
                new_obs.append(obs)
            new_track = {
                **track,
                "track_id": f"{source_name}::{track['track_id']}",
                "observations": new_obs,
            }
            if "space_id" in new_track:
                new_track["space_id"] = f"{source_name}::{new_track['space_id']}"
            combined_raw.append(new_track)

    combined_run_id = name or "multi[" + ",".join(combined_run_id_parts) + "]"
    return AnnotationToolBackingDataset._from_raw(
        run_id=combined_run_id,
        location_root=Path("/"),
        raw_tracks=combined_raw,
        max_anchor_observations=max_anchor_observations,
        shuffle_seed=shuffle_seed,
        per_track_balanced=per_track_balanced,
    )


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
