"""Location-key extraction for annotation-tool tracks.

A "location key" is the bucket identifier used by per-location persistence
trainers and inferencers. Persistent tracks carry a stable `space_id`; transient
tracks (no `space_id` or empty) fall back to a discretized image-space cell
derived from the first positive observation's bounding box centre.
"""

from __future__ import annotations

from typing import Iterable


_POSITIVE = "Positive"


def _bbox_center(bbox: list[float]) -> tuple[float, float] | None:
    if bbox is None or len(bbox) != 4:
        return None
    x_min, y_min, x_max, y_max = bbox
    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)


def _first_positive_bbox(observations: Iterable[dict]) -> list[float] | None:
    for obs in observations:
        if obs.get("state") == _POSITIVE and obs.get("bbox") is not None:
            return obs["bbox"]
    return None


def location_key_for_track(track: dict, cell_size_px: int) -> str:
    """Return the location key for a raw annotation-tool track dict.

    Persistent tracks: `space_id` is used as-is.
    Transient tracks: cell key built from the first positive observation's
    bbox centre, discretized into `cell_size_px`-sized cells.
    """
    space_id = track.get("space_id")
    if space_id:
        return str(space_id)

    bbox = _first_positive_bbox(track.get("observations", []))
    center = _bbox_center(bbox) if bbox is not None else None
    if center is None:
        return f"transient_unlocated_{track['track_id']}"
    cx, cy = center
    cell_x = int(cx // cell_size_px)
    cell_y = int(cy // cell_size_px)
    return f"cell_{cell_x}_{cell_y}_sz{cell_size_px}"
