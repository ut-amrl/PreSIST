"""Shared param-mapping, debug-logging, and visualization helpers used by
``LLMPipelineInferencer`` and ``DirectVLMPipelineInferencer``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from persistence_eval.core.domain import ImageMaskGroup, PersistenceParameterEstimate


def map_to_persistence_estimates(
    raw_estimates: list[dict],
    track_ids: list[str],
    param_keys: Optional[list[str]] = None,
) -> list[PersistenceParameterEstimate]:
    """Align querier output to one estimate per input mask.

    VLM path: each dict carries an "id" key equal to the SegmentationResults
    id (0..N-1); we match by id and tolerate gaps caused by per-object
    failures. LLM path: dicts are in order but may also carry "id"; we fall
    back to positional alignment when no "id" is present.
    """
    n = len(track_ids)
    if raw_estimates and "id" in raw_estimates[0]:
        by_id: dict[int, dict] = {int(e["id"]): e for e in raw_estimates}
        aligned = [by_id.get(i, {}) for i in range(n)]
    else:
        aligned = list(raw_estimates) + [{}] * max(0, n - len(raw_estimates))

    return [
        PersistenceParameterEstimate(
            params=extract_params(raw, param_keys),
            metadata={"track_id": track_ids[i], "raw_estimate": raw},
        )
        for i, raw in enumerate(aligned[:n])
    ]


def extract_params(raw: dict, param_keys: Optional[list[str]] = None) -> dict[str, float]:
    extracted = {
        k: float(v)
        for k, v in raw.items()
        if k != "id" and isinstance(v, (int, float))
    }
    if param_keys is not None:
        extracted = {k: v for k, v in extracted.items() if k in param_keys}
    return extracted


def serialize_querier_debug(querier_debug) -> list[dict]:
    """Flatten a DebugPayload to JSON-serializable entries. Non-text/json
    kinds record only their name/kind to keep dumps readable.
    """
    out: list[dict] = []
    for rec in querier_debug:
        entry = {"name": rec.name, "kind": rec.kind}
        if rec.kind in ("text", "json"):
            entry["value"] = rec.value
        else:
            entry["value"] = f"<{rec.kind} omitted>"
        out.append(entry)
    return out


def write_debug_record(
    *,
    group: ImageMaskGroup,
    scene_caption: str,
    object_descriptions: dict,
    raw_estimates: list[dict],
    estimates: list[PersistenceParameterEstimate],
    querier_debug,
    debug_log: bool,
    debug_output_dir: Optional[Path],
) -> None:
    # Gate strictly on debug_log: this writes the per-anchor JSON record (and
    # optionally echoes it to stderr). The PNG mask-overlay viz is a separate
    # function with its own debug_visualizations gate.
    if not debug_log:
        return

    n = len(group.track_ids)
    if raw_estimates and "id" in raw_estimates[0]:
        by_id: dict[int, dict] = {int(e["id"]): e for e in raw_estimates}
        aligned_raw = [by_id.get(i, {}) for i in range(n)]
    else:
        aligned_raw = list(raw_estimates) + [{}] * max(0, n - len(raw_estimates))

    per_object: dict[str, dict] = {}
    for i, track_id in enumerate(group.track_ids):
        per_object[track_id] = {
            "segmentation_id": i,
            "description": object_descriptions.get(i, ""),
            "raw_estimate": aligned_raw[i] if i < len(aligned_raw) else {},
            "params": estimates[i].params if i < len(estimates) else {},
        }

    record = {
        "image_path": group.image_path,
        "anchor_time": group.anchor_time,
        "track_ids": list(group.track_ids),
        "scene_caption": scene_caption,
        "object_descriptions": {str(k): v for k, v in object_descriptions.items()},
        "raw_estimates": list(raw_estimates),
        "per_object": per_object,
    }

    if debug_log:
        print(json.dumps(record, indent=2, default=str), file=sys.stderr, flush=True)

    if debug_output_dir is not None:
        debug_output_dir.mkdir(parents=True, exist_ok=True)
        (debug_output_dir / f"debug_{group.anchor_time:.6f}.json").write_text(
            json.dumps(record, indent=2, default=str)
        )

    if debug_log and querier_debug is not None:
        qrec = serialize_querier_debug(querier_debug)
        print(
            json.dumps(
                {"anchor_time": group.anchor_time, "querier_debug": qrec},
                indent=2,
                default=str,
            ),
            file=sys.stderr,
            flush=True,
        )
        if debug_output_dir is not None:
            (debug_output_dir / f"debug_{group.anchor_time:.6f}_querier_payload.json").write_text(
                json.dumps(qrec, indent=2, default=str)
            )


def save_mask_visualization(group: ImageMaskGroup, debug_output_dir: Optional[Path]) -> None:
    """Write a matplotlib mask-overlay PNG. No-op when ``debug_output_dir`` is None.

    Gated by the caller via ``debug_visualizations`` — the inferencer only
    sets ``debug_output_dir`` when that flag is on, so passing None here
    means visualizations are disabled and we skip the rendering work
    entirely (previously this wrote to ``tempfile.mktemp`` and silently
    leaked PNGs on every anchor).
    """
    if debug_output_dir is None:
        return

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import binary_erosion

    image_np = group.image.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_np)

    cyan = (0.0, 1.0, 1.0)
    for mask, _track_id in zip(group.masks, group.track_ids):
        mask_np = mask.numpy()
        overlay = np.zeros((*mask_np.shape, 4), dtype=np.float32)
        overlay[mask_np] = [*cyan, 0.3]
        ax.imshow(overlay)
        outline = mask_np & ~binary_erosion(mask_np)
        outline_rgba = np.zeros((*mask_np.shape, 4), dtype=np.float32)
        outline_rgba[outline] = [*cyan, 0.8]
        ax.imshow(outline_rgba)

    track_str = "  |  ".join(group.track_ids)
    image_str = group.image_path or f"anchor_time={group.anchor_time:.3f}s"
    ax.set_title(f"{image_str}\ntracks: {track_str}", fontsize=9)
    ax.axis("off")

    debug_output_dir.mkdir(parents=True, exist_ok=True)
    out = debug_output_dir / f"vis_{group.anchor_time:.6f}.png"
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
