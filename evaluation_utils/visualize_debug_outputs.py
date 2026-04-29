"""Visualize cached debug outputs from a dataset-generation pipeline run.

Hydra-driven; reuses the same configs that drive ``run_llm_labeling_pipeline.py``
to locate cached results on disk and render per-object debug images.

Run as a module:

    python -m evaluation_utils.visualize_debug_outputs --config-name=<config> \
        ++visualize.read_root_dir=/workspace/output \
        ++visualize.write_root_dir=/workspace/output/debug_viz

Configurable knobs are nested under ``visualize.*`` (see ``_VizConfig``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Pipeline imports.
from dataset_generation.intermediate_results_logger.caching_results_logger2 import (
    CachingResultsLoggerRevised as _Logger,
)
from instance_segmentation.abs_instance_segmenter import SegmentationResults
from spf_io.persistence_io import PERSISTENCE_FILE_NAME, readPersistenceData
from utils.visualization_utils import overlay_single_mask


log = logging.getLogger("visualize_debug_outputs")

# Phases known to the pipeline.
PHASE_SCENE = "scene_caption"
PHASE_SEG = "segmentation"
PHASE_DESC = "object_descriptions"
PHASE_PERS = "persistence"
ALL_PHASES = (PHASE_SCENE, PHASE_SEG, PHASE_DESC, PHASE_PERS)
OBJECT_PATHWAY = (PHASE_SEG, PHASE_DESC, PHASE_PERS)  # increasing depth.

# Modes for handling outputs that already exist on disk.
EXISTING_FORCE = "force"               # always regenerate; always counts toward caps
EXISTING_SKIP_COUNT = "skip_count"     # skip regeneration; still counts toward caps
EXISTING_SKIP_EXTEND = "skip_extend"   # skip regeneration; does NOT count toward caps
EXISTING_MODES = (EXISTING_FORCE, EXISTING_SKIP_COUNT, EXISTING_SKIP_EXTEND)


# ---------------------------------------------------------------------------
# Per-image data container
# ---------------------------------------------------------------------------


@dataclass
class _ImageData:
    image_id: str
    seg: Optional[SegmentationResults] = None
    image_path: Optional[str] = None
    image_rgb: Optional[np.ndarray] = None  # (H, W, 3) uint8

    scene_caption: Optional[str] = None
    scene_caption_reasoning: Optional[str] = None

    descriptions: Dict[str, str] = field(default_factory=dict)
    description_reasoning: Dict[str, str] = field(default_factory=dict)
    description_query_images_by_id: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    # Per obj_id: list of (qIm_label, abs_path_to_image) sorted by qIm index.

    persistence_by_id: Dict[str, dict] = field(default_factory=dict)
    persistence_reasoning_by_id: Dict[str, str] = field(default_factory=dict)
    persistence_group_reasoning: Optional[str] = None  # shared across objects.


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_segmentation(read_root: str, segmenter_name: str, image_id: str) -> Optional[Tuple[SegmentationResults, Optional[str]]]:
    path = os.path.join(
        read_root, _Logger.SEGMENTER_RESULTS_DIR, segmenter_name, image_id, "segmentation_results.npz"
    )
    if not os.path.exists(path):
        return None
    try:
        return SegmentationResults.loadFromFile(path)
    except Exception as e:
        log.warning("failed to load segmentation for %s: %s", image_id, e)
        return None


def _resolve_image_path(image_path: Optional[str], read_root: str) -> Optional[str]:
    if not image_path:
        return None
    candidates = [image_path]
    if not os.path.isabs(image_path):
        candidates += [os.path.join(read_root, image_path), os.path.abspath(image_path)]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _load_image_rgb(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    try:
        from PIL import Image as _PIL
        return np.array(_PIL.open(path).convert("RGB"))
    except Exception as e:
        log.warning("failed to load image at %s: %s", path, e)
        return None


def _load_scene_caption(read_root: str, scene_captioner_name: str, image_id: str) -> Tuple[Optional[str], Optional[str]]:
    base = os.path.join(read_root, _Logger.SCENE_CAPTIONER_RESULTS_DIR, scene_captioner_name, image_id)
    cap_path = os.path.join(base, "scene_caption.json")
    caption: Optional[str] = None
    if os.path.exists(cap_path):
        try:
            with open(cap_path, "r", encoding="utf-8") as f:
                caption = json.load(f).get("caption", None)
        except Exception as e:
            log.warning("failed to read scene caption for %s: %s", image_id, e)

    reasoning = _read_text_debug_entry(os.path.join(base, "debug"), prefix="reasoning_trace")
    return caption, reasoning


def _load_object_descriptions(
    read_root: str, segmenter_name: str, describer_name: str, image_id: str
) -> Tuple[Dict[str, str], Dict[str, str]]:
    base = os.path.join(
        read_root, _Logger.OBJECT_DESCRIBER_RESULTS_DIR, segmenter_name, describer_name, image_id
    )
    descriptions: Dict[str, str] = {}
    desc_path = os.path.join(base, "object_descriptions.json")
    if os.path.exists(desc_path):
        try:
            with open(desc_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "descriptions" in payload and "imagePath" in payload:
                payload = payload["descriptions"]
            if isinstance(payload, dict):
                descriptions = {str(k): v for k, v in payload.items()}
        except Exception as e:
            log.warning("failed to read object descriptions for %s: %s", image_id, e)

    debug_dir = os.path.join(base, "debug")
    reasoning = _read_json_dict_entries(debug_dir, prefix="reasoning_traces")
    query_images = _read_query_images_by_obj(debug_dir)
    return descriptions, reasoning, query_images


def _read_query_images_by_obj(debug_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    """Group manifest image entries named ``query_img_<obj_id>_qIm<idx>`` by obj_id.

    Returns ``{obj_id: [(qIm_label, abs_path), ...]}`` with each list sorted by
    qIm index.
    """
    if not os.path.isdir(debug_dir):
        return {}
    out: Dict[str, List[Tuple[int, str, str]]] = {}
    for name, kind, path in _read_manifest_entries(debug_dir):
        if kind != "image" or not name.startswith("query_img_"):
            continue
        rest = name[len("query_img_"):]
        if "_qIm" not in rest:
            continue
        obj_id, q_part = rest.rsplit("_qIm", 1)
        try:
            q_idx = int(q_part.split("_", 1)[0])
        except ValueError:
            continue
        out.setdefault(obj_id, []).append((q_idx, f"qIm{q_idx:02d}", path))
    return {k: [(label, p) for _, label, p in sorted(v)] for k, v in out.items()}


def _load_persistence(read_root: str, config_file_name: str, image_id: str):
    base = os.path.join(read_root, _Logger.FULL_RESULTS_DIR, config_file_name, image_id)
    pers_path = os.path.join(base, PERSISTENCE_FILE_NAME)
    by_id: Dict[str, dict] = {}
    if os.path.exists(pers_path):
        try:
            entries = readPersistenceData(pers_path)
            for e in entries:
                if isinstance(e, dict) and "id" in e:
                    by_id[str(e["id"])] = e
        except Exception as ex:
            log.warning("failed to read persistence params for %s: %s", image_id, ex)

    debug_dir = os.path.join(base, "debug")
    per_obj_reasoning = _read_json_dict_entries(debug_dir, prefix="reasoning_traces")
    group_reasoning = None
    if not per_obj_reasoning:
        group_reasoning = _read_text_debug_entry(debug_dir, prefix="reasoning_trace")

    return by_id, per_obj_reasoning, group_reasoning


# ---------------------------------------------------------------------------
# Debug-folder helpers (manifest-aware via DebugDataReader, with fallback)
# ---------------------------------------------------------------------------


def _read_manifest_entries(debug_dir: str) -> List[Tuple[str, str, str]]:
    """Return list of (name, kind, abs_path). Empty if no manifest."""
    manifest_path = os.path.join(debug_dir, "_manifest.json")
    if not os.path.exists(manifest_path):
        return []
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return []
    out: List[Tuple[str, str, str]] = []
    for entry in manifest.get("entries", []):
        name = entry.get("name")
        kind = entry.get("kind")
        filename = entry.get("filename")
        if not (name and kind and filename):
            continue
        out.append((name, kind, os.path.join(debug_dir, filename)))
    return out


def _read_json_dict_entries(debug_dir: str, *, prefix: str) -> Dict[str, str]:
    """Merge all manifest entries whose name starts with ``prefix`` and whose
    serialized form is a JSON dict mapping str->str.
    """
    if not os.path.isdir(debug_dir):
        return {}
    merged: Dict[str, str] = {}
    for name, kind, path in _read_manifest_entries(debug_dir):
        if not name.startswith(prefix):
            continue
        if kind != "json":
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str):
                    merged[str(k)] = v
    return merged


def _read_text_debug_entry(debug_dir: str, *, prefix: str) -> Optional[str]:
    """Read the first manifest-listed text entry whose name starts with ``prefix``."""
    if not os.path.isdir(debug_dir):
        return None
    for name, kind, path in _read_manifest_entries(debug_dir):
        if not name.startswith(prefix):
            continue
        if kind != "text":
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Output-path resolution
# ---------------------------------------------------------------------------


def _deepest_object_phase(phases: set) -> Optional[str]:
    deepest: Optional[str] = None
    for p in OBJECT_PATHWAY:
        if p in phases:
            deepest = p
    return deepest


def _object_pathway_subpath(
    phases: set, segmenter_name: str, describer_name: str, config_file_name: str
) -> Optional[str]:
    deepest = _deepest_object_phase(phases)
    if deepest is None:
        return None
    if deepest == PHASE_PERS:
        return os.path.join(_Logger.FULL_RESULTS_DIR, config_file_name)
    if deepest == PHASE_DESC:
        return os.path.join(_Logger.OBJECT_DESCRIBER_RESULTS_DIR, segmenter_name, describer_name)
    return os.path.join(_Logger.SEGMENTER_RESULTS_DIR, segmenter_name)


def _scene_only_subpath(phases: set, scene_captioner_name: str) -> Optional[str]:
    if PHASE_SCENE in phases and _deepest_object_phase(phases) is None:
        return os.path.join(_Logger.SCENE_CAPTIONER_RESULTS_DIR, scene_captioner_name)
    return None


def _object_outputs_exist(
    base: str,
    image_id: str,
    obj_id: str,
    *,
    write_composite: bool,
    write_split_files: bool,
) -> bool:
    needed: List[str] = []
    if write_composite:
        needed.append(os.path.join(base, "composite", image_id, f"obj_{obj_id}.png"))
    if write_split_files:
        needed.append(os.path.join(base, "split", image_id, f"obj_{obj_id}", "info.txt"))
    return bool(needed) and all(os.path.exists(p) for p in needed)


def _scene_outputs_exist(
    base: str,
    image_id: str,
    *,
    write_composite: bool,
    write_split_files: bool,
) -> bool:
    needed: List[str] = []
    if write_composite:
        needed.append(os.path.join(base, "composite", f"{image_id}.png"))
    if write_split_files:
        needed.append(os.path.join(base, "split", image_id, "info.txt"))
    return bool(needed) and all(os.path.exists(p) for p in needed)


def _enumerate_image_ids(read_root: str, phases: set, segmenter_name: str, describer_name: str,
                         scene_captioner_name: str, config_file_name: str) -> List[str]:
    """List image_ids by inspecting the deepest enabled phase's directory."""
    if PHASE_PERS in phases:
        d = os.path.join(read_root, _Logger.FULL_RESULTS_DIR, config_file_name)
    elif PHASE_DESC in phases:
        d = os.path.join(read_root, _Logger.OBJECT_DESCRIBER_RESULTS_DIR, segmenter_name, describer_name)
    elif PHASE_SEG in phases:
        d = os.path.join(read_root, _Logger.SEGMENTER_RESULTS_DIR, segmenter_name)
    elif PHASE_SCENE in phases:
        d = os.path.join(read_root, _Logger.SCENE_CAPTIONER_RESULTS_DIR, scene_captioner_name)
    else:
        return []
    if not os.path.isdir(d):
        log.warning("source directory does not exist: %s", d)
        return []
    return sorted(c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c)))


# ---------------------------------------------------------------------------
# Mask border
# ---------------------------------------------------------------------------


def _draw_mask_border(
    image_rgb: np.ndarray,
    mask_t: "torch.Tensor",
    *,
    color_rgb: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 3,
) -> np.ndarray:
    """Draw an opaque outline of ``mask_t`` onto ``image_rgb`` (in-place copy)."""
    import cv2

    m = mask_t.detach().cpu().numpy()
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    if m.dtype != bool:
        m = m > 0.5
    if not m.any():
        return image_rgb
    contours, _ = cv2.findContours(
        m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    out = image_rgb.copy()
    cv2.drawContours(out, contours, -1, color_rgb, thickness=thickness)
    return out


# ---------------------------------------------------------------------------
# Composite figure rendering
# ---------------------------------------------------------------------------


_WRAP_COL = 80
_LINE_HEIGHT_IN = 0.18
_IMAGE_PANEL_W_IN = 8.0
_MIN_TEXT_W_IN = 6.0
_MAX_TEXT_W_IN = 24.0
_TEXT_FONT_SIZE = 9


def _wrap_block(title: str, body: str) -> List[str]:
    if not body:
        return []
    body = body.rstrip()
    out = [title, "-" * len(title)]
    for paragraph in body.split("\n"):
        if not paragraph:
            out.append("")
            continue
        out.extend(textwrap.wrap(paragraph, width=_WRAP_COL) or [""])
    out.append("")
    return out


def _build_text_lines(
    *,
    include_caption: bool,
    img_data: _ImageData,
    obj_id: str,
) -> List[str]:
    lines: List[str] = []
    if img_data.image_path:
        lines += _wrap_block("Source Image", img_data.image_path)
    if include_caption and img_data.scene_caption:
        lines += _wrap_block("Scene Caption", img_data.scene_caption)
        if img_data.scene_caption_reasoning:
            lines += _wrap_block("Scene Caption Reasoning", img_data.scene_caption_reasoning)

    desc_reason = img_data.description_reasoning.get(obj_id)
    if desc_reason:
        lines += _wrap_block("Description Reasoning", desc_reason)

    desc = img_data.descriptions.get(obj_id)
    if desc:
        lines += _wrap_block("Description", desc)

    pers_reason = img_data.persistence_reasoning_by_id.get(obj_id)
    if pers_reason is None and img_data.persistence_group_reasoning:
        pers_reason = "(group-level reasoning)\n\n" + img_data.persistence_group_reasoning
    if pers_reason:
        lines += _wrap_block("Persistence Reasoning", pers_reason)

    pers = img_data.persistence_by_id.get(obj_id)
    if pers is not None:
        pretty = json.dumps(pers, indent=2, ensure_ascii=False)
        lines += _wrap_block("Persistence Estimate", pretty)

    return lines


def _build_scene_only_text_lines(img_data: _ImageData) -> List[str]:
    lines: List[str] = []
    if img_data.image_path:
        lines += _wrap_block("Source Image", img_data.image_path)
    if img_data.scene_caption:
        lines += _wrap_block("Scefne Caption", img_data.scene_caption)
    if img_data.scene_caption_reasoning:
        lines += _wrap_block("Scene Caption Reasoning", img_data.scene_caption_reasoning)
    return lines


def _render_composite(
    image_rgb: np.ndarray,
    overlay_rgb: Optional[np.ndarray],
    text_lines: List[str],
    out_path: str,
    query_images: Optional[List[Tuple[str, np.ndarray]]] = None,
) -> None:
    """Render a side-by-side composite. ``overlay_rgb`` is the left panel; if
    None, ``image_rgb`` is used. If ``query_images`` (list of (label, rgb))
    is provided, an additional row of query-image tiles is appended below."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    left = overlay_rgb if overlay_rgb is not None else image_rgb
    H, W = left.shape[:2]
    image_panel_h_in = _IMAGE_PANEL_W_IN * (H / W)

    n_lines = max(1, len(text_lines))
    target_lines_per_col = max(1, int(image_panel_h_in / _LINE_HEIGHT_IN))
    n_cols = max(1, int(np.ceil(n_lines / target_lines_per_col)))
    text_w_in = min(_MAX_TEXT_W_IN, max(_MIN_TEXT_W_IN, n_cols * _MIN_TEXT_W_IN))

    truncated = False
    max_renderable_lines = int((text_w_in / _MIN_TEXT_W_IN) * target_lines_per_col)
    if n_lines > max_renderable_lines:
        keep = max(1, max_renderable_lines - 2)
        text_lines = text_lines[:keep] + ["", "(truncated; see split-files for full text)"]
        truncated = True

    fig_w = _IMAGE_PANEL_W_IN + text_w_in
    fig_h_top = max(image_panel_h_in, 4.0)

    has_queries = bool(query_images)
    if has_queries:
        n_q = len(query_images)
        tile_w_in = max(2.5, fig_w / max(n_q, 1))
        tile_w_in = min(tile_w_in, fig_w / n_q)
        # Per-tile height by aspect of first image; fall back to square.
        first_q_rgb = query_images[0][1]
        qH, qW = first_q_rgb.shape[:2]
        tile_h_in = tile_w_in * (qH / qW) if qW else tile_w_in
        tile_h_in = max(2.0, min(tile_h_in, image_panel_h_in))
        fig_h = fig_h_top + tile_h_in + 0.4
        fig = plt.figure(figsize=(fig_w, fig_h))
        outer = fig.add_gridspec(2, 1, height_ratios=[fig_h_top, tile_h_in + 0.4])
        top_gs = outer[0].subgridspec(1, 2, width_ratios=[_IMAGE_PANEL_W_IN, text_w_in])
        bot_gs = outer[1].subgridspec(1, n_q)
    else:
        fig = plt.figure(figsize=(fig_w, fig_h_top))
        top_gs = fig.add_gridspec(1, 2, width_ratios=[_IMAGE_PANEL_W_IN, text_w_in])
        bot_gs = None

    ax_img = fig.add_subplot(top_gs[0, 0])
    ax_img.imshow(left)
    ax_img.axis("off")

    ax_txt = fig.add_subplot(top_gs[0, 1])
    ax_txt.axis("off")
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    ax_txt.text(
        0.0,
        1.0,
        "\n".join(text_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=_TEXT_FONT_SIZE,
    )

    if bot_gs is not None:
        for i, (label, rgb) in enumerate(query_images):
            ax_q = fig.add_subplot(bot_gs[0, i])
            ax_q.imshow(rgb)
            ax_q.set_title(label, fontsize=_TEXT_FONT_SIZE)
            ax_q.axis("off")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    if truncated:
        log.debug("truncated composite text for %s", out_path)


# ---------------------------------------------------------------------------
# Split-files writer
# ---------------------------------------------------------------------------


def _write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _write_image(path: str, rgb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from PIL import Image as _PIL

    _PIL.fromarray(rgb).save(path)


_SECTION_SEP = "\n" + ("=" * 72) + "\n"


def _format_section(title: str, body: str) -> str:
    return f"# {title}\n\n{body.rstrip()}\n"


def _build_combined_text(
    *,
    img_data: _ImageData,
    obj_id: Optional[str],
    include_caption: bool,
) -> str:
    """Build a single text blob containing every text-based section that is
    present, separated by section headers. ``obj_id=None`` produces a
    scene-only blob (no per-object sections)."""
    sections: List[str] = []

    if img_data.image_path:
        sections.append(_format_section("Source Image", img_data.image_path))

    if include_caption and img_data.scene_caption:
        sections.append(_format_section("Scene Caption", img_data.scene_caption))
    if include_caption and img_data.scene_caption_reasoning:
        sections.append(_format_section("Scene Caption Reasoning", img_data.scene_caption_reasoning))

    if obj_id is not None:
        desc_reason = img_data.description_reasoning.get(obj_id)
        if desc_reason:
            sections.append(_format_section("Description Reasoning", desc_reason))

        desc = img_data.descriptions.get(obj_id)
        if desc:
            sections.append(_format_section("Description", desc))

        pers_reason = img_data.persistence_reasoning_by_id.get(obj_id)
        if pers_reason is None and img_data.persistence_group_reasoning:
            pers_reason = "(group-level reasoning)\n\n" + img_data.persistence_group_reasoning
        if pers_reason:
            sections.append(_format_section("Persistence Reasoning", pers_reason))

        pers = img_data.persistence_by_id.get(obj_id)
        if pers is not None:
            pretty = json.dumps(pers, indent=2, ensure_ascii=False)
            sections.append(_format_section("Persistence Estimate", pretty))

    return _SECTION_SEP.join(sections)


def _write_split_object(
    out_dir: str,
    overlay_rgb: np.ndarray,
    img_data: _ImageData,
    obj_id: str,
    *,
    include_caption: bool,
) -> None:
    _write_image(os.path.join(out_dir, "mask_overlay.png"), overlay_rgb)
    blob = _build_combined_text(img_data=img_data, obj_id=obj_id, include_caption=include_caption)
    if blob:
        _write_text(os.path.join(out_dir, "info.txt"), blob)
    for label, rgb in _load_query_image_arrays(img_data, obj_id):
        _write_image(os.path.join(out_dir, f"query_{label}.png"), rgb)


def _load_query_image_arrays(
    img_data: _ImageData, obj_id: str
) -> List[Tuple[str, np.ndarray]]:
    entries = img_data.description_query_images_by_id.get(obj_id) or []
    if not entries:
        return []
    out: List[Tuple[str, np.ndarray]] = []
    from PIL import Image as _PIL
    for label, path in entries:
        try:
            arr = np.array(_PIL.open(path).convert("RGB"))
        except Exception as e:
            log.warning("failed to load query image %s: %s", path, e)
            continue
        out.append((label, arr))
    return out


def _write_split_image_scene(image_dir: str, img_data: _ImageData, *, scene_only_mode: bool) -> None:
    """Write image-level scene content (used only in scene-only mode now;
    when objects are present, scene caption is folded into each per-object
    info.txt instead)."""
    if not scene_only_mode:
        return
    blob = _build_combined_text(img_data=img_data, obj_id=None, include_caption=True)
    if blob:
        _write_text(os.path.join(image_dir, "info.txt"), blob)
    if img_data.image_rgb is not None:
        _write_image(os.path.join(image_dir, "scene_image.png"), img_data.image_rgb)


# ---------------------------------------------------------------------------
# Component-name resolution
# ---------------------------------------------------------------------------


def _get_optional(cfg: DictConfig, dotted: str) -> Optional[str]:
    try:
        val = OmegaConf.select(cfg, dotted)
    except Exception:
        return None
    if val is None:
        return None
    return str(val)


def _derive_segmenter_name(seg_cfg) -> Optional[str]:
    """Reproduce the runtime ``name`` of a segmenter from its config tree
    without instantiating it (which would load model weights).

    Handles the cases used by the dataset-gen configs today: an explicit
    ``name`` in the YAML, or a wrapper that composes its name from inner
    components. Returns None if the pattern isn't recognized.
    """
    if seg_cfg is None:
        return None

    explicit = _get_optional(seg_cfg, "name")
    if explicit:
        return explicit

    target = _get_optional(seg_cfg, "_target_") or ""

    if target.endswith("LanguageGroundedSegmenter"):
        disc = _derive_segmenter_name(seg_cfg.get("object_discoverer"))
        lang = _derive_segmenter_name(seg_cfg.get("language_segmenter"))
        if not (disc and lang):
            return None
        suffix = _get_optional(seg_cfg, "name_suffix") or ""
        if suffix and not suffix.startswith("_"):
            suffix = "_" + suffix
        if suffix:
            suffix = "_" + suffix
        return f"langGround_p_{disc}_s_{lang}{suffix}"

    if target.endswith("LanguageGroundedWithPointRepromptSegmenter"):
        inner = _derive_segmenter_name(seg_cfg.get("language_grounded_segmenter"))
        point = _derive_segmenter_name(seg_cfg.get("point_prompted_segmenter"))
        if not (inner and point):
            return None
        suffix = _get_optional(seg_cfg, "name_suffix") or ""
        return f"{inner}_reprompt_{point}{suffix}"

    if target.endswith("ConfidenceThresholdSegmentationWrapper"):
        base = _derive_segmenter_name(seg_cfg.get("base_segmenter"))
        thresh = _get_optional(seg_cfg, "confidence_threshold")
        if not base or thresh is None:
            return None
        return f"{base}_conf_thresh_{thresh}"

    return None


def _disk_segmenter_fallback(read_root: Optional[str]) -> Optional[str]:
    """If exactly one segmenter directory exists on disk, return its name."""
    if not read_root:
        return None
    d = os.path.join(read_root, _Logger.SEGMENTER_RESULTS_DIR)
    if not os.path.isdir(d):
        return None
    candidates = [c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c))]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        log.warning(
            "could not derive segmenter_name from config; multiple candidates on disk under %s: %s",
            d, candidates,
        )
    return None


def _resolve_names(cfg: DictConfig, *, read_root: Optional[str]) -> Dict[str, Optional[str]]:
    """Pull component names from the Hydra config or visualize.* overrides.

    For composed segmenter names (e.g. language-grounded), reproduce the
    runtime composition by walking the OmegaConf tree. As a last resort,
    pick the unique segmenter directory on disk.
    """
    viz = cfg.get("visualize", {}) or {}

    seg_name = viz.get("segmenter_name") or _derive_segmenter_name(
        OmegaConf.select(cfg, "dataset_generator.instanceSegmenter")
    )
    if not seg_name:
        seg_name = _disk_segmenter_fallback(read_root)

    return {
        "segmenter": seg_name,
        "scene_captioner": viz.get("scene_captioner_name") or _get_optional(cfg, "dataset_generator.sceneCaptioner.name"),
        "object_describer": viz.get("object_describer_name") or _get_optional(cfg, "dataset_generator.objectDescriber.name"),
    }


def _resolve_config_file_name(cfg: DictConfig) -> str:
    viz = cfg.get("visualize", {}) or {}
    if viz.get("config_file_name"):
        return str(viz["config_file_name"])
    try:
        from hydra.core.hydra_config import HydraConfig

        hc = HydraConfig.get()
        return str(hc.job.config_name)
    except Exception:
        # Fallback: caller can override.
        raise RuntimeError(
            "Could not determine config_file_name. Pass ++visualize.config_file_name=<name>."
        )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _normalize_phases(raw) -> set:
    if raw is None:
        return set(ALL_PHASES)
    if isinstance(raw, str):
        items = [raw]
    else:
        items = list(raw)
    out = set()
    for p in items:
        if p not in ALL_PHASES:
            raise ValueError(f"Unknown phase {p!r}; must be one of {ALL_PHASES}")
        out.add(p)
    return out


@hydra.main(version_base=None, config_path="../configs/dataset_gen_confs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    viz = cfg.get("visualize", {}) or {}
    read_root = (
        viz.get("read_root_dir")
        or cfg.get("output_root_dir")
        or _get_optional(cfg, "dataset_generator.intermediateLogger.baseLogDirectory")
    )
    if not read_root:
        log.error(
            "read_root_dir not configured (set ++visualize.read_root_dir, ++output_root_dir, "
            "or dataset_generator.intermediateLogger.baseLogDirectory)"
        )
        sys.exit(2)
    read_root = os.path.abspath(read_root)

    write_root = viz.get("write_root_dir")
    if not write_root:
        write_root = os.path.join(read_root, "debug_viz")
    write_root = os.path.abspath(write_root)

    phases = _normalize_phases(viz.get("phases"))
    deepest_obj = _deepest_object_phase(phases)
    if deepest_obj in (PHASE_DESC, PHASE_PERS) and PHASE_SEG not in phases:
        # User asked for a downstream phase but didn't include segmentation.
        # We need segmentation to render per-object panels, so add it implicitly.
        phases.add(PHASE_SEG)

    max_objects_global = viz.get("max_objects_global", 400)
    if max_objects_global is None:
        max_objects_global = float("inf")
    max_objects_per_image = viz.get("max_objects_per_image")
    if max_objects_per_image is None:
        max_objects_per_image = float("inf")
    max_images = viz.get("max_images", 40)
    if max_images is None:
        max_images = float("inf")

    write_composite = bool(viz.get("write_composite", True))
    write_split_files = bool(viz.get("write_split_files", False))
    if not (write_composite or write_split_files):
        log.warning("both write_composite and write_split_files are false; nothing to do")
        return

    existing_mode = str(viz.get("existing_outputs_mode", EXISTING_SKIP_COUNT))
    if existing_mode not in EXISTING_MODES:
        log.error("existing_outputs_mode=%r must be one of %s", existing_mode, EXISTING_MODES)
        sys.exit(2)

    names = _resolve_names(cfg, read_root=read_root)
    segmenter_name = names["segmenter"]
    scene_captioner_name = names["scene_captioner"]
    describer_name = names["object_describer"]

    config_file_name = None
    if PHASE_PERS in phases:
        config_file_name = _resolve_config_file_name(cfg)

    # Validate names we'll need.
    if PHASE_SEG in phases and not segmenter_name:
        log.error(
            "segmenter name unresolved (config has no dataset_generator.instanceSegmenter.name; "
            "pass ++visualize.segmenter_name=<name>)"
        )
        sys.exit(2)
    if PHASE_DESC in phases and not describer_name:
        log.error(
            "object_describer name unresolved (pass ++visualize.object_describer_name=<name>)"
        )
        sys.exit(2)
    if PHASE_SCENE in phases and not scene_captioner_name:
        log.error(
            "scene_captioner name unresolved (pass ++visualize.scene_captioner_name=<name>)"
        )
        sys.exit(2)

    # Load the caption whenever scene_caption or persistence is in phases. If
    # the caption name is unknown but only persistence asked for it, just skip
    # the caption section silently.
    load_caption = (PHASE_SCENE in phases) or (PHASE_PERS in phases and bool(scene_captioner_name))

    object_subpath = _object_pathway_subpath(phases, segmenter_name or "", describer_name or "", config_file_name or "")
    scene_only_subpath = _scene_only_subpath(phases, scene_captioner_name or "")

    image_ids = _enumerate_image_ids(
        read_root, phases, segmenter_name or "", describer_name or "", scene_captioner_name or "", config_file_name or ""
    )
    if viz.get("image_ids"):
        whitelist = set(str(i) for i in viz["image_ids"])
        image_ids = [i for i in image_ids if i in whitelist]

    log.info("found %d candidate image_ids; phases=%s", len(image_ids), sorted(phases))
    log.info("read_root=%s", read_root)
    log.info("write_root=%s", write_root)
    if object_subpath:
        log.info("object output base: %s", os.path.join(write_root, object_subpath))
    if scene_only_subpath:
        log.info("scene-only output base: %s", os.path.join(write_root, scene_only_subpath))

    images_emitted = 0
    objects_emitted = 0
    emitted_image_ids: List[str] = []
    include_caption_in_panel = (PHASE_SCENE in phases) or (PHASE_PERS in phases)

    for image_id in image_ids:
        if images_emitted >= max_images or objects_emitted >= max_objects_global:
            break

        log.info("processing image_id=%s", image_id)
        img_data = _ImageData(image_id=image_id)

        # Segmentation (required by object pathway).
        if PHASE_SEG in phases:
            seg_loaded = _load_segmentation(read_root, segmenter_name, image_id)
            if seg_loaded is None:
                log.warning("segmentation missing for image_id=%s; skipping image", image_id)
                continue
            seg, image_path = seg_loaded
            img_data.seg = seg
            img_data.image_path = _resolve_image_path(image_path, read_root)
            img_data.image_rgb = _load_image_rgb(img_data.image_path)
            if img_data.image_rgb is None:
                log.warning("source image not loadable for image_id=%s; skipping image", image_id)
                continue

        if load_caption:
            cap, cap_reason = _load_scene_caption(read_root, scene_captioner_name, image_id)
            img_data.scene_caption = cap
            img_data.scene_caption_reasoning = cap_reason
            # If scene-only and source image not yet loaded, try to find it from
            # any available phase's segmentation file (best-effort).
            if img_data.image_rgb is None and segmenter_name:
                seg_loaded = _load_segmentation(read_root, segmenter_name, image_id)
                if seg_loaded is not None:
                    _, image_path = seg_loaded
                    img_data.image_path = _resolve_image_path(image_path, read_root)
                    img_data.image_rgb = _load_image_rgb(img_data.image_path)

        if PHASE_DESC in phases:
            descs, desc_reasoning, desc_query_images = _load_object_descriptions(
                read_root, segmenter_name, describer_name, image_id
            )
            img_data.descriptions = descs
            img_data.description_reasoning = desc_reasoning
            img_data.description_query_images_by_id = desc_query_images

        if PHASE_PERS in phases:
            by_id, per_obj_reason, group_reason = _load_persistence(read_root, config_file_name, image_id)
            img_data.persistence_by_id = by_id
            img_data.persistence_reasoning_by_id = per_obj_reason
            img_data.persistence_group_reasoning = group_reason

        emitted_for_this_image = 0

        # ----- scene-only mode: one composite per image, no per-object loop -----
        if scene_only_subpath is not None:
            scene_lines = _build_scene_only_text_lines(img_data)
            if not scene_lines:
                continue
            base = os.path.join(write_root, scene_only_subpath)
            scene_exists = existing_mode != EXISTING_FORCE and _scene_outputs_exist(
                base, image_id,
                write_composite=write_composite,
                write_split_files=write_split_files,
            )
            if scene_exists and existing_mode == EXISTING_SKIP_EXTEND:
                pass  # skip rendering and don't count toward caps
            elif scene_exists and existing_mode == EXISTING_SKIP_COUNT:
                emitted_for_this_image += 1
            else:
                if write_composite:
                    if img_data.image_rgb is None:
                        log.info("scene_caption-only mode but no image available for %s", image_id)
                    else:
                        out_path = os.path.join(base, "composite", f"{image_id}.png")
                        _render_composite(img_data.image_rgb, None, scene_lines, out_path)
                if write_split_files:
                    split_dir = os.path.join(base, "split", image_id)
                    _write_split_image_scene(split_dir, img_data, scene_only_mode=True)
                emitted_for_this_image += 1

        # ----- object-pathway mode: one composite per object -----
        if object_subpath is not None and img_data.seg is not None and img_data.image_rgb is not None:
            base = os.path.join(write_root, object_subpath)

            # Scene caption is folded into each per-object info.txt now —
            # no separate per-image scene file in object-pathway split mode.

            ids = img_data.seg.ids
            masks = img_data.seg.masks
            n_ids = len(ids)
            if max_objects_per_image < n_ids:
                k = int(max_objects_per_image)
                stride = n_ids / k
                selected_indices: List[int] = []
                seen_idx: set = set()
                for j in range(k):
                    idx = min(n_ids - 1, int(round(j * stride)))
                    if idx not in seen_idx:
                        seen_idx.add(idx)
                        selected_indices.append(idx)
            else:
                selected_indices = list(range(n_ids))

            for i in selected_indices:
                if objects_emitted >= max_objects_global:
                    break

                obj_id = str(int(ids[i].item()))

                obj_exists = existing_mode != EXISTING_FORCE and _object_outputs_exist(
                    base, image_id, obj_id,
                    write_composite=write_composite,
                    write_split_files=write_split_files,
                )
                if obj_exists and existing_mode == EXISTING_SKIP_EXTEND:
                    continue  # skip without counting toward caps
                if obj_exists and existing_mode == EXISTING_SKIP_COUNT:
                    emitted_for_this_image += 1
                    objects_emitted += 1
                    continue

                # Object-level skip rules: if this id is missing at a level
                # that has *some* entries, log a warning (one per missing
                # level per object).
                if PHASE_DESC in phases and img_data.descriptions and obj_id not in img_data.descriptions:
                    log.warning("description missing for image_id=%s obj_id=%s", image_id, obj_id)
                if PHASE_PERS in phases and img_data.persistence_by_id and obj_id not in img_data.persistence_by_id:
                    log.warning("persistence entry missing for image_id=%s obj_id=%s", image_id, obj_id)

                overlay_rgb = overlay_single_mask(
                    torch.from_numpy(img_data.image_rgb).permute(2, 0, 1),
                    masks[i],
                    alpha_mask=0.55,
                    alpha_img=0.45,
                    color_bgr=(0, 255, 255),
                )
                overlay_rgb = _draw_mask_border(
                    overlay_rgb, masks[i], color_rgb=(0, 255, 255), thickness=3
                )

                if write_composite:
                    text_lines = _build_text_lines(
                        include_caption=include_caption_in_panel,
                        img_data=img_data,
                        obj_id=obj_id,
                    )
                    out_path = os.path.join(base, "composite", image_id, f"obj_{obj_id}.png")
                    query_imgs = _load_query_image_arrays(img_data, obj_id)
                    _render_composite(
                        img_data.image_rgb,
                        overlay_rgb,
                        text_lines,
                        out_path,
                        query_images=query_imgs or None,
                    )

                if write_split_files:
                    split_dir = os.path.join(base, "split", image_id, f"obj_{obj_id}")
                    _write_split_object(
                        split_dir,
                        overlay_rgb,
                        img_data,
                        obj_id,
                        include_caption=include_caption_in_panel,
                    )

                emitted_for_this_image += 1
                objects_emitted += 1

        if emitted_for_this_image > 0:
            images_emitted += 1
            emitted_image_ids.append(image_id)

    log.info(
        "done: %d image(s) touched, %d object panel(s) emitted, output under %s",
        images_emitted,
        objects_emitted,
        write_root,
    )
    print("emitted image_ids:")
    for iid in emitted_image_ids:
        print(iid)


if __name__ == "__main__":
    main()
