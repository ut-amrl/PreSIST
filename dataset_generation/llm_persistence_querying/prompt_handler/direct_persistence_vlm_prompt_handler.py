"""
Per-object VLM prompt handler that asks for the persistence parameters
*directly* from the image — no scene caption, no object description.

The prompt text is built by concatenating an optional ``prefix_file`` (which
typically describes the image layout the VLM is about to see) and a
``prompt_file`` (the task instructions). Both are loaded once at construction.

The per-object image list is assembled by running a sequence of named
formatters (e.g. ``"full_scene"``, ``"blur_bg_outline_fg"``,
``"zoom_crop_blur_bg_outline_fg"``) so a config can specify the exact image
order without subclassing.

Response parsing reuses the same robust JSON extractor as
``VerboseVLMPersistencePromptHandler`` (handles ``</think>`` traces, JSON
inside prose, singleton arrays, etc.) and then validates the IQR+behav4
schema. The integrating querier assigns ``id``; this handler must not.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from instance_segmentation.abs_instance_segmenter import SegmentationResults
from dataset_generation.llm_persistence_querying.prompt_handler.abs_vlm_prompt_handler import (
    AbsVLMPromptHandler,
)
from utils.image_utils import image_tensor_to_pil
from utils.visualization_utils import mask_highlight_image


_REQUIRED_KEYS = (
    "p25_persistence_hours",
    "p75_persistence_hours",
    "persistence_uncertainty",
    "persistence_conf",
)


# --------------------------------------------------------------------------
# Image formatters
# --------------------------------------------------------------------------
#
# Each formatter takes (image_tensor (3,H,W), mask (H,W) bool, settings) and
# returns one PIL.Image. New formatters can be registered by adding to
# ``_FORMATTERS`` below; callers pick formatters by name via the
# ``image_formatters`` list in the constructor.


def _to_pil_bgr(arr: np.ndarray) -> Image.Image:
    """Wrap a (H, W, 3) uint8 numpy array (BGR or RGB) as a PIL Image.

    ``mask_highlight_image`` produces BGR for the cv2-based styles
    (``mask_overlay``, ``foreground_only``, ``binary_mask``) and RGB for
    ``blur_bg_outline_fg``. We always go through ``Image.fromarray`` here
    and let the caller pick the right style — channel order is consistent
    with the existing ``VerboseVLMPersistencePromptHandler``'s convention
    of treating the output as RGB-displayable.
    """
    return Image.fromarray(arr)


def _zoom_crop_box(
    mask: torch.Tensor, img_w: int, img_h: int, scale_min: float = 0.0
) -> Optional[Tuple[int, int, int, int]]:
    m = mask.detach().cpu()
    if m.dim() == 3 and m.shape[0] == 1:
        m = m[0]
    m_np = m.numpy().astype(bool)
    rows = np.any(m_np, axis=1)
    cols = np.any(m_np, axis=0)
    if not rows.any():
        return None
    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    mh = rmax - rmin + 1
    mw = cmax - cmin + 1
    cy = (rmin + rmax) / 2.0
    cx = (cmin + cmax) / 2.0
    scale = min(max(3 * mw / img_w, 3 * mh / img_h, scale_min), 1.0)
    crop_w = int(np.ceil(scale * img_w))
    crop_h = int(np.ceil(scale * img_h))
    half_w = min(crop_w / 2.0, cx, img_w - cx)
    half_h = min(crop_h / 2.0, cy, img_h - cy)
    return (
        int(round(cx - half_w)),
        int(round(cy - half_h)),
        int(round(cx + half_w)),
        int(round(cy + half_h)),
    )


def _format_full_scene(image_t: torch.Tensor, mask: torch.Tensor, _settings) -> Image.Image:
    return image_tensor_to_pil(image_t)


def _format_blur_bg_outline_fg(
    image_t: torch.Tensor, mask: torch.Tensor, settings: dict
) -> Image.Image:
    arr = mask_highlight_image(
        image_t,
        mask,
        style="blur_bg_outline_fg",
        mask_highlight_color=settings["mask_highlight_color"],
        extra_kwargs={
            "outline_thickness": settings["full_mask_outline_thickness"],
            "background_dim_factor": settings["background_dim_factor"],
            "blur_radius": settings["blur_radius"],
            "highlight_alpha": settings["highlight_alpha"],
        },
    )
    return _to_pil_bgr(arr)


def _format_mask_overlay(
    image_t: torch.Tensor, mask: torch.Tensor, settings: dict
) -> Image.Image:
    arr = mask_highlight_image(
        image_t,
        mask,
        style="mask_overlay",
        mask_highlight_color=settings["mask_highlight_color"],
    )
    return _to_pil_bgr(arr)


def _format_foreground_only(
    image_t: torch.Tensor, mask: torch.Tensor, settings: dict
) -> Image.Image:
    arr = mask_highlight_image(
        image_t, mask, style="foreground_only",
        mask_highlight_color=settings["mask_highlight_color"],
    )
    return _to_pil_bgr(arr)


def _format_zoom_crop_blur_bg_outline_fg(
    image_t: torch.Tensor, mask: torch.Tensor, settings: dict
) -> Optional[Image.Image]:
    H, W = image_t.shape[-2:]
    box = _zoom_crop_box(mask, W, H)
    if box is None:
        return None
    arr = mask_highlight_image(
        image_t,
        mask,
        style="blur_bg_outline_fg",
        mask_highlight_color=settings["mask_highlight_color"],
        extra_kwargs={
            "outline_thickness": settings["crop_mask_outline_thickness"],
            "background_dim_factor": settings["background_dim_factor"],
            "blur_radius": settings["blur_radius"],
            "highlight_alpha": settings["highlight_alpha"],
        },
    )
    return _to_pil_bgr(arr).crop(box)


_FORMATTERS: Dict[str, Callable[[torch.Tensor, torch.Tensor, dict], Optional[Image.Image]]] = {
    "full_scene": _format_full_scene,
    "blur_bg_outline_fg": _format_blur_bg_outline_fg,
    "mask_overlay": _format_mask_overlay,
    "foreground_only": _format_foreground_only,
    "zoom_crop_blur_bg_outline_fg": _format_zoom_crop_blur_bg_outline_fg,
}


def available_image_formatters() -> List[str]:
    return list(_FORMATTERS.keys())


# --------------------------------------------------------------------------
# Prompt handler
# --------------------------------------------------------------------------


class DirectPersistenceVLMPromptHandler(AbsVLMPromptHandler):
    """Build per-object prompts that go directly from image to persistence params."""

    def __init__(
        self,
        name: str,
        prompt_file: str,
        prefix_file: Optional[str] = None,
        image_formatters: Optional[List[str]] = None,
        mask_highlight_color: Tuple[int, int, int] = (0, 255, 255),
        full_mask_outline_thickness: int = 1,
        crop_mask_outline_thickness: int = 3,
        background_dim_factor: float = 0.35,
        blur_radius: int = 8,
        highlight_alpha: float = 0.0,
    ):
        super().__init__(name=name)

        with open(prompt_file, "r", encoding="utf-8") as f:
            self._prompt_body = f.read().strip()

        if prefix_file is not None:
            with open(prefix_file, "r", encoding="utf-8") as f:
                self._prefix_text: Optional[str] = f.read().strip()
        else:
            self._prefix_text = None

        formatters = image_formatters or [
            "blur_bg_outline_fg",
            "zoom_crop_blur_bg_outline_fg",
            "full_scene",
        ]
        unknown = [f for f in formatters if f not in _FORMATTERS]
        if unknown:
            raise ValueError(
                f"Unknown image formatter(s): {unknown}. "
                f"Available: {available_image_formatters()}"
            )
        self._formatter_names = list(formatters)

        self._settings = {
            "mask_highlight_color": tuple(int(x) for x in mask_highlight_color),
            "full_mask_outline_thickness": int(full_mask_outline_thickness),
            "crop_mask_outline_thickness": int(crop_mask_outline_thickness),
            "background_dim_factor": float(background_dim_factor),
            "blur_radius": int(blur_radius),
            "highlight_alpha": float(highlight_alpha),
        }
        self._debug_dir: Optional[Path] = None

    def set_debug_dir(self, path: Path) -> None:
        """Optional hook so the inferencer can route per-object image dumps."""
        self._debug_dir = Path(path)

    # ---- AbsVLMPromptHandler ----

    def buildPromptAndImagesForObject(
        self,
        scene_caption: str,
        object_descriptions: Dict[Any, str],
        relationship_descriptions: Dict[Any, Any],
        time_and_date: Optional[str],
        image: torch.Tensor,
        instances: SegmentationResults,
        object_id: str,
    ) -> Tuple[str, List[Image.Image]]:
        # scene_caption, object_descriptions, relationship_descriptions,
        # time_and_date are ignored — this handler asks the VLM to reason
        # from the image alone.
        del scene_caption, object_descriptions, relationship_descriptions, time_and_date

        matches = (instances.ids == int(object_id)).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            raise ValueError(f"Object id {object_id} not found in instances.ids")
        object_index = int(matches[0].item())
        mask = instances.masks[object_index]

        images: List[Image.Image] = []
        for fname in self._formatter_names:
            img = _FORMATTERS[fname](image, mask, self._settings)
            if img is not None:
                images.append(img)

        if not images:
            raise RuntimeError(
                f"No images produced for object {object_id}; "
                f"formatters={self._formatter_names}"
            )

        if self._prefix_text:
            prompt = f"{self._prefix_text}\n\n{self._prompt_body}"
        else:
            prompt = self._prompt_body

        self._maybe_dump_debug_images(object_id, images)

        return prompt, images

    def parseSingleObjectPersistenceResponse(self, response_text: str) -> Dict[str, Any]:
        """Robustly extract one JSON object then validate the IQR+behav4 schema."""
        raw = _extract_last_single_object(response_text)
        return _validate_and_coerce(raw, response_text)

    # ---- internal ----

    def _maybe_dump_debug_images(self, object_id: str, images: List[Image.Image]) -> None:
        if self._debug_dir is None:
            return
        try:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(images):
                fname = self._debug_dir / f"obj_{object_id}_{i:02d}_{self._formatter_names[i]}.png"
                img.save(fname)
        except Exception:  # debug dumps should never break the pipeline
            pass


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", flags=re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Concatenate the contents of any ```...``` blocks if present, else return text.

    Models frequently wrap JSON in ```json ... ``` markdown fences. We pull the
    fence contents out so downstream scanners see clean JSON.
    """
    matches = _FENCE_RE.findall(text)
    if matches:
        return "\n".join(m.strip() for m in matches)
    return text


def _iter_balanced_json_blocks(text: str):
    """Yield every brace- or bracket-balanced JSON-looking substring in ``text``.

    Walks character-by-character tracking depth and quote state. Skips
    backslash-escaped chars inside strings so braces embedded in JSON strings
    don't break balance. Yields each top-level ``{...}`` or ``[...]`` span as a
    raw string; the caller decides whether it parses.
    """
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch not in "{[":
            i += 1
            continue
        opener = ch
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_str = False
        esc = False
        j = i
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        yield text[i : j + 1]
                        break
            j += 1
        i = j + 1


def _extract_last_single_object(response_text: str) -> Dict[str, Any]:
    """Prefer JSON after the last ``</think>`` tag, strip markdown fences, scan
    for brace-balanced object/array JSON, and return the last valid dict-shaped
    candidate.
    """
    cleaned = response_text.strip()

    def scan(text: str) -> Dict[str, Any]:
        text = _strip_code_fences(text.strip()).strip()
        candidates: List[Dict[str, Any]] = []
        if (text.startswith("{") and text.endswith("}")) or (
            text.startswith("[") and text.endswith("]")
        ):
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    candidates.append(obj)
                elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                    candidates.append(obj[0])
            except json.JSONDecodeError:
                pass
        for block in _iter_balanced_json_blocks(text):
            try:
                obj = json.loads(block)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                candidates.append(obj)
            elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
                candidates.append(obj[0])
        if not candidates:
            raise ValueError(
                f"No valid single-object JSON found in text. "
                f"scanned_len={len(text)} scanned[:1500]={text[:1500]!r} "
                f"full_len={len(response_text)} full_tail[-500:]={response_text[-500:]!r}"
            )
        return candidates[-1]

    end = cleaned.rfind("</think>")
    if end != -1:
        try:
            return scan(cleaned[end + len("</think>") :])
        except ValueError:
            pass
    return scan(cleaned)


def _validate_and_coerce(raw: Dict[str, Any], response_text: str) -> Dict[str, Any]:
    """Ensure the IQR+behav4 schema is present; coerce numerics; return a new dict."""
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(
            f"Parsed JSON missing required keys {missing}. "
            f"Got keys: {sorted(raw.keys())}. Response[:300]: {response_text[:300]!r}"
        )

    try:
        p25 = float(raw["p25_persistence_hours"])
        p75 = float(raw["p75_persistence_hours"])
        conf = float(raw["persistence_conf"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Numeric coercion failed: {e}. Raw: {raw!r}") from e

    if not (p25 > 0 and p75 > 0):
        raise ValueError(f"p25/p75 must be positive; got p25={p25} p75={p75}")
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"persistence_conf out of [0,1]: {conf}")

    out: Dict[str, Any] = dict(raw)
    out["p25_persistence_hours"] = p25
    out["p75_persistence_hours"] = p75
    out["persistence_conf"] = conf
    return out
