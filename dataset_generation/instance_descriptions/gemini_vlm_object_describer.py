"""
Object describer that queries a Google Gemini VL model
(``gemini-2.5-pro`` and siblings) via the ``google-genai`` SDK.

Same interface and image-prep behaviour as ``GemmaVLMObjectDescriber``: makes
one VLM call per object using the raw RGB scene plus a per-object overlay
(mask highlight / bounding box / zoomed crop / context crop / arrow). The
images are produced by the same helpers (this file is the Gemma describer
with engine construction and the per-call image-encoding swapped); future
changes to those helpers must be kept in sync between the two files.

The Gemini client, cumulative token tracker, and usage-log file are owned by
the process-wide singleton in ``gemini._shared`` — configure it once at
process startup via ``configure_gemini_shared(...)`` (eval_persistence.py
already does this when the top-level eval config has a ``gemini:`` block).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import math

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from dataset_generation.instance_descriptions.abs_object_describer import AbstractObjectDescriber
from dataset_generation.intermediate_results_logger.debug_data import DebugPayload, DebugPayloadFactory
from dataset_generation.llm_persistence_querying.gemini.gemini_chat_engine import (
    GeminiChatEngine,
    GeminiChatRequest,
    pil_to_part,
)
from instance_segmentation.abs_instance_segmenter import SegmentationResults
from utils.image_utils import image_tensor_to_pil
from utils.timing import timer
from utils.visualization_utils import mask_highlight_image


_DEFAULT_PROMPT = (
    "You are given two images of the same scene. "
    "The first is the original image. "
    "The second has one object highlighted with a bounding box. "
    "Describe the highlighted object in detail. "
    "Comment on properties that might affect its motion or persistence in the scene over time: "
    "whether it is attached to a surface or part of the environment; "
    "if free-standing, how easy it would be for it to change position; "
    "whether external forces (wind, people, vehicles) might move or deform it; "
    "any signs of imminent motion (walking posture, wheels turned, being held, etc.)."
)


class GeminiVLMObjectDescriber(AbstractObjectDescriber):
    """Describes segmented objects by querying a Gemini VL model with one
    request per object containing the raw scene plus the configured per-object
    overlay images. Structurally mirrors GemmaVLMObjectDescriber; differences
    are confined to the engine wiring.
    """

    def __init__(
        self,
        *,
        name: str,
        model_name: str = "gemini-2.5-pro",
        prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        prefix: Optional[str] = None,
        prefix_file: Optional[str] = None,
        # --- Gemini engine knobs (everything else lives on the shared singleton) ---
        thinking_config: Optional[Any] = None,
        disable_thinking: bool = False,
        extra_generation_config: Optional[Dict[str, Any]] = None,
        allow_unvetted_model: bool = False,
        # --- generation / sampling ---
        max_new_tokens: int = 4096,
        temperature: float = 0.2,
        top_p: float = 0.9,
        # --- overlay appearance (identical to Gemma describer) ---
        mask_style: Optional[str] = "blur_bg_outline_fg",
        mask_highlight_color: Tuple[int, int, int] = (0, 255, 255),
        mask_dilation_px: int = 5,
        include_zoomed_crop: bool = False,
        include_arrow_image: bool = False,
        arrow_color: Tuple[int, int, int] = (0, 255, 255),
        include_box_image: bool = False,
        box_color: Tuple[int, int, int] = (0, 255, 255),
        box_thickness: int = 5,
        box_margin: int = 3,
        include_context_crop: bool = False,
        context_crop_pad_frac: float = 0.35,
        context_crop_min_size_frac: float = 0.25,
        include_full_img_last: bool = False,
        blur_radius: int = 8,
        mask_background_dim_factor: float = 0.35,
        mask_outline_thickness: int = 1,
        highlight_alpha: float = 0.0,
        mask_crop: bool = False,
        crop_mask_style: Optional[str] = None,
        crop_mask_highlight_color: Tuple[int, int, int] = (0, 255, 255),
        crop_mask_dilation_px: int = 5,
        crop_blur_radius: int = 0,
        crop_mask_background_dim_factor: float = 0.35,
        crop_highlight_alpha: float = 0.0,
        crop_mask_outline_thickness: int = 1,
        # --- per-scene fan-out ---
        max_concurrency: int = 16,
        chunk_size: Optional[int] = None,
    ):
        if name is None:
            raise ValueError("name must be provided for GeminiVLMObjectDescriber")
        if prompt is not None and prompt_file is not None:
            raise ValueError("Only one of prompt or prompt_file should be provided.")
        if prefix is not None and prefix_file is not None:
            raise ValueError("Only one of prefix or prefix_file should be provided.")
        super().__init__(name=name)

        if prompt_file is not None:
            with open(prompt_file, "r") as f:
                prompt = f.read()
        if prefix_file is not None:
            with open(prefix_file, "r") as f:
                prefix = f.read()

        self._prefix = (prefix.strip() + "\n\n") if prefix else ""
        self.prompt = (self._prefix + (prompt or _DEFAULT_PROMPT)).strip()
        self.mask_style = mask_style
        self.mask_highlight_color = tuple(int(x) for x in mask_highlight_color)
        self.mask_dilation_px = int(mask_dilation_px)
        self.include_zoomed_crop = include_zoomed_crop
        self.include_arrow_image = include_arrow_image
        self.arrow_color = tuple(int(x) for x in arrow_color)
        self.include_box_image = include_box_image
        self.box_color = tuple(int(x) for x in box_color)
        self.box_thickness = int(box_thickness)
        self.box_margin = int(box_margin)
        self.include_context_crop = bool(include_context_crop)
        self.context_crop_pad_frac = float(context_crop_pad_frac)
        self.context_crop_min_size_frac = float(context_crop_min_size_frac)
        self.include_full_img_last = bool(include_full_img_last)
        self.blur_radius = int(blur_radius)
        self.mask_outline_thickness = int(mask_outline_thickness)
        self.mask_background_dim_factor = float(mask_background_dim_factor)
        self.highlight_alpha = highlight_alpha
        self.mask_crop = bool(mask_crop)
        self.crop_mask_style = crop_mask_style
        self.crop_mask_highlight_color = tuple(int(x) for x in crop_mask_highlight_color)
        self.crop_mask_dilation_px = int(crop_mask_dilation_px)
        self.crop_blur_radius = int(crop_blur_radius)
        self.crop_mask_background_dim_factor = float(crop_mask_background_dim_factor)
        self.crop_highlight_alpha = float(crop_highlight_alpha)
        self.crop_mask_outline_thickness = int(crop_mask_outline_thickness)

        self._debug_dir: Optional[Path] = None
        self._describe_call_idx: int = 0
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_concurrency = int(max_concurrency)
        self.chunk_size = int(chunk_size) if chunk_size is not None else self.max_concurrency
        if self.chunk_size < self.max_concurrency:
            warnings.warn(
                f"{type(self).__name__}: chunk_size ({self.chunk_size}) is less "
                f"than max_concurrency ({self.max_concurrency}); the effective "
                f"in-flight cap will be chunk_size, so max_concurrency is wasted. "
                f"Either raise chunk_size to >= max_concurrency or lower "
                f"max_concurrency to chunk_size.",
                stacklevel=2,
            )

        self._per_query_timer_name = "Timer::gemini_vlm_obj_describer_vlm_query"

        self._engine = GeminiChatEngine(
            model_name=model_name,
            thinking_config=thinking_config,
            disable_thinking=disable_thinking,
            extra_generation_config=extra_generation_config,
            allow_unvetted_model=allow_unvetted_model,
            max_concurrent_requests=self.max_concurrency,
        )

    def set_debug_dir(self, debug_dir: Path) -> None:
        self._debug_dir = debug_dir

    def wants_warmup(self) -> bool:
        """Gemini is HTTP-only; the first call only pays for a TCP/TLS
        handshake (~hundreds of ms) — negligible vs. the seconds-long Gemini
        decode latency, and not worth one real billed API call to amortize.
        """
        return False

    def describeObjects(
        self,
        image: torch.Tensor,
        instances: SegmentationResults,
        image_id: str,
    ) -> Tuple[Dict[int, str], DebugPayload]:
        image_call_idx = self._describe_call_idx
        self._describe_call_idx += 1

        pil_image = image_tensor_to_pil(image)

        # Encode the full scene once as a Gemini ``Part`` and reuse the
        # pre-encoded handle across every per-object request that includes it,
        # so we don't JPEG/base64 the same image N times. Per-object overlays
        # stay as PIL images — the engine encodes them when each request fires.
        full_scene_for_engine: Any = pil_to_part(pil_image)

        print(f"Input image resolution: {pil_image.width}x{pil_image.height}")
        descriptions: Dict[int, str] = {}
        reasoning_by_id: Dict[str, str] = {}
        debug_payload = DebugPayloadFactory.instance().create("object_descriptions", image_id)
        total = len(instances.masks)
        if total == 0:
            return descriptions, debug_payload

        # Process objects in chunks so we don't hold all N per-object overlay
        # images and request payloads in memory at once.
        chunk_size = max(1, self.chunk_size)
        result_idx = 0
        for chunk_start in range(0, total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total)
            chunk_obj_ids: List[int] = []
            chunk_requests: List[GeminiChatRequest] = []

            for i in range(chunk_start, chunk_end):
                mask = instances.masks[i]
                obj_id = int(instances.ids[i].item())

                debug_images, request_images = self._build_query_images(
                    image=image,
                    pil_image=pil_image,
                    full_scene_for_engine=full_scene_for_engine,
                    mask=mask,
                )

                for q_idx, q_img in enumerate(debug_images):
                    if q_img is None:
                        continue
                    debug_payload.add(
                        f"query_img_{obj_id}_qIm{q_idx:02d}",
                        q_img,
                        kind="image",
                    )

                chunk_obj_ids.append(obj_id)
                chunk_requests.append(GeminiChatRequest(
                    prompt=self.prompt,
                    images=request_images,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                ))

            print(
                f"Describing objects {chunk_start + 1}-{chunk_end} of {total} "
                f"via chat_many (max_concurrency="
                f"{min(self.max_concurrency, len(chunk_requests))})..."
            )
            with timer.time_block(
                self._per_query_timer_name,
                batch_count=len(chunk_requests),
            ):
                results = self._engine.chat_many(
                    chunk_requests,
                    max_concurrency=self.max_concurrency,
                )

            for obj_id, result in zip(chunk_obj_ids, results):
                description = result.text
                self._save_debug_text(obj_id, result_idx, image_call_idx, "", description)
                descriptions[obj_id] = description
                if result.reasoning:
                    reasoning_by_id[str(obj_id)] = result.reasoning
                result_idx += 1

            del chunk_requests, chunk_obj_ids, results

        if reasoning_by_id:
            debug_payload.add(
                f"reasoning_traces_img{image_call_idx:04d}",
                reasoning_by_id,
                kind="json",
            )

        return descriptions, debug_payload

    def _build_query_images(
        self,
        *,
        image: torch.Tensor,
        pil_image: "Image.Image",
        full_scene_for_engine: Any,
        mask: torch.Tensor,
    ) -> Tuple[List["Image.Image"], List[Union["Image.Image", Any]]]:
        """Build the per-object image lists for one mask.

        ``debug_images`` is the PIL-only list used for the debug payload.
        ``request_images`` is the same list with the full scene swapped for
        its pre-encoded ``Part`` so the engine doesn't re-encode it for every
        request.
        """
        display_mask = self._dilate_mask(mask, self.mask_dilation_px)
        crop_mask = display_mask if self.mask_crop else None
        if (crop_mask is not None) and (self.crop_mask_style is not None) and (self.crop_mask_dilation_px != self.mask_dilation_px):
            crop_mask = self._dilate_mask(mask, self.crop_mask_dilation_px)

        if self.include_full_img_last:
            debug_images: List[Image.Image] = []
            request_images: List[Union[Image.Image, Any]] = []
        else:
            debug_images = [pil_image]
            request_images = [full_scene_for_engine]

        if self.mask_style is not None:
            extra_kwargs = {
                "blur_radius": self.blur_radius,
                "background_dim_factor": self.mask_background_dim_factor,
                "highlight_alpha": self.highlight_alpha,
                "outline_thickness": self.mask_outline_thickness,
            }
            highlighted_np = mask_highlight_image(
                image,
                display_mask,
                style=self.mask_style,
                mask_highlight_color=self.mask_highlight_color,
                extra_kwargs=extra_kwargs,
            )
            hl_pil = Image.fromarray(highlighted_np)
            debug_images.append(hl_pil)
            request_images.append(hl_pil)

        if self.include_box_image:
            box_img = self._draw_box_image(pil_image, display_mask)
            if box_img is not None:
                debug_images.append(box_img)
                request_images.append(box_img)

        if self.include_zoomed_crop or self.include_context_crop:

            imageToCrop = pil_image
            if (self.mask_crop and self.mask_style is not None):
                extra_kwargs = {
                    "blur_radius": self.crop_blur_radius,
                    "background_dim_factor": self.crop_mask_background_dim_factor,
                    "highlight_alpha": self.crop_highlight_alpha,
                    "outline_thickness": self.crop_mask_outline_thickness,
                }
                highlighted_np = mask_highlight_image(
                    image,
                    crop_mask,
                    style=self.mask_style,
                    mask_highlight_color=self.mask_highlight_color,
                    extra_kwargs=extra_kwargs,
                )
                imageToCrop = Image.fromarray(highlighted_np)

            if self.include_zoomed_crop:
                crop = self._compute_zoom_crop(
                    crop_mask, pil_image.width, pil_image.height
                )
                if crop is not None:
                    crop_img = imageToCrop.crop(crop)
                    debug_images.append(crop_img)
                    request_images.append(crop_img)

            if self.include_context_crop:
                crop = self._compute_context_crop(
                    crop_mask,
                    pil_image.width,
                    pil_image.height,
                    pad_frac=self.context_crop_pad_frac,
                    min_size_frac=self.context_crop_min_size_frac,
                )
                if crop is not None:
                    crop_img = imageToCrop.crop(crop)
                    debug_images.append(crop_img)
                    request_images.append(crop_img)

        if self.include_arrow_image:
            target = self._compute_arrow_target(display_mask)
            if target is not None:
                arrow_img = self._draw_arrow_image(pil_image, target[0], target[1])
                debug_images.append(arrow_img)
                request_images.append(arrow_img)

        if self.include_full_img_last:
            debug_images.append(pil_image)
            request_images.append(full_scene_for_engine)

        return debug_images, request_images

    def _compute_zoom_crop(
        self, mask: torch.Tensor, img_w: int, img_h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return (left, top, right, bottom) crop box centered on the mask."""
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
        mask_h = rmax - rmin + 1
        mask_w = cmax - cmin + 1
        center_y = (rmin + rmax) / 2.0
        center_x = (cmin + cmax) / 2.0

        scale = min(max(3 * mask_w / img_w, 3 * mask_h / img_h), 1.0)
        crop_w = int(np.ceil(scale * img_w))
        crop_h = int(np.ceil(scale * img_h))

        half_w = min(crop_w / 2.0, center_x, img_w - center_x)
        half_h = min(crop_h / 2.0, center_y, img_h - center_y)
        left   = int(round(center_x - half_w))
        right  = int(round(center_x + half_w))
        top    = int(round(center_y - half_h))
        bottom = int(round(center_y + half_h))
        return left, top, right, bottom

    def _compute_arrow_target(self, mask: torch.Tensor) -> Optional[Tuple[int, int]]:
        m = mask.detach().cpu()
        if m.dim() == 3 and m.shape[0] == 1:
            m = m[0]
        m_np = m.numpy().astype(bool)

        rows = np.any(m_np, axis=1)
        if not rows.any():
            return None

        rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        center_row = (rmin + rmax) // 2

        for delta in range(0, rmax - rmin + 1):
            for row in {center_row - delta, center_row + delta}:
                if 0 <= row < m_np.shape[0]:
                    cols = np.where(m_np[row])[0]
                    if len(cols):
                        return int(np.median(cols)), row
        return None

    def _compute_context_crop(
        self,
        mask: torch.Tensor,
        img_w: int,
        img_h: int,
        pad_frac: float = 0.35,
        min_size_frac: float = 0.25,
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

        bw = cmax - cmin + 1
        bh = rmax - rmin + 1

        pad = int(round(pad_frac * max(bw, bh)))
        left = cmin - pad
        right = cmax + pad
        top = rmin - pad
        bottom = rmax + pad

        min_w = int(round(min_size_frac * img_w))
        min_h = int(round(min_size_frac * img_h))

        cx = (left + right) / 2
        cy = (top + bottom) / 2

        crop_w = max(right - left, min_w)
        crop_h = max(bottom - top, min_h)

        left = int(round(cx - crop_w / 2))
        right = int(round(cx + crop_w / 2))
        top = int(round(cy - crop_h / 2))
        bottom = int(round(cy + crop_h / 2))

        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > img_w:
            left -= right - img_w
            right = img_w
        if bottom > img_h:
            top -= bottom - img_h
            bottom = img_h

        left = max(0, left)
        top = max(0, top)
        right = min(img_w, right)
        bottom = min(img_h, bottom)

        return left, top, right, bottom

    def _draw_arrow_image(
        self,
        pil_image: "Image.Image",
        target_x: int,
        target_y: int,
    ) -> "Image.Image":
        img = pil_image.copy()
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size

        cx, cy = img_w / 2.0, img_h / 2.0
        dx = target_x - cx
        dy = target_y - cy
        inv_sqrt2 = 1.0 / math.sqrt(2)
        arrow_dx = math.copysign(inv_sqrt2, dx if dx != 0 else 1.0)
        arrow_dy = math.copysign(inv_sqrt2, dy if dy != 0 else 1.0)

        arrow_len = max(min(img_w, img_h) // 6, 50)
        tail_x = target_x - arrow_dx * arrow_len
        tail_y = target_y - arrow_dy * arrow_len

        lw = max(3, min(img_w, img_h) // 150)
        color = self.arrow_color
        draw.line([(tail_x, tail_y), (target_x, target_y)], fill=color, width=lw)

        head = lw * 4
        px, py = -arrow_dy, arrow_dx
        tip = (target_x, target_y)
        p1 = (target_x - arrow_dx * head + px * head, target_y - arrow_dy * head + py * head)
        p2 = (target_x - arrow_dx * head - px * head, target_y - arrow_dy * head - py * head)
        draw.polygon([tip, p1, p2], fill=color)

        return img

    def _draw_box_image(
        self,
        pil_image: "Image.Image",
        mask: torch.Tensor,
    ) -> Optional["Image.Image"]:
        m = mask.detach().cpu()
        if m.dim() == 3 and m.shape[0] == 1:
            m = m[0]
        m_np = m.numpy().astype(bool)

        rows = np.any(m_np, axis=1)
        cols = np.any(m_np, axis=0)
        if not rows.any():
            return None

        img_w, img_h = pil_image.size
        rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

        left   = max(0, cmin - self.box_margin)
        top    = max(0, rmin - self.box_margin)
        right  = min(img_w - 1, cmax + self.box_margin)
        bottom = min(img_h - 1, rmax + self.box_margin)

        img = pil_image.copy()
        draw = ImageDraw.Draw(img)
        for i in range(self.box_thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=self.box_color,
            )
        return img

    def _dilate_mask(self, mask: torch.Tensor, dilation_px: int = None) -> torch.Tensor:
        if dilation_px is None or dilation_px <= 0:
            return mask
        m = mask.detach().cpu()
        if m.dim() == 3 and m.shape[0] == 1:
            m = m[0]
        dilated = binary_dilation(m.numpy().astype(bool), iterations=dilation_px)
        return torch.from_numpy(dilated).to(dtype=mask.dtype)

    def _save_debug_text(
        self,
        obj_id: int,
        idx: int,
        image_call_idx: int,
        reasoning: str,
        content: str,
    ) -> None:
        if self._debug_dir is None:
            return
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        text_path = self._debug_dir / f"img{image_call_idx:04d}_obj{idx:03d}_id{obj_id}_response.txt"
        with open(text_path, "w") as f:
            f.write("=== REASONING ===\n")
            f.write(reasoning or "(none)\n")
            f.write("\n=== OUTPUT ===\n")
            f.write(content)
        print(
            f"[gemini_vlm_describer] debug text → {text_path}",
            file=sys.stderr, flush=True,
        )

    def _save_debug_images(
        self,
        raw: "Image.Image",
        highlighted: "Image.Image",
        obj_id: int,
        idx: int,
        zoomed: Optional["Image.Image"] = None,
        arrow_img: Optional["Image.Image"] = None,
        image_call_idx: int = 0,
    ) -> None:
        if self._debug_dir is None:
            return
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        raw_path = self._debug_dir / f"img{image_call_idx:04d}_obj{idx:03d}_id{obj_id}_raw.jpg"
        hl_path  = self._debug_dir / f"img{image_call_idx:04d}_obj{idx:03d}_id{obj_id}_highlighted.jpg"
        raw.save(raw_path, format="JPEG", quality=90)
        highlighted.save(hl_path, format="JPEG", quality=90)
        saved = [raw_path, hl_path]
        if zoomed is not None:
            zoom_path = self._debug_dir / f"img{image_call_idx:04d}_obj{idx:03d}_id{obj_id}_zoomed.jpg"
            zoomed.save(zoom_path, format="JPEG", quality=90)
            saved.append(zoom_path)
        if arrow_img is not None:
            arrow_path = self._debug_dir / f"img{image_call_idx:04d}_obj{idx:03d}_id{obj_id}_arrow.jpg"
            arrow_img.save(arrow_path, format="JPEG", quality=90)
            saved.append(arrow_path)
        print(
            f"[gemini_vlm_describer] debug images → {', '.join(str(p) for p in saved)}",
            file=sys.stderr, flush=True,
        )
