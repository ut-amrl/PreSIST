"""
Direct-future-existence VLM querier for Gemma 4 VL models (google/gemma-4-*-it).

Mirrors :class:`GemmaVLMObjectDescriber` in structure: a single VLM call per
object using the Gemma chat engine (local Transformers or vLLM-server), with
the same set of rich image-overlay options (mask highlight, bounding box,
zoomed/context crops, arrow). Differs only in what the VLM is asked: instead of
a free-text description, it returns probabilities at the supplied future time
horizons.

The constructor signature deliberately tracks ``GemmaVLMObjectDescriber`` so the
same overlay configuration can be reused for both describer and direct querier
without copy-paste drift.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from dataset_generation.llm_persistence_querying.abs_direct_future_existence_vlm_querier import (
    AbsDirectFutureExistenceVLMQuerier,
    format_horizons_block,
    parse_probs_aligned_to_query_times,
)
from dataset_generation.llm_persistence_querying.gemma.gemma_chat_engine import (
    GemmaChatEngine,
    pil_to_data_url_jpeg,
)
from utils.image_utils import image_tensor_to_pil
from utils.timing import querier_timer_suffix, timer
from utils.visualization_utils import mask_highlight_image


_DEFAULT_PROMPT = (
    "Estimate, independently for each of the time horizons "
    "listed below (seconds from now), the probability that the highlighted object will still "
    "exist at this location at that future time. Probabilities are floats in [0,1] and should "
    "be non-increasing as the horizon grows.\n\n"
    "Horizons (seconds):\n{query_times_block}\n\n"
    "Respond with ONLY a JSON array — one entry per horizon, in the SAME ORDER as listed:\n"
    "[{{\"dt_seconds\": <horizon>, \"prob\": <float in [0,1]>}}, ...]\n"
    "No markdown, no extra text."
)


class GemmaDirectFutureExistenceVLMQuerier(AbsDirectFutureExistenceVLMQuerier):
    """
    Direct-future-existence querier backed by ``GemmaChatEngine``.

    The prompt must contain the placeholder ``{query_times_block}``. All other
    visual context comes from the rendered image list (original + overlay
    variants chosen by the constructor flags).
    """

    def __init__(
        self,
        *,
        name: str,
        model_name: str = "google/gemma-4-27b-it",
        prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        prefix: Optional[str] = None,
        prefix_file: Optional[str] = None,
        # --- mode ---
        use_vllm_server: bool = False,
        enable_thinking: bool = False,
        # --- local (Transformers) runtime ---
        device_map: Union[str, Dict[str, Any]] = "auto",
        torch_dtype: Union[torch.dtype, str] = "auto",
        # --- vLLM server runtime ---
        vllm_url_root: str = "http://localhost",
        vllm_port: int = 9001,
        vllm_url_suffix: str = "/v1",
        vllm_api_key: str = "EMPTY",
        vllm_request_timeout_s: float = 120.0,
        vllm_extra_request_params: Optional[Dict[str, Any]] = None,
        # --- generation / sampling ---
        max_new_tokens: int = 4096,
        temperature: float = 0.2,
        top_p: float = 0.9,
        # --- overlay appearance (mirrors GemmaVLMObjectDescriber) ---
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
    ) -> None:
        if prompt is not None and prompt_file is not None:
            raise ValueError("Only one of prompt or prompt_file should be provided.")
        if prefix is not None and prefix_file is not None:
            raise ValueError("Only one of prefix or prefix_file should be provided.")
        super().__init__(name=name)

        if prompt_file is not None:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read()
        if prefix_file is not None:
            with open(prefix_file, "r", encoding="utf-8") as f:
                prefix = f.read()

        self._prefix = (prefix.strip() + "\n\n") if prefix else ""
        self._prompt_template = (self._prefix + (prompt or _DEFAULT_PROMPT)).strip()

        self.mask_style = mask_style
        self.mask_highlight_color = tuple(int(x) for x in mask_highlight_color)
        self.mask_dilation_px = int(mask_dilation_px)
        self.include_zoomed_crop = bool(include_zoomed_crop)
        self.include_arrow_image = bool(include_arrow_image)
        self.arrow_color = tuple(int(x) for x in arrow_color)
        self.include_box_image = bool(include_box_image)
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
        self.highlight_alpha = float(highlight_alpha)
        self.mask_crop = bool(mask_crop)
        self.crop_mask_style = crop_mask_style
        self.crop_mask_highlight_color = tuple(int(x) for x in crop_mask_highlight_color)
        self.crop_mask_dilation_px = int(crop_mask_dilation_px)
        self.crop_blur_radius = int(crop_blur_radius)
        self.crop_mask_background_dim_factor = float(crop_mask_background_dim_factor)
        self.crop_highlight_alpha = float(crop_highlight_alpha)
        self.crop_mask_outline_thickness = int(crop_mask_outline_thickness)

        self.enable_thinking = bool(enable_thinking)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        self._per_query_timer_name = (
            "Timer::gemma_direct_future_existence_query"
            + querier_timer_suffix(
                use_vllm_server=use_vllm_server,
                enable_thinking=self.enable_thinking,
            )
        )

        self._engine = GemmaChatEngine(
            model_name=model_name,
            use_vllm_server=use_vllm_server,
            enable_thinking=self.enable_thinking,
            device_map=device_map,
            torch_dtype=torch_dtype,
            vllm_url_root=vllm_url_root,
            vllm_port=vllm_port,
            vllm_url_suffix=vllm_url_suffix,
            vllm_api_key=vllm_api_key,
            vllm_request_timeout_s=vllm_request_timeout_s,
            vllm_extra_request_params=vllm_extra_request_params,
        )

    # ----------------------------------------------------------------------
    # AbsDirectFutureExistenceVLMQuerier interface
    # ----------------------------------------------------------------------

    def query(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        query_times_seconds: Sequence[float],
    ) -> List[float]:
        prompt = self._prompt_template.format(
            query_times_block=format_horizons_block(query_times_seconds),
        )
        pil_image = image_tensor_to_pil(image)

        full_scene_for_engine: Union[Image.Image, str] = (
            pil_to_data_url_jpeg(pil_image) if self._engine.use_vllm_server else pil_image
        )
        request_images = self._build_query_images(
            image=image,
            pil_image=pil_image,
            full_scene_for_engine=full_scene_for_engine,
            mask=mask,
        )

        with timer.time_block(self._per_query_timer_name, batch_count=1):
            result = self._engine.chat(
                prompt=prompt,
                images=request_images,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        return parse_probs_aligned_to_query_times(result.text, query_times_seconds)

    # ----------------------------------------------------------------------
    # Image / overlay construction (lifted from GemmaVLMObjectDescriber)
    # ----------------------------------------------------------------------

    def _build_query_images(
        self,
        *,
        image: torch.Tensor,
        pil_image: Image.Image,
        full_scene_for_engine: Union[Image.Image, str],
        mask: torch.Tensor,
    ) -> List[Union[Image.Image, str]]:
        display_mask = self._dilate_mask(mask, self.mask_dilation_px)
        crop_mask = display_mask if self.mask_crop else None
        if (
            crop_mask is not None
            and self.crop_mask_style is not None
            and self.crop_mask_dilation_px != self.mask_dilation_px
        ):
            crop_mask = self._dilate_mask(mask, self.crop_mask_dilation_px)

        request_images: List[Union[Image.Image, str]] = []
        if not self.include_full_img_last:
            request_images.append(full_scene_for_engine)

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
            request_images.append(Image.fromarray(highlighted_np))

        if self.include_box_image:
            box_img = self._draw_box_image(pil_image, display_mask)
            if box_img is not None:
                request_images.append(box_img)

        if self.include_zoomed_crop or self.include_context_crop:
            image_to_crop = pil_image
            if self.mask_crop and self.mask_style is not None:
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
                image_to_crop = Image.fromarray(highlighted_np)

            if self.include_zoomed_crop:
                crop = self._compute_zoom_crop(crop_mask, pil_image.width, pil_image.height)
                if crop is not None:
                    request_images.append(image_to_crop.crop(crop))

            if self.include_context_crop:
                crop = self._compute_context_crop(
                    crop_mask,
                    pil_image.width,
                    pil_image.height,
                    pad_frac=self.context_crop_pad_frac,
                    min_size_frac=self.context_crop_min_size_frac,
                )
                if crop is not None:
                    request_images.append(image_to_crop.crop(crop))

        if self.include_arrow_image:
            target = self._compute_arrow_target(display_mask)
            if target is not None:
                request_images.append(self._draw_arrow_image(pil_image, target[0], target[1]))

        if self.include_full_img_last:
            request_images.append(full_scene_for_engine)

        return request_images

    @staticmethod
    def _dilate_mask(mask: torch.Tensor, dilation_px: int) -> torch.Tensor:
        if dilation_px is None or dilation_px <= 0:
            return mask
        m = mask.detach().cpu()
        if m.dim() == 3 and m.shape[0] == 1:
            m = m[0]
        dilated = binary_dilation(m.numpy().astype(bool), iterations=dilation_px)
        return torch.from_numpy(dilated).to(dtype=mask.dtype)

    @staticmethod
    def _mask_bbox(mask: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
        m = mask.detach().cpu()
        if m.dim() == 3 and m.shape[0] == 1:
            m = m[0]
        m_np = m.numpy().astype(bool)
        rows = np.any(m_np, axis=1)
        cols = np.any(m_np, axis=0)
        if not rows.any():
            return None
        rmin = int(np.where(rows)[0][0])
        rmax = int(np.where(rows)[0][-1])
        cmin = int(np.where(cols)[0][0])
        cmax = int(np.where(cols)[0][-1])
        return cmin, rmin, cmax, rmax

    def _compute_zoom_crop(
        self, mask: torch.Tensor, img_w: int, img_h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        bbox = self._mask_bbox(mask)
        if bbox is None:
            return None
        cmin, rmin, cmax, rmax = bbox
        mask_w, mask_h = cmax - cmin + 1, rmax - rmin + 1
        center_x = (cmin + cmax) / 2.0
        center_y = (rmin + rmax) / 2.0

        scale = min(max(3 * mask_w / img_w, 3 * mask_h / img_h), 1.0)
        crop_w = int(np.ceil(scale * img_w))
        crop_h = int(np.ceil(scale * img_h))

        half_w = min(crop_w / 2.0, center_x, img_w - center_x)
        half_h = min(crop_h / 2.0, center_y, img_h - center_y)
        return (
            int(round(center_x - half_w)),
            int(round(center_y - half_h)),
            int(round(center_x + half_w)),
            int(round(center_y + half_h)),
        )

    def _compute_context_crop(
        self,
        mask: torch.Tensor,
        img_w: int,
        img_h: int,
        pad_frac: float,
        min_size_frac: float,
    ) -> Optional[Tuple[int, int, int, int]]:
        bbox = self._mask_bbox(mask)
        if bbox is None:
            return None
        cmin, rmin, cmax, rmax = bbox
        bw, bh = cmax - cmin + 1, rmax - rmin + 1

        pad = int(round(pad_frac * max(bw, bh)))
        left, right = cmin - pad, cmax + pad
        top, bottom = rmin - pad, rmax + pad

        min_w = int(round(min_size_frac * img_w))
        min_h = int(round(min_size_frac * img_h))
        cx, cy = (left + right) / 2, (top + bottom) / 2
        crop_w = max(right - left, min_w)
        crop_h = max(bottom - top, min_h)
        left, right = int(round(cx - crop_w / 2)), int(round(cx + crop_w / 2))
        top, bottom = int(round(cy - crop_h / 2)), int(round(cy + crop_h / 2))

        if left < 0:
            right -= left; left = 0
        if top < 0:
            bottom -= top; top = 0
        if right > img_w:
            left -= right - img_w; right = img_w
        if bottom > img_h:
            top -= bottom - img_h; bottom = img_h

        return max(0, left), max(0, top), min(img_w, right), min(img_h, bottom)

    def _compute_arrow_target(self, mask: torch.Tensor) -> Optional[Tuple[int, int]]:
        m = mask.detach().cpu()
        if m.dim() == 3 and m.shape[0] == 1:
            m = m[0]
        m_np = m.numpy().astype(bool)
        rows = np.any(m_np, axis=1)
        if not rows.any():
            return None
        rmin = int(np.where(rows)[0][0])
        rmax = int(np.where(rows)[0][-1])
        center_row = (rmin + rmax) // 2
        for delta in range(0, rmax - rmin + 1):
            for row in {center_row - delta, center_row + delta}:
                if 0 <= row < m_np.shape[0]:
                    cols = np.where(m_np[row])[0]
                    if len(cols):
                        return int(np.median(cols)), row
        return None

    def _draw_arrow_image(
        self, pil_image: Image.Image, target_x: int, target_y: int
    ) -> Image.Image:
        img = pil_image.copy()
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size
        cx, cy = img_w / 2.0, img_h / 2.0
        dx, dy = target_x - cx, target_y - cy
        inv_sqrt2 = 1.0 / math.sqrt(2)
        arrow_dx = math.copysign(inv_sqrt2, dx if dx != 0 else 1.0)
        arrow_dy = math.copysign(inv_sqrt2, dy if dy != 0 else 1.0)

        arrow_len = max(min(img_w, img_h) // 6, 50)
        tail_x = target_x - arrow_dx * arrow_len
        tail_y = target_y - arrow_dy * arrow_len
        lw = max(3, min(img_w, img_h) // 150)
        draw.line([(tail_x, tail_y), (target_x, target_y)], fill=self.arrow_color, width=lw)

        head = lw * 4
        px, py = -arrow_dy, arrow_dx
        tip = (target_x, target_y)
        p1 = (target_x - arrow_dx * head + px * head, target_y - arrow_dy * head + py * head)
        p2 = (target_x - arrow_dx * head - px * head, target_y - arrow_dy * head - py * head)
        draw.polygon([tip, p1, p2], fill=self.arrow_color)
        return img

    def _draw_box_image(
        self, pil_image: Image.Image, mask: torch.Tensor
    ) -> Optional[Image.Image]:
        bbox = self._mask_bbox(mask)
        if bbox is None:
            return None
        cmin, rmin, cmax, rmax = bbox
        img_w, img_h = pil_image.size
        left = max(0, cmin - self.box_margin)
        top = max(0, rmin - self.box_margin)
        right = min(img_w - 1, cmax + self.box_margin)
        bottom = min(img_h - 1, rmax + self.box_margin)

        img = pil_image.copy()
        draw = ImageDraw.Draw(img)
        for i in range(self.box_thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.box_color)
        return img
