"""
Object describer that queries a Gemma 4 VL model (google/gemma-4-*-it).

Makes one VLM call per object with two images:
  1. The raw RGB image (full scene context).
  2. The same image with a bounding box drawn around the object.

Supports both local Transformers inference and an OpenAI-compatible vLLM server,
delegating to GemmaVLMQuerier for all model interaction.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import math

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from dataset_generation.instance_descriptions.abs_object_describer import AbstractObjectDescriber
from dataset_generation.llm_persistence_querying.vlm_querying.gemma_vlm_querier import GemmaVLMQuerier
from instance_segmentation.abs_instance_segmenter import SegmentationResults
from utils.image_utils import image_tensor_to_pil
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


class GemmaVLMObjectDescriber(AbstractObjectDescriber):
    """
    Describes segmented objects by querying a Gemma 4 VL model with two images per object:
    the raw scene and a version with the object's bounding box highlighted.

    Constructor mirrors GemmaVLMQuerier so callers can pass the same kwargs.
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
        # --- overlay appearance ---
        mask_style: str = "blur_bg_outline_fg",
        mask_highlight_color: Tuple[int, int, int] = (0, 255, 255),
        mask_dilation_px: int = 5,
        # --- zoomed crop ---
        include_zoomed_crop: bool = False,
        # --- arrow overlay ---
        include_arrow_image: bool = False,
        arrow_color: Tuple[int, int, int] = (0, 255, 255),
        # --- bounding box overlay ---
        include_box_image: bool = False,
        box_color: Tuple[int, int, int] = (0, 255, 255),
        box_thickness: int = 5,
        box_margin: int = 3,
    ):
        if name is None:
            raise ValueError("name must be provided for GemmaVLMObjectDescriber")
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
        self._debug_dir: Optional[Path] = None
        self._describe_call_idx: int = 0

        self._querier = GemmaVLMQuerier(
            name=f"{name}__querier",
            model_name=model_name,
            use_vllm_server=use_vllm_server,
            device_map=device_map,
            torch_dtype=torch_dtype,
            vllm_url_root=vllm_url_root,
            vllm_port=vllm_port,
            vllm_url_suffix=vllm_url_suffix,
            vllm_api_key=vllm_api_key,
            vllm_request_timeout_s=vllm_request_timeout_s,
            vllm_extra_request_params=vllm_extra_request_params,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def set_debug_dir(self, debug_dir: Path) -> None:
        self._debug_dir = debug_dir

    def describeObjects(
        self,
        image: torch.Tensor,
        instances: SegmentationResults,
    ) -> Dict[int, str]:
        """
        Describe each segmented object by sending two images to the Gemma VL model:
        the raw scene and the scene with the object's bounding box drawn.

        Args:
            image: (C, H, W) torch.Tensor in [0, 1] or [0, 255].
            instances: SegmentationResults with .masks and .ids.

        Returns:
            dict mapping object id -> description string.
        """
        image_call_idx = self._describe_call_idx
        self._describe_call_idx += 1

        pil_image = image_tensor_to_pil(image)

        print(f"Input image resolution: {pil_image.width}x{pil_image.height}")
        descriptions: Dict[int, str] = {}
        total = len(instances.masks)

        for i, mask in enumerate(instances.masks):
            obj_id = int(instances.ids[i].item())
            print(f"Describing object {i + 1}/{total} (id={obj_id})...")

            display_mask = self._dilate_mask(mask)
            highlighted = self._draw_box_image(pil_image, display_mask)

            images = [pil_image, highlighted]

            description = self._querier.queryVLM(
                prompt=self.prompt,
                images=images,
            )
            self._save_debug_text(obj_id, i, image_call_idx, "", description)
            descriptions[obj_id] = description

        return descriptions

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
        """Return (x, y) for the arrow tip: median mask x at the mask's vertical center row."""
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

    def _draw_arrow_image(
        self,
        pil_image: "Image.Image",
        target_x: int,
        target_y: int,
    ) -> "Image.Image":
        """Return a copy of pil_image with an arrow pointing to (target_x, target_y)."""
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
        """Return a copy of pil_image with a rectangle drawn around the mask bounding box."""
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

    def _dilate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if self.mask_dilation_px <= 0:
            return mask
        m = mask.detach().cpu()
        if m.dim() == 3 and m.shape[0] == 1:
            m = m[0]
        dilated = binary_dilation(m.numpy().astype(bool), iterations=self.mask_dilation_px)
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
            f"[gemma_vlm_describer] debug text → {text_path}",
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
            f"[gemma_vlm_describer] debug images → {', '.join(str(p) for p in saved)}",
            file=sys.stderr, flush=True,
        )
