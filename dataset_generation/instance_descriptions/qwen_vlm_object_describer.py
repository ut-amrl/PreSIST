"""
Object describer that queries a Qwen VL model (Qwen2-VL or Qwen3-VL).

Makes one VLM call per object with two images:
  1. The raw RGB image (full scene context).
  2. The same image with the object's mask overlaid as a colored highlight.

Supports both local Transformers inference and an OpenAI-compatible vLLM server,
delegating to QwenVLMQuerier for all model interaction.
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
from dataset_generation.intermediate_results_logger.debug_data import DebugPayload
from dataset_generation.llm_persistence_querying.vlm_querying.qwen_vlm_querier import QwenVLMQuerier
from instance_segmentation.abs_instance_segmenter import SegmentationResults
from utils.image_utils import image_tensor_to_pil
from utils.timing import querier_timer_suffix, timer
from utils.visualization_utils import mask_highlight_image


_DEFAULT_PROMPT = (
    "You are given two images of the same scene. "
    "The first is the original image. "
    "The second has one object highlighted with a colored overlay. "
    "Describe the highlighted object in detail. "
    "Comment on properties that might affect its motion or persistence in the scene over time: "
    "whether it is attached to a surface or part of the environment; "
    "if free-standing, how easy it would be for it to change position; "
    "whether external forces (wind, people, vehicles) might move or deform it; "
    "any signs of imminent motion (walking posture, wheels turned, being held, etc.)."
)



class QwenVLMObjectDescriber(AbstractObjectDescriber):
    """
    Describes segmented objects by querying a Qwen VL model with two images per object:
    the raw scene and a version with the object's mask highlighted.

    Constructor mirrors QwenVLMQuerier so callers can pass the same kwargs.
    """

    def __init__(
        self,
        *,
        name: str,
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
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
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        # --- overlay appearance ---
        mask_style: str = "blur_bg_outline_fg",
        mask_highlight_color: Tuple[int, int, int] = (0, 0, 0),
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
        box_margin: int = 10,
    ):
        if name is None:
            raise ValueError("name must be provided for QwenVLMObjectDescriber")
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
        self._prefix = "You are given an image with an arrow pointing to an object of interest. Please focus your answer on the object the arrow is pointing to, using the rest of the image only for context."
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

        self._per_query_timer_name = (
            "Timer::qwen_vlm_obj_describer_vlm_query"
            + querier_timer_suffix(use_vllm_server=use_vllm_server)
        )

        self._querier = QwenVLMQuerier(
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
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    def set_debug_dir(self, debug_dir: Path) -> None:
        self._debug_dir = debug_dir

    def describeObjects(
        self,
        image: torch.Tensor,
        instances: SegmentationResults,
    ) -> Tuple[Dict[int, str], DebugPayload]:
        """
        Describe each segmented object by sending two images to the Qwen VL model:
        the raw scene and the scene with the object's mask highlighted.

        Args:
            image: (C, H, W) torch.Tensor in [0, 1] or [0, 255].
            instances: SegmentationResults with .masks and .ids.

        Returns:
            dict mapping object id -> description string.
        """
        pil_image = image_tensor_to_pil(image)

        # Display the image resolution
        print(f"Input image resolution: {pil_image.width}x{pil_image.height}")
        descriptions: Dict[int, str] = {}
        debug_payload = DebugPayload()
        total = len(instances.masks)

        for i, mask in enumerate(instances.masks):
            obj_id = int(instances.ids[i].item())
            print(f"Describing object {i + 1}/{total} (id={obj_id})...")

            display_mask = self._dilate_mask(mask)
            highlighted = Image.fromarray(
                mask_highlight_image(
                    image, display_mask,
                    style=self.mask_style,
                    mask_highlight_color=self.mask_highlight_color,
                )
            )

            images = [pil_image, highlighted]
            zoomed: Optional[Image.Image] = None
            if self.include_zoomed_crop:
                crop_box = self._compute_zoom_crop(mask, pil_image.width, pil_image.height)
                if crop_box is not None:
                    zoomed = pil_image.crop(crop_box)
                    images.append(zoomed)

            arrow_img: Optional[Image.Image] = None
            if self.include_arrow_image:
                target = self._compute_arrow_target(mask)
                if target is not None:
                    arrow_img = self._draw_arrow_image(pil_image, *target)
                    print("Drawing arrow and saving it to images 0")
                    images[0] = arrow_img

            box_img: Optional[Image.Image] = None
            if self.include_box_image:
                box_img = self._draw_box_image(pil_image, mask)
                if box_img is not None:
                    images[0] = box_img

            # self._save_debug_images(pil_image, highlighted, obj_id, i, zoomed, arrow_img)
            self._save_debug_images(images[0], images[0],
                                    obj_id, i, zoomed, arrow_img)

            query_images = [images[0]]
            for q_idx, q_img in enumerate(query_images):
                if q_img is None:
                    continue
                debug_payload.add(
                    f"query_img_{obj_id}_qIm{q_idx:02d}",
                    q_img,
                    kind="image",
                )

            with timer.time_block(self._per_query_timer_name):
                description, sub_debug = self._querier.queryVLM(
                    prompt=self.prompt,
                    images=query_images,
                )
            debug_payload.merge(sub_debug)
            descriptions[obj_id] = description

        return descriptions, debug_payload

    def _compute_zoom_crop(
        self, mask: torch.Tensor, img_w: int, img_h: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Return (left, top, right, bottom) crop box centered on the mask.

        The crop has the same aspect ratio as the original image and is the
        smallest such crop where both dimensions are >= 3x the mask's extent.
        Returns None if the mask is empty.
        """
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

        # Smallest scale (fraction of full image) satisfying both 3x constraints.
        scale = min(max(3 * mask_w / img_w, 3 * mask_h / img_h), 1.0)
        crop_w = int(np.ceil(scale * img_w))
        crop_h = int(np.ceil(scale * img_h))

        # Clip symmetrically around the object center so both sides have equal
        # margins even when the 3x crop would exceed the image boundary.
        # Aspect ratio is not preserved in that case.
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

        # Search outward from center_row until we find a row with mask pixels.
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
        """Return a copy of pil_image with an arrow pointing to (target_x, target_y).

        The arrow is drawn at a 45-degree diagonal whose tail is on the
        image-center side of the target point.
        """
        img = pil_image.copy()
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size

        # Pick the 45-degree diagonal most aligned with center → target.
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

        # Filled arrowhead triangle.
        head = lw * 4
        px, py = -arrow_dy, arrow_dx  # perpendicular
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
        """Return a copy of pil_image with a rectangle drawn around the mask bounding box.

        The rectangle is expanded by box_margin pixels on each side, clipped to the image.
        Returns None if the mask is empty.
        """
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

    def _save_debug_images(
        self,
        raw: "Image.Image",
        highlighted: "Image.Image",
        obj_id: int,
        idx: int,
        zoomed: Optional["Image.Image"] = None,
        arrow_img: Optional["Image.Image"] = None,
    ) -> None:
        if self._debug_dir is None:
            return
        self._debug_dir.mkdir(parents=True, exist_ok=True)
        raw_path = self._debug_dir / f"obj{idx:03d}_id{obj_id}_raw.jpg"
        hl_path  = self._debug_dir / f"obj{idx:03d}_id{obj_id}_highlighted.jpg"
        raw.save(raw_path, format="JPEG", quality=90)
        highlighted.save(hl_path, format="JPEG", quality=90)
        saved = [raw_path, hl_path]
        if zoomed is not None:
            zoom_path = self._debug_dir / f"obj{idx:03d}_id{obj_id}_zoomed.jpg"
            zoomed.save(zoom_path, format="JPEG", quality=90)
            saved.append(zoom_path)
        if arrow_img is not None:
            arrow_path = self._debug_dir / f"obj{idx:03d}_id{obj_id}_arrow.jpg"
            arrow_img.save(arrow_path, format="JPEG", quality=90)
            saved.append(arrow_path)
        print(
            f"[qwen_vlm_describer] debug images → {', '.join(str(p) for p in saved)}",
            file=sys.stderr, flush=True,
        )
