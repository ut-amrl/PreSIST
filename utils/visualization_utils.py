import torch
import numpy as np
import cv2

import os
from PIL import Image, ImageFilter

from typing import Tuple, Optional, Type


def overlay_single_mask(
    image_t: torch.Tensor,
    mask_t: torch.Tensor,
    alpha_mask: float = 0.4,
    alpha_img: float = 0.6,
    color_bgr=(0, 0, 255),
) -> np.ndarray:
    """
    Overlay a single mask on an image.

    image_t:
        (3, H, W) or (H, W, 3) torch.Tensor, values in [0,1] or [0,255]
    mask_t:
        (H, W) bool / 0-1 tensor

    Returns:
        (H, W, 3) uint8 image with the mask overlaid in the given color.
    """
    # ---- prep image ----
    if image_t.dim() == 3 and image_t.shape[0] in (1, 3):  # (C,H,W) -> (H,W,C)
        img_np = image_t.detach().cpu().permute(1, 2, 0).numpy()
    else:
        img_np = image_t.detach().cpu().numpy()

    # scale to 0–255 if needed
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)

    # ensure 3 channels
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 1:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    H, W, _ = img_np.shape
    out = img_np.copy()

    # ---- prep mask ----
    mask_np = mask_t.detach().cpu().numpy()
    if mask_np.dtype != bool:
        mask_np = mask_np > 0.5
    assert mask_np.shape == (H, W), f"Mask has shape {mask_np.shape}, expected {(H, W)}"

    # colored mask
    colored_mask = np.zeros_like(out, dtype=np.uint8)
    colored_mask[mask_np] = np.array(color_bgr, dtype=np.uint8)

    # blend
    blended = cv2.addWeighted(colored_mask, alpha_mask, out, alpha_img, 0.0)
    out[mask_np] = blended[mask_np]

    return out


def blur_bkg_outline_fg(
    original_image,
    obj_mask,
    blur_radius: int = 8,
    background_dim_factor: float = 0.35,
    highlight_color=(255, 255, 0),
    highlight_alpha: float = 0.0,
    outline_color=(0, 255, 0),
    outline_alpha: float = 0.9,
    outline_thickness: int = 1,
) -> np.ndarray:
    """
    Create a 'visual prompt' image that keeps the full scene but
    highlights the object defined by obj_mask:

      - background is blurred + dimmed
      - object region is kept sharp and slightly tinted
      - an outline is drawn around the object

    Supports:
      - original_image: PIL.Image.Image, np.ndarray (H,W,3/4) or (3,H,W),
                        or torch.Tensor with shape (3,H,W) or (H,W,3/4)
      - obj_mask: np.ndarray or torch.Tensor with shape (H,W) or (1,H,W)
    """

    # --- 1. Normalize original_image to RGB uint8 numpy array in HWC ---
    img_np = _to_rgb_hwc_np(original_image)  # (H, W, 3) float32
    H, W, _ = img_np.shape
    img_pil = Image.fromarray(img_np.astype(np.uint8), mode="RGB")

    # --- 2. Normalize mask to boolean shape (H, W) ---
    mask_bool = _to_bool_mask_hw(obj_mask, H, W)

    # --- 3. Create blurred + dimmed background ---
    if (not blur_radius or blur_radius <= 0):
        # Skip blurring if radius is 0 or negative, just use the original image as background
        blurred_np = img_np.astype(np.float32)
    else:
        blurred_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blurred_np = np.asarray(blurred_pil).astype(np.float32)

    # Start from blurred as background composite
    comp = blurred_np.copy()

    # Dim background further (only where mask is False)
    comp[~mask_bool] *= background_dim_factor

    # Overwrite object region with original (sharp) image
    comp[mask_bool] = img_np[mask_bool]

    # --- 4. Softly tint the object region with highlight_color ---
    highlight_color_arr = np.array(highlight_color, dtype=np.float32)
    obj_region = comp[mask_bool]
    comp[mask_bool] = (
        (1.0 - highlight_alpha) * obj_region +
        highlight_alpha * highlight_color_arr
    )

    # --- 5. Build a simple outline from the mask via dilation ---
    if outline_thickness > 0:
        t = int(outline_thickness)
        k = 2 * t + 1
        mask_pad = np.pad(mask_bool, t, mode="constant", constant_values=False)
        dilated = np.zeros_like(mask_bool)
        for dy in range(k):
            for dx in range(k):
                dilated |= mask_pad[dy:dy + H, dx:dx + W]

        outline = dilated & ~mask_bool  # ring around the object

        # Where the mask touches the image edge, the outward ring gets clipped.
        # Add an inward strip of thickness ceil(t/2) along the image borders,
        # restricted to mask pixels, so the outline is still visible on that side.
        t_inner = (t + 1) // 2
        edge_strip = np.zeros_like(mask_bool)
        edge_strip[:t_inner, :] = True
        edge_strip[-t_inner:, :] = True
        edge_strip[:, :t_inner] = True
        edge_strip[:, -t_inner:] = True
        outline |= mask_bool & edge_strip

        # Blend outline color into boundary pixels
        outline_color_arr = np.array(outline_color, dtype=np.float32)
        out_region = comp[outline]
        comp[outline] = (
            (1.0 - outline_alpha) * out_region +
            outline_alpha * outline_color_arr
        )

    # --- 6. Clip and convert back to uint8 numpy array ---
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    return comp


def _to_rgb_hwc_np(image) -> np.ndarray:
    """
    Convert various image formats to a float32 RGB numpy array (H, W, 3).
    Handles:
      - PIL.Image.Image
      - np.ndarray: (H,W), (H,W,3/4), (3,H,W)
      - torch.Tensor: (H,W), (H,W,3/4), (3,H,W)
    """
    try:
        import torch
        is_torch = isinstance(image, torch.Tensor)
    except ImportError:
        is_torch = False

    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        return arr

    if is_torch:
        img = image.detach().cpu()
        if img.ndim == 3:
            # (C,H,W) or (H,W,C)
            if img.shape[0] in (1, 3, 4) and img.shape[0] <= img.shape[-1]:
                # assume (C,H,W)
                img = img.permute(1, 2, 0)  # -> (H,W,C)
        arr = img.numpy()
    else:
        arr = np.asarray(image)

    # Now arr is numpy, but shape can be (H,W), (H,W,C) or (C,H,W)
    if arr.ndim == 2:
        # grayscale -> RGB
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[0] <= arr.shape[-1]:
            # Likely (C,H,W)
            arr = np.transpose(arr, (1, 2, 0))  # -> (H,W,C)

    if arr.shape[-1] == 4:
        arr = arr[..., :3]  # drop alpha

    arr = arr.astype(np.float32)
    return arr


def _to_bool_mask_hw(mask, H: int, W: int) -> np.ndarray:
    """
    Convert a mask to a boolean numpy array of shape (H, W).
    Supports np.ndarray or torch.Tensor with shape (H,W) or (1,H,W).
    """
    try:
        import torch
        is_torch = isinstance(mask, torch.Tensor)
    except ImportError:
        is_torch = False

    if is_torch:
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)

    if mask_np.ndim == 3:
        # (1,H,W) or (C,H,W) -> take first channel
        mask_np = mask_np[0]

    if mask_np.shape != (H, W):
        raise ValueError(
            f"Mask shape {mask_np.shape} does not match image shape {(H, W)}"
        )

    return mask_np.astype(bool)

def mask_highlight_image(
    image_t: torch.Tensor,
    mask_t: torch.Tensor,
    style: str = "mask_overlay",
    mask_highlight_color: Tuple[int, int, int] = (0, 0, 255),
    alpha_mask: float = 0.6,
    alpha_img: float = 0.4,
    # Add additional kwargs for other styles as needed
    extra_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """
    Create an image that highlights a single mask for VLM input.

    Args:
        image_t: (3, H, W) torch.Tensor, values in [0,1] or [0,255]
        mask_t: (H, W) bool or 0-1 tensor
        style: "mask_overlay" | "foreground_only" | "binary_mask"
            - "mask_overlay": strong color overlay on mask
            - "foreground_only": background black/dark, mask region unchanged
            - "binary_mask": background black, mask region white (silhouette)
            - "blur_bg_outline_fg": blur+dim background, keep foreground sharp, add outline
        mask_highlight_color: BGR tuple for highlight (used when style="mask_overlay" or "blur_bg_outline_fg")
        alpha_mask, alpha_img: blend factors for mask_overlay

    Returns:
        (H, W, 3) uint8 numpy array.
    """

    if (extra_kwargs is None):
        extra_kwargs = {}
    if (style == "blur_bg_outline_fg"):
        return blur_bkg_outline_fg(
            image_t,
            mask_t,
            blur_radius=extra_kwargs.get("blur_radius", 8),
            background_dim_factor=extra_kwargs.get("background_dim_factor", 0.35),
            highlight_color=mask_highlight_color,
            highlight_alpha=extra_kwargs.get("highlight_alpha", 0.0),
            outline_color=mask_highlight_color,
            outline_alpha=extra_kwargs.get("outline_alpha", 0.9),
            outline_thickness=extra_kwargs.get("outline_thickness", 1),
        )

    if image_t.dim() == 3 and image_t.shape[0] in (1, 3):
        img_np = image_t.detach().cpu().permute(1, 2, 0).numpy()
    else:
        img_np = image_t.detach().cpu().numpy()

    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)

    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 1:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    H, W, _ = img_np.shape
    mask_np = mask_t.detach().cpu().numpy()
    if mask_np.dtype != bool:
        mask_np = mask_np > 0.5
    assert mask_np.shape == (H, W), f"Mask shape {mask_np.shape} vs image {(H, W)}"

    if style in ("foreground_only", "binary_mask"):
        # Set background to black
        out = img_np.copy()
        out[~mask_np] = 0
        if style == "binary_mask":
            # Set foreground to white
            out[mask_np] = 255

        return out

    if style == "mask_overlay":
        return overlay_single_mask(
            image_t, mask_t, alpha_mask=alpha_mask, alpha_img=alpha_img, color_bgr=mask_highlight_color
        )

    raise ValueError(
        f"Unknown style {style!r}. Use 'mask_overlay', 'foreground_only', 'binary_mask', or 'blur_bg_outline_fg'."
    )


def saveMasksOnImages(masks, image, outputDir, overlayColor=(0, 0, 255), alphaMask=0.4, alphaImg=0.6, overwriteExisting=True, scores=None):
    # Save each mask overlaid on the original image
    for idx, mask in enumerate(masks):

        # Print the confidence score for the mask if it exists
        if (scores is not None) and (idx < len(scores)):
            print(f"Mask {idx} confidence score: {scores[idx]:.4f}")

        overlay_path = os.path.join(outputDir, f"mask_{idx:03d}.png")
        if (not overwriteExisting) and os.path.exists(overlay_path):
            continue
        overlay_np = overlay_single_mask(
            image,
            mask,
            alpha_mask=alphaMask,
            alpha_img=alphaImg,
            color_bgr=overlayColor,
        )
        overlay_pil = Image.fromarray(overlay_np)
        overlay_pil.save(overlay_path)