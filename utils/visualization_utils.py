import torch
import numpy as np
import cv2

import os
from PIL import Image

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


def saveMasksOnImages(masks, image, outputDir, overlayColor=(0, 0, 255), alphaMask=0.4, alphaImg=0.6, overwriteExisting=True):
    # Save each mask overlaid on the original image
    for idx, mask in enumerate(masks):

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