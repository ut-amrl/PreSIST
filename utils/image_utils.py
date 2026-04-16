
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2

# Import tuple
from typing import Tuple


def image_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert (C, H, W) tensor to PIL Image. Handles uint8 or float in [0,1]."""
    if image.ndim != 3 or image.shape[0] not in (1, 3, 4):
        raise ValueError("Expected image (C,H,W) with C in {1,3,4}.")
    img = image[:3].detach().cpu().permute(1, 2, 0).numpy()
    if img.max() <= 1.0:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def numpy_bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """(H, W, 3) BGR uint8 numpy to PIL RGB."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def loadImageAsTensor(filename):
    """Load an image from a file and convert it to a tensor."""
    image = Image.open(filename).convert("RGB")
    return image


class LazyImage:
    """Loads an image from disk on first access and caches the result.

    Pass preloaded=<tensor> when the image is already loaded,
    or leave it None to defer loading until .get() is first called.
    loader defaults to loadImageForDatasetGeneration if not provided.
    """

    def __init__(self, path: str, preloaded=None, loader=None):
        self._path = path
        self._image = preloaded
        self._loader = loader

    def get(self):
        if self._image is None:
            loader = self._loader or loadImageForDatasetGeneration
            self._image = loader(self._path)
        return self._image


def loadImageForDatasetGeneration(image_path: str):
    # ---- load image ----
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)  # (H, W, 3), uint8
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W), uint8

    return img_tensor


def load_image_as_tensor_zero_to_one(image_path: str) -> torch.Tensor:
    """
    Load an image from disk and convert to a (3, H, W) torch.FloatTensor in [0, 1].
    """
    img_pil = Image.open(image_path).convert("RGB")
    to_tensor = transforms.ToTensor()  # converts to float32 in [0,1], shape (C,H,W)
    img_tensor = to_tensor(img_pil)
    return img_tensor

# ---------- helpers to convert your tensors to PIL images ----------
def tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    """
    Convert image torch.Tensor -> PIL.Image, assuming shape (C, H, W)
    and values in [0, 1] or [0, 255].
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected image as torch.Tensor, got {type(image)}")

    if image.dim() == 3 and image.shape[0] in (1, 3):
        img = image.detach().cpu().float()
        if img.max() <= 1.0:
            img = img * 255.0
        img = img.clamp(0, 255).byte()
        img = img.permute(1, 2, 0).numpy()  # (C,H,W)->(H,W,C)

        if img.shape[2] == 1:
            img = img[:, :, 0]  # grayscale

        return Image.fromarray(img)
    else:
        raise ValueError(
            f"Expected image tensor of shape (C, H, W) with C in {{1,3}}, "
            f"got {tuple(image.shape)}"
        )


def mask_to_pil(mask: torch.Tensor) -> Image.Image:
    """
    Convert a single instance mask (H, W) or (1, H, W) tensor to a binary
    PIL.Image ('L' mode) with values {0, 255} as expected by DAM.
    The paper and code treat masks as binary {0,1} images. :contentReference[oaicite:3]{index=3}
    """
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu()
    else:
        m = torch.as_tensor(mask)

    # Accept (1, H, W) or (H, W)
    if m.dim() == 3 and m.shape[0] == 1:
        m = m[0]
    if m.dim() != 2:
        raise ValueError(
            f"Expected mask of shape (H, W) or (1, H, W), got {tuple(m.shape)}"
        )

    # Threshold to binary and scale to 0 / 255
    m = (m > 0.5).to(torch.uint8).numpy()
    m = (m * 255).astype(np.uint8)

    return Image.fromarray(m)  # "L"


# ----------------------------
# Instance-image preprocessors
# ----------------------------

def apply_mask(image: torch.Tensor, mask: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """
    image: (B, C, H, W)
    mask:  (B, 1, H, W) or (B, H, W) with {0,1} or [0,1]
    Returns masked image (B, C, H, W)
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=image.dtype)
    if fill == 0.0:
        return image * mask
    # fill background with constant
    return image * mask + fill * (1.0 - mask)


def crop_and_resize(image: torch.Tensor, mask: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Crop each image to mask bbox, then resize to out_hw (h,w).
    image: (B, C, H, W)
    mask:  (B, 1, H, W) or (B, H, W)
    """
    B, C, H, W = image.shape
    bboxes = masks_to_bboxes(mask)  # (B,4)
    out = []
    oh, ow = out_hw
    for b in range(B):
        x1, y1, x2, y2 = bboxes[b].tolist()
        patch = image[b:b+1, :, y1:y2, x1:x2]  # (1,C,hp,wp)
        patch = F.interpolate(patch, size=(oh, ow), mode="bilinear", align_corners=False)
        out.append(patch)
    return torch.cat(out, dim=0)  # (B,C,oh,ow)


def masks_to_bboxes(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: (B, 1, H, W) or (B, H, W), values >0 treated as foreground.
    Returns bboxes: (B, 4) as (x1, y1, x2, y2) in pixel coords (inclusive-exclusive).
    If a mask is empty, bbox becomes whole image.
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    B, _, H, W = mask.shape
    mask_bin = mask > 0

    bboxes = torch.zeros((B, 4), device=mask.device, dtype=torch.long)
    for b in range(B):
        ys, xs = torch.where(mask_bin[b, 0])
        if xs.numel() == 0:
            bboxes[b] = torch.tensor([0, 0, W, H], device=mask.device)
        else:
            x1, x2 = xs.min(), xs.max() + 1
            y1, y2 = ys.min(), ys.max() + 1
            bboxes[b] = torch.tensor([x1, y1, x2, y2], device=mask.device)
    return bboxes