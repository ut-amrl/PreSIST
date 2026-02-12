# labeling/image_source.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

def tensor_to_pil(img: Any) -> Image.Image:
    """
    Supports:
      - torch Tensor CHW or HWC
      - numpy array
      - PIL
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, np.ndarray):
        arr = img
    elif hasattr(img, "detach") and hasattr(img, "cpu") and hasattr(img, "numpy"):
        arr = img.detach().cpu().numpy()
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    # CHW -> HWC if needed
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    # If float in [0,1] or [0,255], normalize to uint8
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    return Image.fromarray(arr).convert("RGB")


@dataclass
class DatasetItem:
    image_id: str
    image_pil: Image.Image
    path: str
    relpath: str
    component: str
    index: int

class DatasetImageSource:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def get_id(self, idx: int) -> str:
        meta = self.dataset.get_metadata(idx)
        image_id = meta.get("image_id")
        if image_id is None:
            # You may choose to fallback to relpath instead
            raise ValueError(f"Metadata for idx={idx} has no image_id.")
        return str(image_id)

    def get_relname(self, idx: int) -> str:
        meta = self.dataset.get_metadata(idx)
        return meta.get("relpath") or meta.get("path") or meta.get("image_id") or idx

    def get(self, idx: int) -> DatasetItem:
        rec = self.dataset[idx]     # full load (slow)
        return DatasetItem(
            image_id=str(rec["image_id"]),
            image_pil=tensor_to_pil(rec["image"]),
            path=str(rec["path"]),
            relpath=str(rec["relpath"]),
            component=str(rec["component"]),
            index=int(rec["index"]),
        )


# class DatasetImageSource:
#     """
#     Random-access wrapper around your RegistryBackedImageDataset.
#     """
#     def __init__(self, dataset):
#         self.dataset = dataset

#     def __len__(self) -> int:
#         return len(self.dataset)

#     def get(self, idx: int) -> DatasetItem:
#         rec: Dict[str, Any] = self.dataset[idx]
#         image_id = rec.get("image_id", None)
#         if image_id is None:
#             raise ValueError(
#                 f"dataset[{idx}] has image_id=None. "
#                 "Labeling requires stable image_id to locate segmentation results."
#             )
#         return DatasetItem(
#             image_id=str(image_id),
#             image_pil=tensor_to_pil(rec["image"]),
#             path=str(rec.get("path", "")),
#             relpath=str(rec.get("relpath", "")),
#             component=str(rec.get("component", "")),
#             index=int(rec.get("index", idx)),
#         )


