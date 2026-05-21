"""Shared mask decoder for annotation-tool datasets.

Accepts PNG/JPG (anything PIL can open in "L" mode), `.npy`, and `.npz`
(reads the ``"mask"`` array). Always returns a 2-D boolean tensor where
nonzero ⇒ mask.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _to_bool_2d(arr: np.ndarray, path: str) -> torch.Tensor:
    if arr.ndim != 2:
        raise ValueError(
            f"mask at {path!r} must be 2-D; got shape {arr.shape}"
        )
    return torch.from_numpy(np.asarray(arr) > 0)


def load_mask_bool(path: str | Path) -> torch.Tensor:
    """Load a binary mask from disk and return a 2-D ``bool`` tensor.

    Supports:
      - ``.npy``: a single 2-D array.
      - ``.npz``: archive with a ``"mask"`` entry holding a 2-D array.
      - anything else: opened with PIL, converted to L, thresholded > 0.
    """
    p = str(path)
    lower = p.lower()
    if lower.endswith(".npy"):
        arr = np.load(p)
        return _to_bool_2d(arr, p)
    if lower.endswith(".npz"):
        with np.load(p) as data:
            arr = data["mask"]
        return _to_bool_2d(arr, p)
    with Image.open(p) as im:
        arr = np.array(im.convert("L"))
    return _to_bool_2d(arr, p)
