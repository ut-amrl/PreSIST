"""Shared mask→SegmentationResults helpers for VLM/LLM pipeline inferencers."""
from __future__ import annotations

import torch

from instance_segmentation.abs_instance_segmenter import SegmentationResults


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Compute tight XYXY bounding boxes from a (K, H, W) bool tensor."""
    boxes = []
    for mask in masks:
        nonzero = mask.nonzero(as_tuple=False)
        if nonzero.numel() == 0:
            boxes.append([0.0, 0.0, 0.0, 0.0])
        else:
            r_min, c_min = nonzero[:, 0].min().item(), nonzero[:, 1].min().item()
            r_max, c_max = nonzero[:, 0].max().item(), nonzero[:, 1].max().item()
            boxes.append([float(c_min), float(r_min), float(c_max), float(r_max)])
    return torch.tensor(boxes, dtype=torch.float32)


def build_segmentation_results(
    masks: torch.Tensor, hw: tuple[int, int]
) -> SegmentationResults:
    h, w = hw
    if masks.shape[0] == 0:
        return SegmentationResults.empty(h=h, w=w)
    return SegmentationResults(
        ids=torch.arange(masks.shape[0], dtype=torch.int64),
        masks=masks.bool(),
        boxes=masks_to_boxes(masks),
        scores=torch.ones(masks.shape[0], dtype=torch.float32),
    )
