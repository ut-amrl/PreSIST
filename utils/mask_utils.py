from __future__ import annotations

from typing import Optional

import torch


def mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    intersection = torch.logical_and(mask_a, mask_b).sum().item()
    union = torch.logical_or(mask_a, mask_b).sum().item()
    if union == 0:
        return 0.0
    return float(intersection / union)


def filter_mask_indices_by_min_area(
    masks: torch.Tensor,
    min_area_pixels: int,
) -> list[int]:
    if masks.shape[0] == 0 or min_area_pixels <= 0:
        return list(range(masks.shape[0]))

    areas = masks.to(dtype=torch.int64).sum(dim=(1, 2))
    keep = (areas >= int(min_area_pixels)).nonzero(as_tuple=True)[0]
    return keep.tolist()


def dedupe_mask_indices_by_iou(
    masks: torch.Tensor,
    scores: Optional[torch.Tensor],
    iou_threshold: float,
) -> list[int]:
    if masks.shape[0] <= 1 or iou_threshold > 1.0:
        return list(range(masks.shape[0]))

    if scores is None:
        ordered_indices = list(range(masks.shape[0]))
    else:
        ordered_indices = sorted(
            range(masks.shape[0]),
            key=lambda idx: (-float(scores[idx].item()), idx),
        )

    kept = []
    for idx in ordered_indices:
        should_drop = False
        for kept_idx in kept:
            if mask_iou(masks[idx], masks[kept_idx]) > iou_threshold:
                should_drop = True
                break
        if not should_drop:
            kept.append(idx)

    kept.sort()
    return kept
