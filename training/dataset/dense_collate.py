"""Custom collate function for the ``dense`` mode of
:class:`_SegmentationDataset`.

Stacks the dense tensors normally and concatenates per-instance tensors
in flat form ``(Σ N_i, ...)`` with a ``batch_index`` tensor. Driven by
``include_per_instance``: when False, the per-instance tensors are
dropped from the collated batch entirely so the training hot path
doesn't carry them. Stage 5 wires this flag to
``DenseLoss.requires_per_instance_tensors()``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch


def make_dense_collate(include_per_instance: bool):
    """Returns a callable suitable for ``DataLoader(collate_fn=...)``."""

    def _collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        # Required dense tensors: always present.
        out: Dict[str, Any] = {
            "image": torch.stack([s["image"] for s in batch], dim=0),
            "target_map": torch.stack([s["target_map"] for s in batch], dim=0),
            "loss_mask": torch.stack([s["loss_mask"] for s in batch], dim=0),
            "pixel_weights": torch.stack([s["pixel_weights"] for s in batch], dim=0),
            "image_ids": [s["image_id"] for s in batch],
            "image_paths": [s["image_path"] for s in batch],
        }

        if include_per_instance:
            inst_masks: List[torch.Tensor] = []
            inst_labels: List[torch.Tensor] = []
            batch_index: List[torch.Tensor] = []
            for b, s in enumerate(batch):
                m = s["instance_masks"]   # (N_b, H, W)
                l = s["instance_labels"]  # (N_b, 3)
                if m.numel() == 0:
                    continue
                inst_masks.append(m)
                inst_labels.append(l)
                batch_index.append(
                    torch.full((m.shape[0],), b, dtype=torch.int64)
                )
            if inst_masks:
                out["instance_masks"] = torch.cat(inst_masks, dim=0)
                out["instance_labels"] = torch.cat(inst_labels, dim=0)
                out["batch_index"] = torch.cat(batch_index, dim=0)
            else:
                # No instances anywhere in this batch — emit empty
                # tensors of the right rank so downstream code can do
                # uniform indexing.
                H = batch[0]["instance_masks"].shape[-2] if batch[0]["instance_masks"].ndim >= 2 else 0
                W = batch[0]["instance_masks"].shape[-1] if batch[0]["instance_masks"].ndim >= 2 else 0
                out["instance_masks"] = torch.zeros((0, H, W), dtype=torch.uint8)
                out["instance_labels"] = torch.zeros((0, 3), dtype=torch.float32)
                out["batch_index"] = torch.zeros((0,), dtype=torch.int64)

        return out

    return _collate
