"""
DirectVLMQueryInferencer: caches the per-instance image and mask references so
that DirectVLMPredictor can issue a VLM query at evaluation time using the
actual ``query_times`` for each episode.

When the dataset surfaces on-disk mask paths via ``ImageMaskGroup.mask_paths``
the inferencer just records them — no extra IO. For datasets that synthesise
masks at iter time (no file backing) the inferencer falls back to writing each
mask as a compressed npz under ``<run_dir>/vlm_context/``.

This inferencer does NOT call any captioning or description model — the paired
querier (``AbsDirectFutureExistenceVLMQuerier``) is responsible for whatever
visual or textual context the VLM needs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from persistence_eval.core.domain import ImageMaskGroup, PersistenceParameterEstimate
from persistence_eval.methods.inferencers.base import AbsPersistenceParameterInferencer


_VLM_CONTEXT_SUBDIR = "vlm_context"


class DirectVLMQueryInferencer(AbsPersistenceParameterInferencer):
    """
    For each track, records ``image_path`` + ``mask_path`` in
    ``estimate.metadata["vlm_context"]``. ``DirectVLMPredictor`` reloads them
    at eval time and hands them to the querier.
    """

    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self._context_dir: Path | None = None

    def set_run_dir(self, run_dir: Path) -> None:
        # Only created lazily inside _save_mask, so datasets that already
        # surface mask paths never touch the filesystem here.
        self._context_dir = Path(run_dir) / _VLM_CONTEXT_SUBDIR

    def infer(self, group: ImageMaskGroup) -> list[PersistenceParameterEstimate]:
        if group.image_path is None:
            raise ValueError(
                "DirectVLMQueryInferencer requires ImageMaskGroup.image_path so the "
                "predictor can re-load the image at eval time."
            )
        if group.mask_paths is not None and len(group.mask_paths) != len(group.track_ids):
            raise ValueError(
                f"ImageMaskGroup.mask_paths length ({len(group.mask_paths)}) does not "
                f"match track_ids length ({len(group.track_ids)})."
            )

        estimates: list[PersistenceParameterEstimate] = []
        for i, track_id in enumerate(group.track_ids):
            mask_path = (
                group.mask_paths[i]
                if group.mask_paths is not None
                else str(self._save_mask(group.anchor_time, i, group.masks[i]))
            )
            estimates.append(
                PersistenceParameterEstimate(
                    params={},
                    metadata={
                        "track_id": track_id,
                        "vlm_context": {
                            "image_path": group.image_path,
                            "mask_path": mask_path,
                        },
                    },
                )
            )
        return estimates

    def _save_mask(self, anchor_time: float, idx: int, mask: torch.Tensor) -> Path:
        if self._context_dir is None:
            raise RuntimeError(
                "DirectVLMQueryInferencer.set_run_dir() must be called before infer() "
                "for datasets that don't surface mask paths via ImageMaskGroup.mask_paths."
            )
        self._context_dir.mkdir(parents=True, exist_ok=True)
        out = self._context_dir / f"mask_{anchor_time:.6f}_{idx}.npz"
        np.savez_compressed(out, mask=mask.cpu().numpy().astype(np.bool_))
        return out.resolve()
