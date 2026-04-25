from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

from utils.timing import timer
from instance_segmentation.abs_instance_segmenter import SegmentationResults
from dataset_generation.scene_level_captioning.abs_scene_level_captioner import AbstractSceneLevelCaptioner
from dataset_generation.instance_descriptions.abs_object_describer import AbstractObjectDescriber
from dataset_generation.llm_persistence_querying.abs_llm_persistence_querier import AbsLLMPersistenceQuerier

from persistence_eval.core.domain import ImageMaskGroup, PersistenceParameterEstimate
from persistence_eval.methods.inferencers.base import AbsPersistenceParameterInferencer


def _masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Compute tight XYXY bounding boxes from a (K, H, W) bool tensor."""
    boxes = []
    for mask in masks:  # iterate K dimension
        nonzero = mask.nonzero(as_tuple=False)  # (N, 2) rows=[row, col]
        if nonzero.numel() == 0:
            boxes.append([0.0, 0.0, 0.0, 0.0])
        else:
            r_min, c_min = nonzero[:, 0].min().item(), nonzero[:, 1].min().item()
            r_max, c_max = nonzero[:, 0].max().item(), nonzero[:, 1].max().item()
            boxes.append([float(c_min), float(r_min), float(c_max), float(r_max)])
    return torch.tensor(boxes, dtype=torch.float32)


class LLMPipelineInferencer(AbsPersistenceParameterInferencer):
    """
    Persistence parameter inferencer that mirrors the dataset-generation pipeline
    (scene captioning → object description → LLM/VLM querying) without segmentation.

    The anchor masks are used directly as instances passed to the describer and querier,
    so no separate segmentation model is needed at inference time.

    The scene is captioned once per group and all masks described together, matching
    the efficiency of the dataset-generation batch path. group.masks is (K, H, W) bool.
    """

    def __init__(
        self,
        sceneCaptioner: AbstractSceneLevelCaptioner,
        objectDescriber: AbstractObjectDescriber,
        llmPersistenceQuerier: AbsLLMPersistenceQuerier,
        param_keys: list[str] | None = None,
        debug_log: bool = False,
        debug_visualizations: bool = True,
    ) -> None:
        self._captioner = sceneCaptioner
        self._describer = objectDescriber
        self._querier = llmPersistenceQuerier
        self._param_keys = param_keys
        self._debug_log = debug_log
        self._debug_visualizations = debug_visualizations
        self._debug_output_dir: Path | None = None
        self._captioner_warmed_up = False
        self._describer_warmed_up = False
        self._querier_warmed_up = False

    def set_run_dir(self, run_dir: Path) -> None:
        if self._debug_visualizations:
            self._debug_output_dir = run_dir / "debug_vis"
            if hasattr(self._describer, "set_debug_dir"):
                self._describer.set_debug_dir(self._debug_output_dir / "object_describer")

    def infer(self, group: ImageMaskGroup) -> list[PersistenceParameterEstimate]:
        # print(
        #     f"[llm_pipeline] image={group.image_path or 'unknown'}  tracks={group.track_ids}",
        #     file=sys.stderr, flush=True,
        # )
        self._save_visualization(group)
        instances = self._build_segmentation_results(group.masks, group.image.shape[-2:])



        describer_ctx = timer.time_block(f"Timer::obj_desc::{type(self._describer).__name__}") if self._describer_warmed_up else nullcontext()
        with describer_ctx:
            object_descriptions = self._describer.describeObjects(group.image, instances)
        self._describer_warmed_up = True

        # self._maybe_debug(group, "", object_descriptions, "", "")
        # return []  # TEMP: skip querying while debugging describer output

        captioner_ctx = timer.time_block(f"Timer::scene_cap::{type(self._captioner).__name__}") if self._captioner_warmed_up else nullcontext()
        with captioner_ctx:
            scene_caption = self._captioner.generateCaption(group.image)
        self._captioner_warmed_up = True

        # print("Object descriptions:", object_descriptions)



        querier_ctx = timer.time_block(f"Timer::llm_querier::{type(self._querier).__name__}") if self._querier_warmed_up else nullcontext()
        raw_estimates = None
        for attempt in range(3):
            try:
                with querier_ctx:
                    _, _, raw_estimates = self._querier.estimatePersistenceParameters(
                        scene_caption,
                        object_descriptions,
                        {},    # relationship descriptions — always empty at inference time
                        None,  # time_and_date
                        group.image,
                        instances,
                    )
                break
            except ValueError as e:
                if attempt < 2:
                    print(f"[llm_pipeline] querier failed (attempt {attempt + 1}/3), retrying: {e}", file=sys.stderr, flush=True)
                else:
                    print(f"[llm_pipeline] querier failed after 3 attempts, skipping group: {e}", file=sys.stderr, flush=True)
        self._querier_warmed_up = True

        if raw_estimates is None:
            return [PersistenceParameterEstimate(params={}, metadata={"track_id": tid, "raw_estimate": {}}) for tid in group.track_ids]

        estimates = self._map_to_persistence_estimates(raw_estimates, group.track_ids)
        self._maybe_debug(group, scene_caption, object_descriptions, raw_estimates, estimates)
        # exit(1)
        return estimates

    # TODO: temporary visualization — remove when done debugging
    def _save_visualization(self, group: ImageMaskGroup) -> None:
        import tempfile

        import matplotlib.pyplot as plt
        import numpy as np

        image_np = group.image.permute(1, 2, 0).numpy()  # CHW → HWC
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image_np)

        cyan = (0.0, 1.0, 1.0)
        for mask, track_id in zip(group.masks, group.track_ids):
            mask_np = mask.numpy()
            overlay = np.zeros((*mask_np.shape, 4), dtype=np.float32)
            overlay[mask_np] = [*cyan, 0.3]
            ax.imshow(overlay)
            # outline via erosion difference
            from scipy.ndimage import binary_erosion  # noqa: PLC0415
            outline = mask_np & ~binary_erosion(mask_np)
            outline_rgba = np.zeros((*mask_np.shape, 4), dtype=np.float32)
            outline_rgba[outline] = [*cyan, 0.8]
            ax.imshow(outline_rgba)

        track_str = "  |  ".join(group.track_ids)
        image_str = group.image_path or f"anchor_time={group.anchor_time:.3f}s"
        ax.set_title(f"{image_str}\ntracks: {track_str}", fontsize=9)
        ax.axis("off")

        if self._debug_output_dir is not None:
            self._debug_output_dir.mkdir(parents=True, exist_ok=True)
            out = self._debug_output_dir / f"vis_{group.anchor_time:.6f}.png"
        else:
            out = Path(tempfile.mktemp(suffix=".png"))

        fig.savefig(out, bbox_inches="tight", dpi=120)
        plt.close(fig)
        # print(f"[llm_pipeline] visualization → {out}", file=sys.stderr, flush=True)

    def _maybe_debug(
        self,
        group: ImageMaskGroup,
        scene_caption: str,
        object_descriptions: dict,
        raw_estimates: list[dict],
        estimates: list[PersistenceParameterEstimate],
    ) -> None:
        if not self._debug_log and self._debug_output_dir is None:
            return

        # Build per-object mapping: track_id → {segmentation_id, description, raw_estimate, params}
        n = len(group.track_ids)
        if raw_estimates and "id" in raw_estimates[0]:
            by_id: dict[int, dict] = {int(e["id"]): e for e in raw_estimates}
            aligned_raw = [by_id.get(i, {}) for i in range(n)]
        else:
            aligned_raw = list(raw_estimates) + [{}] * max(0, n - len(raw_estimates))

        per_object: dict[str, dict] = {}
        for i, track_id in enumerate(group.track_ids):
            seg_id = i
            per_object[track_id] = {
                "segmentation_id": seg_id,
                "description": object_descriptions.get(seg_id, ""),
                "raw_estimate": aligned_raw[i] if i < len(aligned_raw) else {},
                "params": estimates[i].params if i < len(estimates) else {},
            }

        record = {
            "image_path": group.image_path,
            "anchor_time": group.anchor_time,
            "track_ids": list(group.track_ids),
            "scene_caption": scene_caption,
            "object_descriptions": {str(k): v for k, v in object_descriptions.items()},
            "raw_estimates": list(raw_estimates),
            "per_object": per_object,
        }

        if self._debug_log:
            print(json.dumps(record, indent=2, default=str), file=sys.stderr, flush=True)

        if self._debug_output_dir is not None:
            self._debug_output_dir.mkdir(parents=True, exist_ok=True)
            fname = self._debug_output_dir / f"debug_{group.anchor_time:.6f}.json"
            fname.write_text(json.dumps(record, indent=2, default=str))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_segmentation_results(
        masks: torch.Tensor, hw: tuple[int, int]
    ) -> SegmentationResults:
        h, w = hw
        if masks.shape[0] == 0:
            return SegmentationResults.empty(h=h, w=w)
        return SegmentationResults(
            ids=torch.arange(masks.shape[0], dtype=torch.int64),
            masks=masks.bool(),
            boxes=_masks_to_boxes(masks),
            scores=torch.ones(masks.shape[0], dtype=torch.float32),
        )

    def _map_to_persistence_estimates(
        self,
        raw_estimates: list[dict],
        track_ids: list[str],
    ) -> list[PersistenceParameterEstimate]:
        """
        Align querier output to one estimate per input mask.

        VLM path (ChainedVLMPersistenceQuerier): each dict carries an "id" key equal
        to the SegmentationResults id (0..N-1), so we match by id and tolerate gaps
        caused by per-object VLM failures.

        LLM path (ChainedPromptQueryPersistenceQuerier): dicts are in order but may
        also carry "id" fields parsed from the response; fall back to positional
        alignment when no "id" is present.
        """
        n = len(track_ids)

        if raw_estimates and "id" in raw_estimates[0]:
            by_id: dict[int, dict] = {int(e["id"]): e for e in raw_estimates}
            aligned = [by_id.get(i, {}) for i in range(n)]
        else:
            aligned = list(raw_estimates) + [{}] * max(0, n - len(raw_estimates))

        return [
            PersistenceParameterEstimate(
                params=self._extract_params(raw),
                metadata={"track_id": track_ids[i], "raw_estimate": raw},
            )
            for i, raw in enumerate(aligned[:n])
        ]

    def _extract_params(self, raw: dict) -> dict[str, float]:
        """Extract float-valued entries, optionally filtered to self._param_keys."""
        extracted = {
            k: float(v)
            for k, v in raw.items()
            if k != "id" and isinstance(v, (int, float))
        }
        if self._param_keys is not None:
            extracted = {k: v for k, v in extracted.items() if k in self._param_keys}
        return extracted
