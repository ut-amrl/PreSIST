"""Persistence inferencer that asks a VLM for the persistence parameters
*directly* from per-object images, skipping the scene-caption and
object-description stages used by ``LLMPipelineInferencer``.

Reuses ``ChainedVLMPersistenceQuerier`` (which already runs one VLM call per
object and parses strict JSON) together with a prompt handler that ignores
the caption/description args.
"""
from __future__ import annotations

import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

from utils.timing import timer
from dataset_generation.llm_persistence_querying.abs_llm_persistence_querier import AbsLLMPersistenceQuerier

from persistence_eval.core.domain import ImageMaskGroup, PersistenceParameterEstimate
from persistence_eval.methods.inferencers.base import AbsPersistenceParameterInferencer
from persistence_eval.methods.inferencers._seg_helpers import build_segmentation_results
from persistence_eval.methods.inferencers._pipeline_helpers import (
    map_to_persistence_estimates,
    save_mask_visualization,
    write_debug_record,
)


class DirectVLMPipelineInferencer(AbsPersistenceParameterInferencer):
    """One VLM call per object that returns strict-JSON persistence params.

    The querier passes empty caption and object-description dicts to the
    prompt handler; the prompt handler is expected to ignore them.
    """

    def __init__(
        self,
        llmPersistenceQuerier: AbsLLMPersistenceQuerier,
        param_keys: list[str] | None = None,
        debug_log: bool = False,
        debug_visualizations: bool = False,
        strict_object_coverage: bool = False,
    ) -> None:
        self._querier = llmPersistenceQuerier
        self._param_keys = param_keys
        self._debug_log = bool(debug_log)
        self._debug_visualizations = bool(debug_visualizations)
        self._strict_object_coverage = bool(strict_object_coverage)
        self._debug_output_dir: Path | None = None

        if self._debug_log:
            from dataset_generation.intermediate_results_logger.debug_data import (  # noqa: PLC0415
                DebugConfig,
                DebugPayloadFactory,
            )
            DebugPayloadFactory.instance().configure(
                debug_config=DebugConfig(enabled=True, phases=["persistence"]),
                base_log_directory="/tmp",  # unused: we never flush
                config_file_name="eval_inferencer_debug",
                component_names={},
            )

    def set_run_dir(self, run_dir: Path) -> None:
        if not self._debug_visualizations:
            return
        self._debug_output_dir = run_dir / "debug_vis" / "direct_vlm"
        handler = getattr(self._querier, "vlm_prompt_handler", None)
        if handler is not None and hasattr(handler, "set_debug_dir"):
            handler.set_debug_dir(self._debug_output_dir / "prompt_images")

    def infer(
        self, group: ImageMaskGroup, *, one_off: bool = False,
    ) -> tuple[list[PersistenceParameterEstimate], Optional[float]]:
        save_mask_visualization(group, self._debug_output_dir)
        instances = build_segmentation_results(group.masks, group.image.shape[-2:])

        image_id = group.image_path or f"anchor_t{group.anchor_time}_{'_'.join(group.track_ids)}"

        timer_name = f"Timer::vlm_direct_querier::{type(self._querier).__name__}"
        raw_estimates = None
        querier_debug = None
        elapsed_s = 0.0
        for attempt in range(3):
            try:
                live_ctx = nullcontext() if one_off else timer.time_block(timer_name)
                start = time.perf_counter()
                with live_ctx:
                    _, _, raw_estimates, querier_debug = self._querier.estimatePersistenceParameters(
                        "",     # scene_caption — ignored by DirectPersistenceVLMPromptHandler
                        {},     # object_descriptions — ignored
                        {},     # relationship_descriptions — ignored
                        None,   # time_and_date
                        group.image,
                        instances,
                        image_id,
                    )
                elapsed_s = time.perf_counter() - start
                break
            except ValueError as e:
                if attempt < 2:
                    print(
                        f"[direct_vlm] querier failed (attempt {attempt + 1}/3), retrying: {e}",
                        file=sys.stderr, flush=True,
                    )
                else:
                    print(
                        f"[direct_vlm] querier failed after 3 attempts, skipping group: {e}",
                        file=sys.stderr, flush=True,
                    )

        elapsed_ms: Optional[float] = None if one_off else elapsed_s * 1000.0

        if raw_estimates is None:
            msg = (
                f"[direct_vlm] querier exhausted retries for anchor_time="
                f"{group.anchor_time} image={group.image_path}; whole group "
                f"({len(group.track_ids)} tracks) lost."
            )
            if self._strict_object_coverage:
                raise RuntimeError(msg)
            print(f"WARNING {msg}", file=sys.stderr, flush=True)
            return (
                [
                    PersistenceParameterEstimate(params={}, metadata={"track_id": tid, "raw_estimate": {}})
                    for tid in group.track_ids
                ],
                elapsed_ms,
            )

        estimates = map_to_persistence_estimates(raw_estimates, group.track_ids, self._param_keys)

        missing_idx = [i for i, est in enumerate(estimates) if not est.params]
        if missing_idx:
            missing_tids = [group.track_ids[i] for i in missing_idx]
            msg = (
                f"[direct_vlm] anchor_time={group.anchor_time} "
                f"image={group.image_path}: {len(missing_idx)}/{len(group.track_ids)} "
                f"tracks have empty params after querier finished: "
                f"{missing_tids[:5]}{'...' if len(missing_tids) > 5 else ''}"
            )
            if self._strict_object_coverage:
                raise RuntimeError(msg)
            print(f"WARNING {msg}", file=sys.stderr, flush=True)

        write_debug_record(
            group=group,
            scene_caption="",
            object_descriptions={},
            raw_estimates=raw_estimates,
            estimates=estimates,
            querier_debug=querier_debug,
            debug_log=self._debug_log,
            debug_output_dir=self._debug_output_dir,
        )
        return estimates, elapsed_ms
