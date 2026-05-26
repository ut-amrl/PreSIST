from __future__ import annotations

import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

from utils.timing import timer
from dataset_generation.scene_level_captioning.abs_scene_level_captioner import AbstractSceneLevelCaptioner
from dataset_generation.instance_descriptions.abs_object_describer import AbstractObjectDescriber
from dataset_generation.llm_persistence_querying.abs_llm_persistence_querier import AbsLLMPersistenceQuerier

from persistence_eval.core.domain import ImageMaskGroup, PersistenceParameterEstimate
from persistence_eval.methods.inferencers.base import AbsPersistenceParameterInferencer
from persistence_eval.methods.inferencers._seg_helpers import build_segmentation_results
from persistence_eval.methods.inferencers._pipeline_cache import (
    PipelineCache,
    replay_timing,
)
from persistence_eval.methods.inferencers._pipeline_helpers import (
    map_to_persistence_estimates,
    save_mask_visualization,
    write_debug_record,
)


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
        debug_visualizations: bool = False,
        strict_object_coverage: bool = False,
        cache_dir: Optional[Path] = None,
        cache_force_regen: bool = False,
        cache_force_regen_phases: Optional[list[str]] = None,
    ) -> None:
        """
        strict_object_coverage: when True, raise instead of returning
            empty-params sentinels if the querier exhausts all retries and
            the group still has tracks without params. Default False
            preserves the prior silent-fill behaviour. Per-seg-id retries
            on partial LLM output are handled inside the querier itself
            (see ChainedPromptQueryPersistenceQuerier.missingIdRetryAttempts).
        cache_dir: when set, results for each phase (scene caption,
            per-object descriptions, persistence-querier output) are read
            from / written to this directory and indexed by component name,
            image_id, and track_id(s). Cached entries replay their original
            elapsed time into the timer so timing reports stay meaningful.
            None (default) disables caching entirely.
        cache_force_regen: when True, skip all cache reads (writes still
            happen). Useful for a clean re-run without ``rm -rf``ing the
            cache directory. Ignored when ``cache_force_regen_phases`` is
            non-empty.
        cache_force_regen_phases: per-phase allow-list of phases to
            regenerate; all others still hit cache. Valid entries are
            ``scene_caption``, ``object_descriptions``, ``persistence``.
            Wins over ``cache_force_regen``.
        """
        self._captioner = sceneCaptioner
        self._describer = objectDescriber
        self._querier = llmPersistenceQuerier
        self._param_keys = param_keys
        self._debug_log = debug_log
        self._debug_visualizations = debug_visualizations
        self._strict_object_coverage = bool(strict_object_coverage)
        self._debug_output_dir: Path | None = None

        self._cache: Optional[PipelineCache] = (
            PipelineCache(
                cache_dir,
                force_regen=cache_force_regen,
                force_regen_phases=cache_force_regen_phases,
            )
            if cache_dir is not None
            else None
        )

        # Component identifiers used as cache-key prefixes. Fall back to the
        # class name if the component doesn't expose its own name.
        self._captioner_name = getattr(sceneCaptioner, "name", None) or type(sceneCaptioner).__name__
        self._describer_name = getattr(objectDescriber, "name", None) or type(objectDescriber).__name__
        self._querier_name = getattr(llmPersistenceQuerier, "name", None) or type(llmPersistenceQuerier).__name__

        # The chained querier asks DebugPayloadFactory for a payload, which is
        # disabled by default outside the dataset-generation pipeline (factory
        # is never configured here). Configure it for the persistence phase so
        # the querier's per-batch prompts and raw LLM responses are buffered
        # and propagate back via the 4th return value of
        # estimatePersistenceParameters. We never call flush(), so text/JSON
        # entries stay in-memory only — no files are written by the factory.
        if self._debug_log:
            from dataset_generation.intermediate_results_logger.debug_data import (  # noqa: PLC0415
                DebugConfig,
                DebugPayloadFactory,
            )
            DebugPayloadFactory.instance().configure(
                debug_config=DebugConfig(enabled=True, phases=["persistence"]),
                base_log_directory="/tmp",  # unused: we never flush
                config_file_name="eval_inferencer_debug",
                component_names={},  # persistence phase layout only needs configFileName
            )

    def wants_warmup(self) -> bool:
        """True iff any of the three pipeline components want to be warmed.
        Each component opts in/out via its own ``wants_warmup()`` (defaulting
        to True if absent — local model backends typically benefit). Returns
        False only when *every* component is HTTP-only or otherwise warmup-
        unfriendly (e.g. all-Gemini pipelines).
        """
        def _wants(component) -> bool:
            fn = getattr(component, "wants_warmup", None)
            return bool(fn()) if callable(fn) else True
        return _wants(self._captioner) or _wants(self._describer) or _wants(self._querier)

    def set_run_dir(self, run_dir: Path) -> None:
        if self._debug_visualizations:
            self._debug_output_dir = run_dir / "debug_vis"
            if hasattr(self._describer, "set_debug_dir"):
                self._describer.set_debug_dir(self._debug_output_dir / "object_describer")

    def infer(
        self, group: ImageMaskGroup, *, one_off: bool = False,
    ) -> tuple[list[PersistenceParameterEstimate], Optional[float]]:
        # print(
        #     f"[llm_pipeline] image={group.image_path or 'unknown'}  tracks={group.track_ids}",
        #     file=sys.stderr, flush=True,
        # )
        save_mask_visualization(group, self._debug_output_dir)
        instances = build_segmentation_results(group.masks, group.image.shape[-2:])

        image_id = group.image_path or f"anchor_t{group.anchor_time}_{'_'.join(group.track_ids)}"

        object_descriptions, desc_elapsed_s = self._get_object_descriptions(
            group, instances, image_id, one_off=one_off,
        )
        scene_caption, caption_elapsed_s = self._get_scene_caption(
            group, image_id, one_off=one_off,
        )
        raw_estimates, querier_debug, persistence_elapsed_s = self._get_persistence(
            group, instances, image_id, scene_caption, object_descriptions, one_off=one_off,
        )

        # Sum of original-computation elapseds (live for misses, replayed for
        # cache hits) so cache hits don't collapse the reported latency to ~0.
        # Debug/viz overhead (mask save, debug record write) lives outside
        # this sum. None on one_off so callers don't fold warmup into stats.
        elapsed_ms: Optional[float] = (
            None
            if one_off
            else (caption_elapsed_s + desc_elapsed_s + persistence_elapsed_s) * 1000.0
        )

        # Below is the original post-querier handling, unchanged. We bail out
        # early with empty-params sentinels iff every retry attempt raised
        # ValueError inside _get_persistence.

        if raw_estimates is None:
            msg = (
                f"[llm_pipeline] querier exhausted retries for anchor_time="
                f"{group.anchor_time} image={group.image_path}; whole group "
                f"({len(group.track_ids)} tracks) lost."
            )
            if self._strict_object_coverage:
                raise RuntimeError(msg)
            print(f"WARNING {msg}", file=sys.stderr, flush=True)
            return (
                [PersistenceParameterEstimate(params={}, metadata={"track_id": tid, "raw_estimate": {}}) for tid in group.track_ids],
                elapsed_ms,
            )

        estimates = map_to_persistence_estimates(raw_estimates, group.track_ids, self._param_keys)

        # Even after the querier's own per-sub-batch missing-id retries, the
        # final merged output can still under-cover the requested seg_ids
        # (e.g. retries exhausted with errorOnMissingObjectIds=False, or the
        # LLM persistently dropped certain ids). Surface this once per anchor
        # so it's never silent, and escalate to a raise under strict mode.
        missing_idx = [i for i, est in enumerate(estimates) if not est.params]
        if missing_idx:
            missing_tids = [group.track_ids[i] for i in missing_idx]
            msg = (
                f"[llm_pipeline] anchor_time={group.anchor_time} "
                f"image={group.image_path}: {len(missing_idx)}/{len(group.track_ids)} "
                f"tracks have empty params after querier finished: "
                f"{missing_tids[:5]}{'...' if len(missing_tids) > 5 else ''}"
            )
            if self._strict_object_coverage:
                raise RuntimeError(msg)
            print(f"WARNING {msg}", file=sys.stderr, flush=True)

        write_debug_record(
            group=group,
            scene_caption=scene_caption,
            object_descriptions=object_descriptions,
            raw_estimates=raw_estimates,
            estimates=estimates,
            querier_debug=querier_debug,
            debug_log=self._debug_log,
            debug_output_dir=self._debug_output_dir,
        )
        return estimates, elapsed_ms

    # ----------------------------------------------------------- cached phases

    def _scene_caption_timer_name(self) -> str:
        return f"Timer::scene_cap::{type(self._captioner).__name__}"

    def _describer_timer_name(self) -> str:
        return f"Timer::obj_desc::{type(self._describer).__name__}"

    def _querier_timer_name(self) -> str:
        return f"Timer::llm_querier::{type(self._querier).__name__}"

    def _get_scene_caption(
        self, group: ImageMaskGroup, image_id: str, *, one_off: bool = False,
    ) -> tuple[str, float]:
        timer_name = self._scene_caption_timer_name()

        if self._cache is not None and not one_off:
            cached_elapsed = self._cache.cached_scene_caption_elapsed(
                self._captioner_name, image_id,
            )
            if cached_elapsed is not None:
                caption = self._cache.get_scene_caption(self._captioner_name, image_id)
                if caption is not None:
                    replay_timing(timer_name, cached_elapsed)
                    return caption, cached_elapsed

        # Cache miss (or one-off): compute. one_off=True suppresses both the
        # timer.time_block recording and the cache write so a warmup call
        # contaminates neither the timing aggregates nor the cached elapsed.
        live_ctx = nullcontext() if one_off else timer.time_block(timer_name)
        start = time.perf_counter()
        with live_ctx:
            caption, _ = self._captioner.generateCaption(group.image, image_id=image_id)
        elapsed = time.perf_counter() - start

        if self._cache is not None and not one_off:
            self._cache.put_scene_caption(
                self._captioner_name, image_id,
                caption=caption, elapsed_s=elapsed,
            )
        return caption, elapsed

    def _get_object_descriptions(
        self, group: ImageMaskGroup, instances, image_id: str, *, one_off: bool = False,
    ) -> tuple[dict, float]:
        timer_name = self._describer_timer_name()
        K = len(group.track_ids)
        merged: dict = {}
        total_elapsed_s = 0.0

        # Determine which positions are cached vs missing.
        miss_positions: list[int] = []
        if self._cache is not None and not one_off:
            for pos in range(K):
                tid = group.track_ids[pos]
                cached = self._cache.get_object_description(
                    self._describer_name, image_id, tid,
                )
                if cached is None:
                    miss_positions.append(pos)
                else:
                    description, elapsed = cached
                    merged[pos] = description
                    replay_timing(timer_name, elapsed)
                    total_elapsed_s += elapsed
        else:
            miss_positions = list(range(K))

        if not miss_positions:
            return merged, total_elapsed_s

        # Compute the misses. SegmentationResults.ids = positions in the
        # original group (build_segmentation_results uses arange), so we can
        # subset directly by position.
        filtered = instances.filterByIds(miss_positions)
        live_ctx = nullcontext() if one_off else timer.time_block(timer_name)
        start = time.perf_counter()
        with live_ctx:
            fresh, _ = self._describer.describeObjects(
                group.image, filtered, image_id=image_id,
            )
        elapsed = time.perf_counter() - start
        total_elapsed_s += elapsed

        # Distribute total elapsed across miss items so per-item cache replays
        # are realistic. This is an approximation; the describer doesn't
        # necessarily scale linearly with N, but it's the best we have.
        per_item_elapsed = elapsed / max(1, len(miss_positions))

        for pos in miss_positions:
            description = fresh.get(pos, "")
            merged[pos] = description
            if self._cache is not None and not one_off:
                self._cache.put_object_description(
                    self._describer_name, image_id, group.track_ids[pos],
                    description=description, elapsed_s=per_item_elapsed,
                )
        return merged, total_elapsed_s

    def _get_persistence(
        self,
        group: ImageMaskGroup,
        instances,
        image_id: str,
        scene_caption: str,
        object_descriptions: dict,
        *,
        one_off: bool = False,
    ) -> tuple[Optional[list], Any, float]:
        timer_name = self._querier_timer_name()

        # Cache hit: replay timing and skip both the call AND the retry loop.
        # Returns (raw_estimates, None, cached_elapsed) — querier_debug is not
        # cached, and write_debug_record already tolerates None. The cache key
        # folds in a content hash of scene_caption + object_descriptions so any
        # change upstream invalidates this entry.
        if self._cache is not None and not one_off:
            cached = self._cache.get_persistence(
                self._querier_name, image_id, group.track_ids,
                scene_caption=scene_caption,
                object_descriptions=object_descriptions,
            )
            if cached is not None:
                raw_estimates, elapsed = cached
                replay_timing(timer_name, elapsed)
                return raw_estimates, None, elapsed

        # Cache miss (or one-off): run with the same 3-retry loop as before.
        raw_estimates = None
        querier_debug = None
        elapsed = 0.0
        for attempt in range(3):
            try:
                live_ctx = nullcontext() if one_off else timer.time_block(timer_name)
                start = time.perf_counter()
                with live_ctx:
                    _, _, raw_estimates, querier_debug = self._querier.estimatePersistenceParameters(
                        scene_caption,
                        object_descriptions,
                        {},    # relationship descriptions — always empty at inference time
                        None,  # time_and_date
                        group.image,
                        instances,
                        image_id,
                    )
                elapsed = time.perf_counter() - start
                break
            except ValueError as e:
                if attempt < 2:
                    print(f"[llm_pipeline] querier failed (attempt {attempt + 1}/3), retrying: {e}", file=sys.stderr, flush=True)
                else:
                    print(f"[llm_pipeline] querier failed after 3 attempts, skipping group: {e}", file=sys.stderr, flush=True)

        if raw_estimates is not None and self._cache is not None and not one_off:
            self._cache.put_persistence(
                self._querier_name, image_id, group.track_ids,
                scene_caption=scene_caption,
                object_descriptions=object_descriptions,
                raw_estimates=list(raw_estimates), elapsed_s=elapsed,
            )

        return raw_estimates, querier_debug, elapsed
