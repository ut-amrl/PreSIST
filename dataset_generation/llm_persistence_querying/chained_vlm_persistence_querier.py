"""
Chained VLM persistence querier: mirrors ChainedPromptQueryPersistenceQuerier.
Uses a VLM prompt handler and VLM querier; calls the VLM once per object with
original + masked images, then aggregates parsed responses and assigns object ids.
"""
from __future__ import annotations

import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch

from instance_segmentation.abs_instance_segmenter import SegmentationResults
from dataset_generation.intermediate_results_logger.debug_data import DebugPayload, DebugPayloadFactory
from dataset_generation.llm_persistence_querying.abs_llm_persistence_querier import AbsLLMPersistenceQuerier
from dataset_generation.llm_persistence_querying.vlm_querying.abs_vlm_querier import AbsVLMQuerier
from dataset_generation.llm_persistence_querying.prompt_handler.abs_vlm_prompt_handler import AbsVLMPromptHandler
from utils.timing import querier_timer_suffix, timer

from tqdm import tqdm


class ChainedVLMPersistenceQuerier(AbsLLMPersistenceQuerier):
    """
    Persistence querier that uses a VLM: one prompt + (original image, masked image) per object,
    then aggregates all responses and assigns object ids.
    """

    def __init__(
        self,
        vlmPromptHandler: AbsVLMPromptHandler,
        llmQuerier: AbsVLMQuerier,
        perObjectRetryAttempts: int = 2,
        errorOnMissingObjectIds: bool = False,
        max_concurrency: int = 8,
        error_on_object_failure: Optional[bool] = None,
    ):
        """
        Args:
            vlmPromptHandler: Builds per-object prompt text, prepares the per-object image list,
                and parses single-object response.
            llmQuerier: Sends (prompt, images) to the VLM.
            perObjectRetryAttempts: If a single-object call raises (VLM error, parse failure,
                schema validation failure), re-run that object up to this many additional
                attempts before giving up. Retried attempts are tagged ``_r{n}`` in the debug
                payload. Set to 0 to disable.
            errorOnMissingObjectIds: After all per-object retries are exhausted, if any input
                ids are still missing from the aggregated output, raise ValueError (True) or
                print a WARNING and continue (False).
            max_concurrency: When the underlying ``AbsVLMQuerier`` is using a vLLM server
                (``getattr(llmQuerier, "use_vllm_server", True)``), dispatch per-object calls
                through a thread pool capped at ``min(max_concurrency, len(instances))``.
                Local-Transformers queriers (``use_vllm_server=False``) are forced to serial
                since parallel calls would serialize on the model lock and risk OOM.
            error_on_object_failure: Deprecated alias for ``errorOnMissingObjectIds``.
                If provided, overrides ``errorOnMissingObjectIds`` and emits a DeprecationWarning.
        """
        if error_on_object_failure is not None:
            warnings.warn(
                "error_on_object_failure is deprecated; use errorOnMissingObjectIds instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            errorOnMissingObjectIds = bool(error_on_object_failure)

        name = f"CQ_VLM_{vlmPromptHandler.name}_{llmQuerier.name}_errOnFail{errorOnMissingObjectIds}"
        if perObjectRetryAttempts != 2:
            name += f"_retry{perObjectRetryAttempts}"
        if max_concurrency != 8:
            name += f"_conc{max_concurrency}"
        super().__init__(name=name)
        self.vlm_prompt_handler = vlmPromptHandler
        self.llm_querier = llmQuerier
        self.errorOnMissingObjectIds = bool(errorOnMissingObjectIds)
        self.perObjectRetryAttempts = max(0, int(perObjectRetryAttempts))
        self.max_concurrency = max(1, int(max_concurrency))
        self._uses_vllm_server = bool(getattr(llmQuerier, "use_vllm_server", True))
        self._per_query_timer_name = (
            f"Timer::{llmQuerier.__class__.__name__}_persistence_vlm_query"
            + querier_timer_suffix(
                use_vllm_server=self._uses_vllm_server,
                enable_thinking=getattr(llmQuerier, "enable_thinking", None),
            )
        )
        self._per_query_per_call_timer_name = f"{self._per_query_timer_name}_per_call"

    def estimatePersistenceParameters(
        self,
        scene_caption: str,
        object_descriptions: dict,
        relationship_descriptions: dict,
        time_and_date: Optional[str],
        image: torch.Tensor,
        instances: SegmentationResults,
        image_id: str,
    ) -> Tuple[Dict[str, str], Dict[str, str], List[Dict[str, Any]], DebugPayload]:
        """
        For each object: ask prompt handler for (prompt, images), call VLM, parse response,
        assign id. Returns (prompts_dict, responses_dict, persistence_parameter_estimates, DebugPayload)
        where dicts map object_id (str) -> prompt/response.
        """
        debug_payload = DebugPayloadFactory.instance().create("persistence", image_id)
        if not len(instances):
            return {}, {}, [], debug_payload

        obj_ids: List[str] = [str(int(instances.ids[i].item())) for i in range(len(instances))]
        input_ids = set(obj_ids)

        prompts_dict: Dict[str, str] = {}
        responses_dict: Dict[str, str] = {}
        persistence_parameter_estimates: List[Dict[str, Any]] = []

        def run_one(
            obj_id: str, attempt: int
        ) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Optional[Exception]]:
            """Build prompt+images, query, parse. Returns (prompt, response, parsed_or_None, error_or_None)."""
            try:
                prompt, images_for_vlm = self.vlm_prompt_handler.buildPromptAndImagesForObject(
                    scene_caption,
                    object_descriptions,
                    relationship_descriptions,
                    time_and_date,
                    image,
                    instances,
                    object_id=obj_id,
                )
            except Exception as e:
                return "", None, None, e

            try:
                with timer.time_block(self._per_query_per_call_timer_name):
                    response, sub_debug = self.llm_querier.queryVLM(prompt, images_for_vlm)
            except Exception as e:
                return prompt, None, None, e

            tag = f"obj_{obj_id}" + (f"_r{attempt}" if attempt > 0 else "")
            debug_payload.add(f"obj_input_id_{tag}", obj_id, kind="json")
            debug_payload.add(f"prompt_{tag}", prompt, kind="text")
            if sub_debug is not None:
                for rec in sub_debug:
                    debug_payload.add(
                        f"{rec.name}_{tag}",
                        rec.value,
                        kind=rec.kind,
                        defer=rec.defer,
                    )

            try:
                parsed = self.vlm_prompt_handler.parseSingleObjectPersistenceResponse(response)
            except Exception as e:
                return prompt, response, None, e

            if not isinstance(parsed, dict):
                return prompt, response, None, ValueError(
                    f"parseSingleObjectPersistenceResponse returned non-dict ({type(parsed).__name__})"
                )

            # Drop any id the model may have produced; we assign the canonical one.
            parsed.pop("id", None)
            parsed["id"] = obj_id
            return prompt, response, parsed, None

        def dispatch(ids: List[str], attempt: int):
            def task(oid: str, _att=attempt):
                return oid, run_one(oid, attempt=_att)

            use_threads = self._uses_vllm_server and self.max_concurrency > 1 and len(ids) > 1
            desc = "Querying VLM for objects in scene" + (f" (retry {attempt})" if attempt > 0 else "")
            if use_threads:
                workers = min(self.max_concurrency, len(ids))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    return list(tqdm(pool.map(task, ids), total=len(ids), desc=desc, leave=False))
            return [task(oid) for oid in tqdm(ids, desc=desc, leave=False)]

        # Initial pass.
        n_calls = len(obj_ids)
        with timer.time_block(self._per_query_timer_name, batch_count=n_calls):
            results = dispatch(obj_ids, attempt=0)

        seen_ids: set[str] = set()
        failed: List[Tuple[str, Exception]] = []
        for obj_id, (prompt, response, parsed, error) in results:
            if error is not None:
                failed.append((obj_id, error))
                if prompt:
                    prompts_dict.setdefault(obj_id, prompt)
                if response is not None:
                    responses_dict.setdefault(obj_id, response)
                continue
            if obj_id in seen_ids:
                continue
            seen_ids.add(obj_id)
            prompts_dict[obj_id] = prompt
            responses_dict[obj_id] = response or ""
            persistence_parameter_estimates.append(parsed)

        # Retry rounds for objects that failed.
        for attempt in range(1, self.perObjectRetryAttempts + 1):
            if not failed:
                break
            retry_ids = [oid for oid, _ in failed]
            print(
                f"[persistence_vlm] image {image_id}: retrying {len(retry_ids)} object(s) "
                f"(attempt {attempt}/{self.perObjectRetryAttempts}): {retry_ids[:10]}"
                f"{'...' if len(retry_ids) > 10 else ''}",
                file=sys.stderr,
                flush=True,
            )

            retry_results = dispatch(retry_ids, attempt=attempt)

            still_failed: List[Tuple[str, Exception]] = []
            for obj_id, (prompt, response, parsed, error) in retry_results:
                if error is not None:
                    still_failed.append((obj_id, error))
                    if prompt:
                        prompts_dict.setdefault(obj_id, prompt)
                    if response is not None:
                        responses_dict[obj_id] = response
                    continue
                if obj_id in seen_ids:
                    continue
                seen_ids.add(obj_id)
                prompts_dict[obj_id] = prompt
                responses_dict[obj_id] = response or ""
                persistence_parameter_estimates.append(parsed)
            failed = still_failed

        if failed:
            err_summary = "; ".join(f"{oid}: {type(e).__name__}: {e}" for oid, e in failed[:5])
            if len(failed) > 5:
                err_summary += f"; ... ({len(failed) - 5} more)"
            missing_ids = sorted(input_ids - seen_ids)
            msg = (
                f"[persistence_vlm] image {image_id}: {len(failed)} of {len(input_ids)} "
                f"objects failed after {self.perObjectRetryAttempts} retries. "
                f"missing_ids={missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}. "
                f"errors: {err_summary}"
            )
            if self.errorOnMissingObjectIds:
                raise ValueError(msg)
            print(f"WARNING {msg}", file=sys.stderr, flush=True)

        # Deterministic ordering.
        persistence_parameter_estimates.sort(key=lambda d: str(d.get("id", "")))

        return prompts_dict, responses_dict, persistence_parameter_estimates, debug_payload
