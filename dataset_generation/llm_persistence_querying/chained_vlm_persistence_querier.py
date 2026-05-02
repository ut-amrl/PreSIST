"""
Chained VLM persistence querier: mirrors ChainedPromptQueryPersistenceQuerier.
Uses a VLM prompt handler and VLM querier; calls the VLM once per object with
original + masked images, then aggregates parsed responses and assigns object ids.
"""
from __future__ import annotations

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
        error_on_object_failure: bool = False,
    ):
        """
        Args:
            vlmPromptHandler: Builds per-object prompt text, prepares the per-object image list,
                and parses single-object response.
            llmQuerier: Sends (prompt, images) to the VLM.
        """
        super().__init__(name=f"CQ_VLM_{vlmPromptHandler.name}_{llmQuerier.name}_errOnFail{error_on_object_failure}")
        self.vlm_prompt_handler = vlmPromptHandler
        self.llm_querier = llmQuerier
        self.error_on_object_failure = error_on_object_failure
        self._per_query_timer_name = (
            f"Timer::{llmQuerier.__class__.__name__}_persistence_vlm_query"
            + querier_timer_suffix(
                use_vllm_server=getattr(llmQuerier, "use_vllm_server", True),
                enable_thinking=getattr(llmQuerier, "enable_thinking", None),
            )
        )

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

        prompts_dict: Dict[str, str] = {}
        responses_dict: Dict[str, str] = {}
        persistence_parameter_estimates: List[Dict[str, Any]] = []

        # for i in range(len(instances)):
        for i in tqdm(range(len(instances)), desc="Querying VLM for objects in scene", leave=False):
            obj_id = str(int(instances.ids[i].item()))

            prompt, images_for_vlm = self.vlm_prompt_handler.buildPromptAndImagesForObject(
                scene_caption,
                object_descriptions,
                relationship_descriptions,
                time_and_date,
                image,
                instances,
                object_id=obj_id,
            )

            try:
                with timer.time_block(self._per_query_timer_name):
                    response, sub_debug = self.llm_querier.queryVLM(prompt, images_for_vlm)
                debug_payload.merge(sub_debug)
                parsed = self.vlm_prompt_handler.parseSingleObjectPersistenceResponse(response)
                parsed["id"] = obj_id
                persistence_parameter_estimates.append(parsed)

                prompts_dict[obj_id] = prompt
                responses_dict[obj_id] = response
            except Exception as e:
                print(f"Skipping object id: Error processing object {obj_id} ({i+1}/{len(instances)}): {e}")
                print("----------------------------------------------------------------------------------------------------------")
                if (self.error_on_object_failure):
                    raise e
                continue

        return prompts_dict, responses_dict, persistence_parameter_estimates, debug_payload
