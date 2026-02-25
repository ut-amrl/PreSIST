"""
Chained VLM persistence querier: mirrors ChainedPromptQueryPersistenceQuerier.
Uses a VLM prompt handler and VLM querier; calls the VLM once per object with
original + masked images, then aggregates parsed responses and assigns object ids.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from instance_segmentation.abs_instance_segmenter import SegmentationResults
from dataset_generation.llm_persistence_querying.abs_llm_persistence_querier import AbsLLMPersistenceQuerier
from dataset_generation.llm_persistence_querying.vlm_querying.abs_vlm_querier import AbsVLMQuerier
from dataset_generation.llm_persistence_querying.prompt_handler.abs_vlm_prompt_handler import AbsVLMPromptHandler

from tqdm import tqdm


class ChainedVLMPersistenceQuerier(AbsLLMPersistenceQuerier):
    """
    Persistence querier that uses a VLM: one prompt + (original image, masked image) per object,
    then aggregates all responses and assigns object ids.
    """

    def __init__(
        self,
        vlmPromptHandler: AbsVLMPromptHandler,
        vlmQuerier: AbsVLMQuerier,
        error_on_object_failure: bool = False,
    ):
        """
        Args:
            vlmPromptHandler: Builds per-object prompt text, prepares the per-object image list,
                and parses single-object response.
            vlmQuerier: Sends (prompt, images) to the VLM.
        """
        super().__init__(name=f"CQ_VLM_{vlmPromptHandler.name}_{vlmQuerier.name}_errOnFail{error_on_object_failure}")
        self.vlm_prompt_handler = vlmPromptHandler
        self.vlm_querier = vlmQuerier
        self.error_on_object_failure = error_on_object_failure

    def estimatePersistenceParameters(
        self,
        scene_caption: str,
        object_descriptions: dict,
        relationship_descriptions: dict,
        time_and_date: Optional[str],
        image: torch.Tensor,
        instances: SegmentationResults,
    ) -> Tuple[Dict[str, str], Dict[str, str], List[Dict[str, Any]]]:
        """
        For each object: ask prompt handler for (prompt, images), call VLM, parse response,
        assign id. Returns (prompts_dict, responses_dict, persistence_parameter_estimates)
        where dicts map object_id (str) -> prompt/response.
        """
        if not len(instances):
            return {}, {}, []

        prompts_dict: Dict[str, str] = {}
        responses_dict: Dict[str, str] = {}
        persistence_parameter_estimates: List[Dict[str, Any]] = []

        # for i in range(len(instances)):
        for i in tqdm(range(len(instances)), desc="Querying VLM for objects in scene", leave=False):
            obj_id = int(instances.ids[i].item())

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
                response = self.vlm_querier.queryVLM(prompt, images_for_vlm)
                parsed = self.vlm_prompt_handler.parseSingleObjectPersistenceResponse(response)
                parsed["id"] = obj_id
                persistence_parameter_estimates.append(parsed)

                prompts_dict[str(obj_id)] = prompt
                responses_dict[str(obj_id)] = response
            except Exception as e:
                print(f"Skipping object id: Error processing object {obj_id} ({i+1}/{len(instances)}): {e}")
                print("----------------------------------------------------------------------------------------------------------")
                if (self.error_on_object_failure):
                    raise e
                continue

        return prompts_dict, responses_dict, persistence_parameter_estimates
