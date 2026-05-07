from concurrent.futures import ThreadPoolExecutor

import torch

from dataset_generation.intermediate_results_logger.debug_data import DebugPayloadFactory
from dataset_generation.llm_persistence_querying.abs_llm_persistence_querier import AbsLLMPersistenceQuerier
from dataset_generation.llm_persistence_querying.llm_querying.abs_llm_querier import AbsLLMQuerier
from dataset_generation.llm_persistence_querying.prompt_handler.abs_prompt_handler import AbsPromptHandler

from instance_segmentation.abs_instance_segmenter import SegmentationResults
from utils.timing import querier_timer_suffix, timer

from typing import Optional

class ChainedPromptQueryPersistenceQuerier(AbsLLMPersistenceQuerier):
    def __init__(
            self,
            promptHandler: AbsPromptHandler,
            llmQuerier: AbsLLMQuerier,
            maxObjsPerQuery: Optional[int] = 50,
            errorOnMissingObjectIds: bool = True,
            useSpatialGrouping: bool = False,
            max_concurrency: int = 8,
    ):
        """
        A persistence querier that uses a prompt handler to create prompts,
        and an LLM querier to send those prompts to an LLM.

        Args:
            prompt_handler:
                An instance of AbsPromptHandler that creates prompts
                from scene captions, object descriptions, and relationship descriptions.

            llm_querier:
                An instance of AbsLLMQuerier that sends prompts to an LLM
                and returns the LLM's responses.
        """
        name = f"CQ_{promptHandler.name}_{llmQuerier.name}"
        if (maxObjsPerQuery is not None):
            name += f"_mO{maxObjsPerQuery}"
        super().__init__(name=name)
        self.prompt_handler = promptHandler
        self.llm_querier = llmQuerier
        self.maxObjsPerQuery = maxObjsPerQuery
        self.errorOnMissingObjectIds = errorOnMissingObjectIds
        self.useSpatialGrouping = useSpatialGrouping
        self.max_concurrency = max(1, int(max_concurrency))
        self._per_query_timer_name = (
            f"Timer::{llmQuerier.__class__.__name__}_persistence_llm_query"
            + querier_timer_suffix(
                use_vllm_server=getattr(llmQuerier, "use_vllm_server", True),
                enable_thinking=getattr(llmQuerier, "enable_thinking", None),
            )
        )


    def divideIntoObjectGroups(self, object_descriptions, instances):
        """
        Divides objects into groups for prompting, if maxObjsPerQuery is set.
        Returns a list of object_id lists.
        """
        if (self.maxObjsPerQuery is None) or (len(object_descriptions) <= self.maxObjsPerQuery):
            return [(object_descriptions, instances)]

        if self.useSpatialGrouping:
            instanceGroups = instances.divideIntoSpatialSubsets(self.maxObjsPerQuery)
        else:
            instanceGroups = instances.divideIntoSubsets(self.maxObjsPerQuery)
        objDescGroups = []
        for instanceGroup in instanceGroups:
            obj_ids = set([str(id.item()) for id in instanceGroup.ids])
            obj_desc_group = {obj_id: desc for obj_id, desc in object_descriptions.items() if obj_id in obj_ids}
            objDescGroups.append(obj_desc_group)

        return list(zip(objDescGroups, instanceGroups))



    def estimatePersistenceParameters(
        self,
        scene_caption,
        object_descriptions,
        relationship_descriptions,
        time_and_date,
        image: torch.Tensor,
        instances: SegmentationResults,
        image_id: str,
    ):

        debug_payload = DebugPayloadFactory.instance().create("persistence", image_id)

        groups = list(self.divideIntoObjectGroups(object_descriptions, instances))
        batch_prompts = [
            self.prompt_handler.generatePrompt(
                scene_caption,
                obj_desc_group,
                relationship_descriptions,
                time_and_date,
            )
            for (obj_desc_group, _) in groups
        ]

        per_call_timer_name = f"{self._per_query_timer_name}_per_call"

        def run_one(batch_idx, prompt):
            with timer.time_block(per_call_timer_name):
                return self.llm_querier.queryLLM(prompt)

        n_batches = len(batch_prompts)
        with timer.time_block(self._per_query_timer_name, batch_count=n_batches):
            if self.max_concurrency <= 1 or n_batches <= 1:
                llm_outputs = [run_one(i, p) for i, p in enumerate(batch_prompts)]
            else:
                workers = min(self.max_concurrency, n_batches)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    llm_outputs = list(pool.map(
                        lambda ip: run_one(ip[0], ip[1]),
                        enumerate(batch_prompts),
                    ))

        prompts = []
        responses = []
        persistence_parameter_estimates = []
        for batch_idx, ((obj_desc_group, _), prompt, (response, sub_debug)) in enumerate(
            zip(groups, batch_prompts, llm_outputs)
        ):
            batch_tag = f"batch_{batch_idx:03d}"
            debug_payload.add(
                f"batch_obj_ids_{batch_tag}",
                sorted(str(k) for k in obj_desc_group.keys()),
                kind="json",
            )
            debug_payload.add(
                f"prompt_{batch_tag}",
                prompt,
                kind="text",
            )
            if sub_debug is not None:
                for rec in sub_debug:
                    debug_payload.add(
                        f"{rec.name}_{batch_tag}",
                        rec.value,
                        kind=rec.kind,
                        defer=rec.defer,
                    )

            parsed = self.prompt_handler.parsePersistenceParamsFromLLMResponse(response)

            input_ids = {str(k) for k in obj_desc_group.keys()}
            output_ids = {str(p.get("id")) for p in parsed if isinstance(p, dict) and p.get("id") is not None}
            missing_ids = input_ids - output_ids
            if missing_ids:
                msg = (
                    f"[persistence] image {image_id}: {len(missing_ids)} of {len(input_ids)} "
                    f"input object ids missing from LLM output: {sorted(missing_ids)}"
                )
                if self.errorOnMissingObjectIds:
                    raise ValueError(msg)
                print(f"WARNING {msg}")

            prompts.append(prompt)
            responses.append(response)
            persistence_parameter_estimates.append(parsed)

        if (len(persistence_parameter_estimates) == 1):
            return prompts, responses[0], persistence_parameter_estimates[0], debug_payload

        mergedPrompt = "\n\n".join(prompts)
        mergedResponse = "\n\n".join(responses)
        # Note: persistence_parameter_estimates is a list of dicts, so we just concatenate
        mergedPersistenceParameterEstimates = []
        for estimate in persistence_parameter_estimates:
            mergedPersistenceParameterEstimates.extend(estimate)

        mergedPersistenceParameterEstimates.sort(
            key=lambda d: str(d.get("id", "")) if isinstance(d, dict) else ""
        )

        return [mergedPrompt], mergedResponse, mergedPersistenceParameterEstimates, debug_payload
