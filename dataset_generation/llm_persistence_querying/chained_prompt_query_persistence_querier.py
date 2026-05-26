import sys
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
            missingIdRetryAttempts: int = 2,
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

            missingIdRetryAttempts:
                After each sub-batch's initial query, if some of the input
                object ids are missing from the LLM's parsed response (the
                LLM silently dropped them), re-query just those missing ids
                up to this many times. Recovered ids are merged into the
                batch result. After retries are exhausted, behaviour depends
                on ``errorOnMissingObjectIds``: raise ValueError if True, or
                emit a WARNING and continue if False. Set to 0 to disable.
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
        self.missingIdRetryAttempts = max(0, int(missingIdRetryAttempts))
        self._per_query_timer_name = (
            f"Timer::{llmQuerier.__class__.__name__}_persistence_llm_query"
            + querier_timer_suffix(
                use_vllm_server=getattr(llmQuerier, "use_vllm_server", True),
                enable_thinking=getattr(llmQuerier, "enable_thinking", None),
            )
        )
        self._per_query_per_call_timer_name = f"{self._per_query_timer_name}_per_call"


    def wants_warmup(self) -> bool:
        """Defer to the wrapped llm_querier: this class is a pure passthrough
        for warmup purposes (it doesn't hold model weights itself). Defaults
        to True if the inner querier doesn't express an opinion, since most
        local model backends benefit from warmup.
        """
        inner = getattr(self.llm_querier, "wants_warmup", None)
        return bool(inner()) if callable(inner) else True

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
            # Stringify on both sides: object_descriptions keys may be ints
            # (e.g. GemmaVLMObjectDescriber) while instanceGroup.ids is always
            # an int tensor — without the str() on the dict side, the
            # membership check silently produced an empty obj_desc_group and
            # the prompt rendered no objects.
            obj_ids = set([str(id.item()) for id in instanceGroup.ids])
            obj_desc_group = {obj_id: desc for obj_id, desc in object_descriptions.items() if str(obj_id) in obj_ids}
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

        prompts, responses, persistence_parameter_estimates = self.estimatePersistenceParametersInternal(
            scene_caption,
            object_descriptions,
            relationship_descriptions,
            time_and_date,
            image,
            instances,
            image_id,
            debug_payload,
        )
        mergedPrompt = "\n\n".join(prompts)
        mergedResponse = "\n\n".join(responses)
        # Note: persistence_parameter_estimates is a list of dicts, so we just concatenate
        mergedPersistenceParameterEstimates = []
        for estimate in persistence_parameter_estimates:
            mergedPersistenceParameterEstimates.extend(estimate)

        mergedPersistenceParameterEstimates.sort(
            key=lambda d: str(d.get("id", "")) if isinstance(d, dict) else ""
        )


        if (len(persistence_parameter_estimates) == 1):
            return prompts, responses[0], persistence_parameter_estimates[0], debug_payload

        return [mergedPrompt], mergedResponse, mergedPersistenceParameterEstimates, debug_payload


    def estimatePersistenceParametersInternal(
        self,
        scene_caption,
        object_descriptions,
        relationship_descriptions,
        time_and_date,
        image: torch.Tensor,
        instances: SegmentationResults,
        image_id: str,
        debug_payload,
        groups_override=None,
        curr_retry_attempt=0,
    ):

        if (groups_override is not None):
            groups = groups_override
        else:
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

        def run_one(prompt):
            with timer.time_block(self._per_query_per_call_timer_name):
                return self.llm_querier.queryLLM(prompt)

        n_batches = len(batch_prompts)
        with timer.time_block(self._per_query_timer_name, batch_count=n_batches):
            if self.max_concurrency <= 1 or n_batches <= 1:
                llm_outputs = [run_one(p) for p in batch_prompts]
            else:
                workers = min(self.max_concurrency, n_batches)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    llm_outputs = list(pool.map(
                        run_one,
                        batch_prompts,
                    ))

        prompts = []
        responses = []
        missing_from_each_batch = []
        totalMissing = 0
        persistence_parameter_estimates = []
        for batch_idx, ((obj_desc_group, _), prompt, (response, sub_debug)) in enumerate(
            zip(groups, batch_prompts, llm_outputs)
        ):
            batch_tag = f"batch_{batch_idx:03d}"
            if curr_retry_attempt > 0:
                batch_tag += f"_r{curr_retry_attempt}"
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

            raw_parsed = self.prompt_handler.parsePersistenceParamsFromLLMResponse(response)

            input_ids = {str(k) for k in obj_desc_group.keys()}
            # Filter to records the caller actually asked about. This drops:
            #   - records missing or with null "id" (unparseable)
            #   - records whose "id" is out-of-scope for this sub-batch
            #     (LLM hallucinated an id we didn't ask about)
            #   - duplicate records for the same id within a single response
            #     (the LLM occasionally repeats itself); we keep the first
            #     occurrence and discard the rest.
            seen_ids = set()
            parsed = []
            for p in raw_parsed:
                if not isinstance(p, dict):
                    continue
                pid = p.get("id")
                if pid is None:
                    continue
                pid_str = str(pid)
                if pid_str not in input_ids or pid_str in seen_ids:
                    continue
                seen_ids.add(pid_str)
                parsed.append(p)

            output_ids = seen_ids
            missing_ids = input_ids - output_ids
            missing_from_each_batch.append((missing_ids, len(input_ids)))
            totalMissing += len(missing_ids)

            if missing_ids:
                msg = (
                    f"[persistence] image {image_id}: {len(missing_ids)} of {len(input_ids)} "
                    f"input object ids missing from LLM output: {sorted(missing_ids)}"
                )
                if (curr_retry_attempt < self.missingIdRetryAttempts):
                    retries_remaining = self.missingIdRetryAttempts - curr_retry_attempt
                    msg += f" (will retry up to {retries_remaining} more time{'s' if retries_remaining != 1 else ''})"
                if (self.errorOnMissingObjectIds and curr_retry_attempt >= self.missingIdRetryAttempts):
                    raise ValueError(msg)
                print(f"WARNING {msg}")

            if len(parsed) == 0:
                resp_snippet = (response or "")[:500]
                print(
                    f"WARNING [persistence] image {image_id} batch {batch_idx}: "
                    f"parsed 0 usable records from LLM response "
                    f"(input_ids={sorted(input_ids)}, raw_parsed_count={len(raw_parsed)}). "
                    f"Response[:500]: {resp_snippet!r}",
                    file=sys.stderr,
                    flush=True,
                )

            prompts.append(prompt)
            responses.append(response)
            persistence_parameter_estimates.append(parsed)

        if (curr_retry_attempt < self.missingIdRetryAttempts) and totalMissing > 0:
            # If there are any missing ids across the batches, and we have retry attempts left, generate new groups for just the missing ids and retry them
            new_groups = self.generateGroupsForRetry(missing_from_each_batch, groups)
            if new_groups:
                retry_prompts, retry_responses, retry_estimates = self.estimatePersistenceParametersInternal(
                    scene_caption,
                    object_descriptions,
                    relationship_descriptions,
                    time_and_date,
                    image,
                    instances,
                    image_id,
                    debug_payload,
                    groups_override=new_groups,
                    curr_retry_attempt=curr_retry_attempt + 1,
                )
                # Merge retry results back into main results
                prompts.extend(retry_prompts)
                responses.extend(retry_responses)
                persistence_parameter_estimates.extend(retry_estimates)

        return prompts, responses, persistence_parameter_estimates


    def generateGroupsForRetry(self, missing_from_each_batch, groups):
        """
        Given a list of (missing_ids, total_ids) for each batch, and the original groups,
        generate new groups for retrying just the missing ids.

        Returns a list of new groups in the same format as divideIntoObjectGroups output.
        """
        new_groups = []

        for (missing_ids, total_ids), (obj_desc_group, instance_group) in zip(missing_from_each_batch, groups):
            if not missing_ids:
                continue

            # If all ids are missing, split the group in half to avoid overwhelming the LLM with a large retry batch
            if (len(missing_ids) == total_ids) and (len(missing_ids) > 1):
                missing_ids_1 = set(list(missing_ids)[: len(missing_ids) // 2])
                missing_ids_2 = missing_ids - missing_ids_1
                for missing_ids_subset in [missing_ids_1, missing_ids_2]:
                     if missing_ids_subset:
                        new_obj_desc_group = {obj_id: desc for obj_id, desc in obj_desc_group.items() if obj_id in missing_ids_subset}
                        new_instance_group = instance_group.filterByIds(missing_ids_subset)
                        new_groups.append((new_obj_desc_group, new_instance_group))
            else:
                # Filter the original obj_desc_group and instance_group to just the missing ids
                new_obj_desc_group = {obj_id: desc for obj_id, desc in obj_desc_group.items() if obj_id in missing_ids}
                new_instance_group = instance_group.filterByIds(missing_ids)

                new_groups.append((new_obj_desc_group, new_instance_group))

        return new_groups