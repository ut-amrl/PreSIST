import torch

from dataset_generation.llm_persistence_querying.abs_llm_persistence_querier import AbsLLMPersistenceQuerier
from dataset_generation.llm_persistence_querying.llm_querying.abs_llm_querier import AbsLLMQuerier
from dataset_generation.llm_persistence_querying.prompt_handler.abs_prompt_handler import AbsPromptHandler

from instance_segmentation.abs_instance_segmenter import SegmentationResults

from typing import Optional

class ChainedPromptQueryPersistenceQuerier(AbsLLMPersistenceQuerier):
    def __init__(
            self,
            promptHandler: AbsPromptHandler,
            llmQuerier: AbsLLMQuerier,
            maxObjsPerQuery: Optional[int] = 50,
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


    def divideIntoObjectGroups(self, object_descriptions, instances):
        """
        Divides objects into groups for prompting, if maxObjsPerQuery is set.
        Returns a list of object_id lists.
        """
        if (self.maxObjsPerQuery is None) or (len(object_descriptions) <= self.maxObjsPerQuery):
            return [(object_descriptions, instances)]

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
    ):

        prompts = []
        responses = []
        persistence_parameter_estimates = []
        for obj_desc_group, _ in self.divideIntoObjectGroups(object_descriptions, instances):

            prompt = self.prompt_handler.generatePrompt(
                scene_caption,
                obj_desc_group,
                relationship_descriptions,
                time_and_date
            )
            print("Sending prompt to LLM...")
            print(prompt)
            response = self.llm_querier.queryLLM(prompt)
            print("Received response from LLM:")
            print(response)
            print("Parsing persistence parameters from response...")
            parsed = self.prompt_handler.parsePersistenceParamsFromLLMResponse(response)

            prompts.append(prompt)
            responses.append(response)
            persistence_parameter_estimates.append(parsed)

        if (len(persistence_parameter_estimates) == 1):
            return prompts, responses[0], persistence_parameter_estimates[0]

        mergedPrompt = "\n\n".join(prompts)
        mergedResponse = "\n\n".join(responses)
        # Note: persistence_parameter_estimates is a list of dicts, so we just concatenate
        mergedPersistenceParameterEstimates = []
        for estimate in persistence_parameter_estimates:
            mergedPersistenceParameterEstimates.extend(estimate)


        return [mergedPrompt], mergedResponse, mergedPersistenceParameterEstimates
