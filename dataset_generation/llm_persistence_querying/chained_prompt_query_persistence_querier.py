import torch

from dataset_generation.llm_persistence_querying.abs_llm_persistence_querier import AbsLLMPersistenceQuerier
from dataset_generation.llm_persistence_querying.llm_querying.abs_llm_querier import AbsLLMQuerier
from dataset_generation.llm_persistence_querying.prompt_handler.abs_prompt_handler import AbsPromptHandler

from instance_segmentation.abs_instance_segmenter import SegmentationResults

class ChainedPromptQueryPersistenceQuerier(AbsLLMPersistenceQuerier):
    def __init__(
            self,
            promptHandler: AbsPromptHandler,
            llmQuerier: AbsLLMQuerier
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
        super().__init__(name=f"CQ_{promptHandler.name}_{llmQuerier.name}")
        self.prompt_handler = promptHandler
        self.llm_querier = llmQuerier

    def estimatePersistenceParameters(
        self,
        scene_caption,
        object_descriptions,
        relationship_descriptions,
        time_and_date,
        image: torch.Tensor,
        instances: SegmentationResults,
    ):
        prompt = self.prompt_handler.generatePrompt(
            scene_caption,
            object_descriptions,
            relationship_descriptions,
            time_and_date
        )
        print("Sending prompt to LLM...")
        print(prompt)
        response = self.llm_querier.queryLLM(prompt)
        print("Received response from LLM:")
        print(response)
        print("Parsing persistence parameters from response...")
        persistence_parameter_estimates = self.prompt_handler.parsePersistenceParamsFromLLMResponse(response)
        return [prompt], response, persistence_parameter_estimates
