from dataset_generation.llm_persistence_querying.abs_llm_persistence_querier import AbsLLMPersistenceQuerier


class DummyPersistenceQuerier(AbsLLMPersistenceQuerier):
    def __init__(
            self
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
        super().__init__(name="DummyPersistenceQuerier")

    def estimatePersistenceParameters(self, scene_caption, object_descriptions, relationship_descriptions, time_and_date):

        prompt = ""
        response = ""
        persistence_parameter_estimates = {}
        return [prompt], response, persistence_parameter_estimates