from abc import ABC, abstractmethod
from typing import Tuple

from dataset_generation.intermediate_results_logger.debug_data import DebugPayload


class AbsLLMQuerier(ABC):

    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name

    @abstractmethod
    def queryLLM(self, prompt) -> Tuple[str, DebugPayload]:
        """Returns (response_text, DebugPayload). Subclasses with no debug data
        should return (response, DebugPayload.empty())."""
        pass
