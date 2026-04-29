

from abc import ABC, abstractmethod
from typing import Tuple

from dataset_generation.intermediate_results_logger.debug_data import DebugPayload


class AbstractSceneLevelCaptioner(ABC):

    def __init__(self, name: str):
        self.name = name

    def getName(self) -> str:
        return self.name

    @abstractmethod
    def generateCaption(self, image) -> Tuple[str, DebugPayload]:
        """Returns (caption, DebugPayload). Subclasses with no debug data may
        return (caption, DebugPayload.empty())."""
        pass