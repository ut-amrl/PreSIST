

from abc import ABC, abstractmethod
from typing import Tuple

from dataset_generation.intermediate_results_logger.debug_data import DebugPayload


class AbstractObjectDescriber(ABC):

    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name

    @abstractmethod
    def describeObjects(self, image, instances) -> Tuple[dict, DebugPayload]:
        """Returns (descriptions, DebugPayload). Subclasses with no debug data may
        return (descriptions, DebugPayload.empty())."""
        pass