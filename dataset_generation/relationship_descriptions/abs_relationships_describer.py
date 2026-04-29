from abc import ABC, abstractmethod
from typing import Tuple

from dataset_generation.intermediate_results_logger.debug_data import DebugPayload


class AbsRelationshipDescriber(ABC):

    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name

    @abstractmethod
    def describeRelationships(self, image, instances, objectDescriptions) -> Tuple[object, DebugPayload]:
        """Returns (relationships, DebugPayload). Subclasses with no debug data may
        return (relationships, DebugPayload.empty())."""
        pass