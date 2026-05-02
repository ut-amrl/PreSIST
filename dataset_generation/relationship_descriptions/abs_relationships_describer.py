from abc import ABC, abstractmethod
from typing import Tuple

from dataset_generation.intermediate_results_logger.debug_data import DebugPayload


class AbsRelationshipDescriber(ABC):

    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name

    @abstractmethod
    def describeRelationships(self, image, instances, objectDescriptions, image_id: str) -> Tuple[object, DebugPayload]:
        """Returns (relationships, DebugPayload).

        Subclasses obtain a DebugPayload via
        ``DebugPayloadFactory.instance().create("relationships", image_id)``,
        add entries to it, and return it alongside the relationships. The
        caller (the logger) is responsible for calling ``flush()`` on the
        payload once it has been written.
        """
        pass
