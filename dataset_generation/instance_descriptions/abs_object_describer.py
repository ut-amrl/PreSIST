

from abc import ABC, abstractmethod
from typing import Tuple

from dataset_generation.intermediate_results_logger.debug_data import DebugPayload


class AbstractObjectDescriber(ABC):

    def __init__(self, name):
        self.name = name

    def getName(self):
        return self.name

    @abstractmethod
    def describeObjects(self, image, instances, image_id: str) -> Tuple[dict, DebugPayload]:
        """Returns (descriptions, DebugPayload).

        Subclasses obtain a DebugPayload via
        ``DebugPayloadFactory.instance().create("object_descriptions", image_id)``,
        add entries to it, and return it alongside the descriptions. The
        caller (the logger) is responsible for calling ``flush()`` on the
        payload once it has been written.
        """
        pass
