

from abc import ABC, abstractmethod
from typing import Tuple

from dataset_generation.intermediate_results_logger.debug_data import DebugPayload


class AbstractSceneLevelCaptioner(ABC):

    def __init__(self, name: str):
        self.name = name

    def getName(self) -> str:
        return self.name

    @abstractmethod
    def generateCaption(self, image, image_id: str) -> Tuple[str, DebugPayload]:
        """Returns (caption, DebugPayload).

        Subclasses obtain a DebugPayload via
        ``DebugPayloadFactory.instance().create("scene_caption", image_id)``,
        add entries to it, and return it alongside the caption. The caller
        (the logger) is responsible for calling ``flush()`` on the payload
        once it has been written.
        """
        pass
