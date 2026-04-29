from abc import ABC, abstractmethod
from typing import List, Tuple

from dataset_generation.intermediate_results_logger.debug_data import DebugPayload

# PIL.Image for type hints; avoid hard dependency
try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None


class AbsVLMQuerier(ABC):
    """
    Abstract Vision-Language Model querier.
    Parallel to AbsLLMQuerier but accepts a prompt and a list of images.
    """

    def __init__(self, name: str):
        self.name = name

    def getName(self) -> str:
        return self.name

    @abstractmethod
    def queryVLM(
        self, prompt: str, images: List["PILImage.Image"]
    ) -> Tuple[str, DebugPayload]:
        """
        Send a single prompt with attached images to the VLM and return the
        model's reply along with a DebugPayload of any per-call debug data
        (e.g. reasoning traces). Subclasses with no debug data should return
        (response, DebugPayload.empty()).

        Args:
            prompt: Text prompt (may include instructions; placeholder tokens
                for images are handled by the implementation).
            images: List of PIL Images in the order they appear in the prompt.

        Returns:
            Tuple of (decoded response text, DebugPayload).
        """
        pass
