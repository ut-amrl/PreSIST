from abc import ABC, abstractmethod
from typing import List

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
    def queryVLM(self, prompt: str, images: List["PILImage.Image"]) -> str:
        """
        Send a single prompt with attached images to the VLM and return the model's reply.

        Args:
            prompt: Text prompt (may include instructions; placeholder tokens
                for images are handled by the implementation).
            images: List of PIL Images in the order they appear in the prompt.

        Returns:
            Decoded response text.
        """
        pass
