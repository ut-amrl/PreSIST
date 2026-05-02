from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from instance_segmentation.abs_instance_segmenter import SegmentationResults


class AbsVLMPromptHandler(ABC):
    """
    Abstract prompt handler for per-object VLM persistence estimation.
    Generates one prompt per object (no object id in the prompt; id is assigned at aggregation time).
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def buildPromptAndImagesForObject(
        self,
        scene_caption: str,
        object_descriptions: Dict[Any, str],
        relationship_descriptions: Dict[Any, Any],
        time_and_date: Optional[str],
        image: torch.Tensor,
        instances: SegmentationResults,
        object_id: str,
    ) -> Tuple[str, List[Image.Image]]:
        """
        Build the per-object prompt and the corresponding image list to send to the VLM.

        Implementations are responsible for any image preparation (e.g. creating a masked/highlighted
        view for the object of interest) and for deciding whether/how to incorporate other object
        descriptions into the prompt.
        """
        pass

    @abstractmethod
    def parseSingleObjectPersistenceResponse(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the VLM response for a single object into one persistence-parameter dict.
        Keys should include expected_persistence_hours, persistence_uncertainty,
        static_likelihood, notes. The integrating class will add "id" when aggregating.
        """
        pass
