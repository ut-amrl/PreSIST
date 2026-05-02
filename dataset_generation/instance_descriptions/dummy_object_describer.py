from utils.image_utils import tensor_image_to_pil, mask_to_pil
from instance_segmentation.abs_instance_segmenter import SegmentationResults
from dataset_generation.instance_descriptions.abs_object_describer import AbstractObjectDescriber
from dataset_generation.intermediate_results_logger.debug_data import DebugPayload
import torch
import numpy as np
from typing import Optional, Dict, Tuple


from PIL import Image


class DummyObjectDescriber(AbstractObjectDescriber):
    def __init__(
        self
    ):
        super().__init__("DummyObjectDescriber")



    # ---------- main API ----------

    def describeObjects(
        self,
        image: torch.Tensor,
        instances: SegmentationResults,
        image_id: str,
    ) -> Tuple[Dict[int, str], DebugPayload]:
        """Describe objects in the image based on the provided instances."""
        descriptions: Dict[int, str] = {}
        return descriptions, DebugPayload.empty()
