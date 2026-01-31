from utils.image_utils import tensor_image_to_pil, mask_to_pil
from instance_segmentation.abs_instance_segmenter import SegmentationResults
from dataset_generation.instance_descriptions.abs_object_describer import AbstractObjectDescriber
import torch
import numpy as np
from typing import Optional, Dict


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
    ) -> Dict[int, str]:
        """
        Describe objects in the image based on the provided instances.

        Args:
            image:
                Input image tensor of shape (C, H, W).
            instances:
                SegmentationResults with:
                    - instances.masks: iterable of per-instance masks (H, W) or (1, H, W)
                    - instances.ids: tensor/list of instance IDs
                    - instances.boxes: tensor of (N, 4) in XYXY format (currently unused)

        Returns:
            dict[obj_id -> description string]
        """
        descriptions: Dict[int, str] = {}


        return descriptions
