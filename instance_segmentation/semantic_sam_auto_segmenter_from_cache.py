from __future__ import annotations

import os
from typing import Optional, Sequence, Union, List
import numpy as np
import torch
from PIL import Image

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from instance_segmentation.abs_instance_segmenter import AbsInstanceSegmenter, SegmentationResults



# Can be 1-6, float in [0..1], string key, or list of levels
# 1 is least detailed, 6 is most detailed
# if key in ("all", "max", "fine+"):
#     return [1, 2, 3, 4, 5, 6]
# if key in ("fine", "detailed", "parts"):
#     return [2, 3, 4, 5, 6]
# if key in ("medium", "default"):
#     return [2, 3, 4]
# if key in ("coarse", "semantic", "big"):


GranularitySpec = Union[int, float, str, Sequence[int]]


import yaml
import json
import argparse
import logging

logger = logging.getLogger(__name__)



class SemanticSAMInstanceSegmenterFromCache(AbsInstanceSegmenter):
    """
    Semantic-SAM-backed segmenter that:
      1) preprocesses like the repo (short-edge resize to 640),
      2) runs auto mask generation,
      3) maps masks + boxes back to the *original* input resolution.
    """

    def __init__(
        self,
        *,
        checkpoint: str,
        model_type: str,  # typically "T" or "L"
        device: Optional[str] = None,
        granularity: GranularitySpec = "all",
        resize_short_edge: int = 640,
        min_area_perc: float = 0.0,
        name: Optional[str] = None,
        generator_kwargs: Optional[dict] = None,
    ):
        if name is None:
            raise ValueError("A name must be provided for the SemanticSAMInstanceSegmenter.")
        super().__init__(name=name)



    def segmentInstances(self, image: torch.Tensor) -> SegmentationResults:
        # Throw an error if it actually gets here -- this segmenter is only meant to be used with cached results, so segmentInstances should never be called.
        raise RuntimeError("SemanticSAMInstanceSegmenterFromCache is only meant to be used with cached segmentation results. segmentInstances should never be called.")