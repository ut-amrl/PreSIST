from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
import torch

from instance_segmentation.abs_instance_segmenter import SegmentationResults


class AbsLLMPersistenceQuerier(ABC):

    def __init__(self, name: str):
        self.name = name

    def getName(self) -> str:
        return self.name

    @abstractmethod
    def estimatePersistenceParameters(
        self,
        sceneCaption: str,
        objectDescriptions: dict,
        relationshipDescriptions: dict,
        timeAndDate: Optional[str],
        image: torch.Tensor,
        instances: SegmentationResults
    ):
        """
        Estimate persistence parameters for each object.

        image: (C, H, W) tensor for the scene.
        instances: SegmentationResults (masks, ids, etc.) for the scene.
        Implementations may use them for vision-based estimation (e.g. VLM with mask highlights).
        """
        pass