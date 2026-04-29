from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import torch

from dataset_generation.intermediate_results_logger.debug_data import DebugPayload
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
    ) -> Tuple[Any, Any, Any, DebugPayload]:
        """
        Estimate persistence parameters for each object.

        Returns (prompts, rawResponse, persistenceParameters, DebugPayload).
        Subclasses with no debug data may return DebugPayload.empty() in the last slot.

        image: (C, H, W) tensor for the scene.
        instances: SegmentationResults (masks, ids, etc.) for the scene.
        Implementations may use them for vision-based estimation (e.g. VLM with mask highlights).
        """
        pass