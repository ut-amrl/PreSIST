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
        instances: SegmentationResults,
        image_id: str,
    ) -> Tuple[Any, Any, Any, DebugPayload]:
        """
        Estimate persistence parameters for each object.

        Returns (prompts, rawResponse, persistenceParameters, DebugPayload).

        Subclasses obtain a DebugPayload via
        ``DebugPayloadFactory.instance().create("persistence", image_id)``,
        add entries to it, and return it alongside the other outputs. The
        caller (the logger) is responsible for calling ``flush()`` on the
        payload once it has been written.

        image: (C, H, W) tensor for the scene.
        instances: SegmentationResults (masks, ids, etc.) for the scene.
        Implementations may use them for vision-based estimation (e.g. VLM with mask highlights).
        """
        pass
