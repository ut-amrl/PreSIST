from __future__ import annotations

from typing import List, Optional, Tuple
import re
import torch

from dataset_generation.intermediate_results_logger.debug_data import DebugPayload


class AbsObjectDiscoverer:
    """Identify prominent/notable objects in a scene as short language descriptions."""

    def __init__(self, name: str):
        self.name = name

    def getName(self) -> str:
        return self.name

    def identify(self, image: torch.Tensor) -> Tuple[List[str], DebugPayload]:
        """
        Args:
            image: torch.Tensor (C,H,W) uint8 RGB(A)

        Returns:
            Tuple of (list of brief object descriptions, DebugPayload).
            Subclasses with no debug data should return (descs, DebugPayload.empty()).
        """
        raise NotImplementedError
