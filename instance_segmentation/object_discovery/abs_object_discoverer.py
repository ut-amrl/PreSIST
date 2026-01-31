from __future__ import annotations

from typing import List, Optional
import re
import torch


class AbsObjectDiscoverer:
    """Identify prominent/notable objects in a scene as short language descriptions."""

    def __init__(self, name: str):
        self.name = name

    def getName(self) -> str:
        return self.name

    def identify(self, image: torch.Tensor) -> List[str]:
        """
        Args:
            image: torch.Tensor (C,H,W) uint8 RGB(A)

        Returns:
            List[str]: brief object descriptions (noun phrases recommended)
        """
        raise NotImplementedError
