from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch
from PIL import Image



@dataclass
class LanguagePromptedSegmentationResults:
    """
    Flat outputs for an image + multiple prompts.

    prompt_ids: (N,) int64 - which input prompt produced each mask
    masks:      (N,H,W) bool
    boxes:      (N,4) float32 XYXY
    scores:     (N,) float32 (0 if unavailable)
    """
    prompt_ids: torch.Tensor
    masks: torch.Tensor
    boxes: torch.Tensor
    scores: torch.Tensor


class AbsLanguagePromptedSegmenter:
    """Given an image and language prompts, return 0..K masks per prompt (flattened)."""

    def __init__(self, name: str):
        self.name = name

    def getName(self) -> str:
        return self.name

    def segment_from_prompts(self, image: torch.Tensor, prompts: List[str]) -> LanguagePromptedSegmentationResults:
        raise NotImplementedError

