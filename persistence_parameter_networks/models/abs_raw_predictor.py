import torch
import torch.nn as nn
from typing import Tuple

from abc import ABC, abstractmethod

# Abstract base class for raw parameter predictors
class AbsRawPredictor(nn.Module, ABC):
    def __init__(self, outputLength: int):
        super(AbsRawPredictor, self).__init__()
        self.outputLength = outputLength
        self.confidence_from_logits = True

    @abstractmethod
    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Abstract method to predict raw parameters from image and mask
        # Should be implemented by subclasses
        # Should output a tensor of shape (batch_size, outputLength)
        # Output tensor should be real-valued without activation functions applied
        pass




def ensure_bk1hw(mask: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """
    Accepts:
      - (B, 1, H, W)
      - (B, K, 1, H, W)
      - (B, 1, 1, H, W)  (treated as K=1)
    Returns:
      masks: (B, K, 1, H, W)
      multi: True if K>1
    """
    if mask.dim() == 4:
        # (B, 1, H, W) -> (B, 1, 1, H, W)
        return mask.unsqueeze(1), False

    if mask.dim() == 5:
        B, K, one, H, W = mask.shape
        if one != 1:
            raise ValueError(f"Expected mask shape (B,K,1,H,W). Got {tuple(mask.shape)}")
        return mask, (K > 1)

    raise ValueError(f"mask must be (B,1,H,W) or (B,K,1,H,W). Got {tuple(mask.shape)}")
