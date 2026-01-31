from abc import ABC, abstractmethod

import torch
from torch import nn

class AbsImgEncoder(ABC, nn.Module):
    """Abstract class for image encoders."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the image encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Encoded feature tensor of size (B, getOutputDim()).
        """
        pass


    def getOutputDim(self) -> int:
        """Get the length of the output feature vector.

        Returns:
            int: Length of the output feature vector.
        """
        pass