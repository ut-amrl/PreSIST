"""Dense-prediction variant of :mod:`rich_output_network`.

This module is a copy/sibling of ``rich_output_network.py`` that exposes the
4 intermediate ViT layers needed by a DPT-style decoder. The originals are
kept untouched.

Only the bits that the dense path actually uses are retained:
  - ``DenseGeoLocalizationNet``: wraps the DINOv2 backbone and returns a
    list of 4 intermediate ``(B, 1024, H, W)`` token maps.

Everything else from the original network (``GeM`` / ``L2Norm`` /
``LocalAdapt`` / global-only aggregation) is intentionally absent — those
were features of the per-mask cross-attention model and are not needed
here.

The SELA adapters living inside each DINOv2 block are preserved
automatically: ``get_intermediate_layers`` runs the same block stack as
the standard forward, so adapter contributions are baked into every
tapped layer's output.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

# get_backbone (and the ViT it builds) are unchanged between per-mask and
# dense usage, so we import rather than duplicate.
from ..rich_output_network import get_backbone


class DenseGeoLocalizationNet(nn.Module):
    """Dense variant: surface multiple intermediate ViT layers.

    Returns dict:
      - ``intermediate_token_maps``: list of 4 ``(B, 1024, H, W)`` tensors,
        sampled from blocks indexed by ``intermediate_layer_indices``.
      - ``grid_hw``: ``(H, W)`` of the patch grid.

    The 4 chosen layers (default: ``[5, 11, 17, 23]`` for ViT-L/24) give a
    DPT decoder access to features from across the depth of the backbone.
    """

    def __init__(
        self,
        args,
        intermediate_layer_indices: Sequence[int] = (5, 11, 17, 23),
    ):
        super().__init__()
        self.backbone = get_backbone(args)
        self.features_dim = int(getattr(args, "features_dim", 1024))
        self.intermediate_layer_indices = list(intermediate_layer_indices)

        # Same img_hw / patch_size inference logic as the original
        # GeoLocalizationNet, kept for any downstream consumer that needs
        # it. Not strictly required for the DPT path.
        self.patch_size = getattr(self.backbone, "patch_size", None)
        img_size = getattr(self.backbone, "img_size", None)
        if isinstance(img_size, int):
            self.img_hw = (img_size, img_size)
        elif isinstance(img_size, (tuple, list)) and len(img_size) == 2:
            self.img_hw = (int(img_size[0]), int(img_size[1]))
        else:
            im = getattr(args, "image_size", None)
            if isinstance(im, int):
                self.img_hw = (im, im)
            elif isinstance(im, (tuple, list)) and len(im) == 2:
                self.img_hw = (int(im[0]), int(im[1]))
            else:
                self.img_hw = None

    def forward(self, x: torch.Tensor) -> dict:
        # get_intermediate_layers(n=list, reshape=True, norm=True) returns
        # a tuple of (B, C, H, W) tensors — one per requested block index,
        # with the same LayerNorm applied as the final-layer output.
        intermediate = self.backbone.get_intermediate_layers(
            x,
            n=self.intermediate_layer_indices,
            reshape=True,
            norm=True,
        )
        # All four maps share the same spatial shape.
        _, _, H, W = intermediate[0].shape
        return {
            "intermediate_token_maps": list(intermediate),
            "grid_hw": (H, W),
        }
