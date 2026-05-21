"""Dense-prediction variant of :mod:`rich_img_encoder`.

A copy/sibling of the per-mask ``RichOutputSelaEncoder`` that wraps
:class:`DenseGeoLocalizationNet` instead of the original
``GeoLocalizationNet``. The original ``rich_img_encoder.py`` is left
unchanged.

Behavioral parity points worth preserving:
  - Same freeze rule: when ``freeze_dino=True``, only the SELA adapter
    parameters (name contains ``"adapter"``) remain trainable.
  - Same adapter zero-init for ``D_fc2`` layers (matches encoder.py).
  - Same pretrained-adapter loading path (``backbone_network.model.``
    prefix strip), so adapter checkpoints from the per-mask model are
    drop-in here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .dense_rich_output_network import DenseGeoLocalizationNet
from ..rich_img_encoder import RichOutputSelaEncoderArgs


class DenseRichOutputSelaEncoder(nn.Module):
    """Returns a dict containing ``intermediate_token_maps`` (list of 4
    ``(B, 1024, H, W)`` feature maps) for a DPT-style decoder."""

    def __init__(self, sela_args: RichOutputSelaEncoderArgs):
        super().__init__()

        self.model = DenseGeoLocalizationNet(sela_args)

        self.img_hw = getattr(self.model, "img_hw", None)
        self.features_dim = int(
            getattr(self.model, "features_dim", getattr(sela_args, "features_dim", 1024))
        )

        if sela_args.freeze_dino:
            print("Freezing dinov2 backbone")
            for name, param in self.model.backbone.named_parameters():
                if "adapter" not in name:
                    param.requires_grad = False
        else:
            print("Not freezing dinov2 backbone")

        if not sela_args.resume:
            for n, m in self.named_modules():
                if "adapter" in n:
                    for n2, m2 in m.named_modules():
                        if "D_fc2" in n2:
                            if isinstance(m2, nn.Linear):
                                nn.init.constant_(m2.weight, 0.0)
                                nn.init.constant_(m2.bias, 0.0)

        if sela_args.pretrained_adapter_path is not None:
            print(
                "Loading pretrained adapter from: ",
                sela_args.pretrained_adapter_path,
            )
            ckpt = torch.load(sela_args.pretrained_adapter_path, map_location="cpu")

            state_dict = (
                ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            )

            try:
                for key in state_dict.keys():
                    if "." not in key:
                        print(key)
            except Exception:
                pass

            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("backbone_network.model."):
                    new_key = key[len("backbone_network.model.") :]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

            missing, unexpected = self.model.load_state_dict(
                new_state_dict, strict=False
            )
            if missing:
                print(
                    f"[DenseRichOutputSelaEncoder] Missing keys when loading adapter: {len(missing)}"
                )
            if unexpected:
                print(
                    f"[DenseRichOutputSelaEncoder] Unexpected keys when loading adapter: {len(unexpected)}"
                )

            if sela_args.freeze_pre_sela:
                print("Freezing pretrained Sela adapter")
                for _, param in self.model.named_parameters():
                    param.requires_grad = False
            else:
                print("Not freezing pretrained Sela adapter")

    def forward(self, x: torch.Tensor) -> dict:
        return self.model(x)

    def getOutputDim(self) -> int:
        return int(self.features_dim)
