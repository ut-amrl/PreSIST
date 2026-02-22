import torch
import torch.nn as nn

from .rich_output_network import GeoLocalizationNet
from typing import Union


class RichOutputSelaEncoderArgs:
    """
    Mirrors SelaEncoderArgs but for rich_output_network.GeoLocalizationNet.

    Notes:
      - features_dim is set by get_backbone(args) inside rich_output_network (same as network.py).
      - use_local controls whether GeoLocalizationNet returns local_map/local_tokens_hw.
    """
    def __init__(
        self,
        foundation_model_path,
        registers: bool = False,
        resume: bool = False,
        image_size: int = 518,
        freeze_dino: bool = True,
        use_adapter: bool = True,
        pretrained_adapter_path: str | None = None,
        freeze_pre_sela: bool = False,
        use_local: bool = True,
        return_global_only: bool = False,
    ):
        self.foundation_model_path = foundation_model_path
        self.registers = registers
        self.resume = resume
        self.image_size = image_size
        self.freeze_dino = freeze_dino
        self.use_adapter = use_adapter
        self.pretrained_adapter_path = pretrained_adapter_path
        self.freeze_pre_sela = freeze_pre_sela
        self.use_local = use_local
        self.return_global_only = return_global_only


class RichOutputSelaEncoder(nn.Module):
    """

    Compatibility goals:
      - attention_based_predictor.MultiMaskObjectConditionedCrossAttnPredictor expects:
          backbone(image) -> dict with keys like patch_tokens/patch_tokens_hw/grid_hw/...
          backbone.img_hw exists
          backbone.features_dim exists (or can be inferred)
      - This wrapper proxies img_hw/features_dim and returns the dict in forward().
    """
    def __init__(self, sela_args: RichOutputSelaEncoderArgs):
        super().__init__()

        # Build the rich-output backbone
        self.model = GeoLocalizationNet(sela_args, use_local=sela_args.use_local, return_global_only=sela_args.return_global_only)  # Pass use_local to control local outputs

        # Proxy attrs used by attention_based_predictor
        self.img_hw = getattr(self.model, "img_hw", None)
        # Prefer the constructed model's metadata; fall back to args.
        self.features_dim = int(getattr(self.model, "features_dim", getattr(sela_args, "features_dim", 1024)))

        # Freeze DINO (but keep adapters trainable)
        if sela_args.freeze_dino:
            print("Freezing dinov2 backbone")
            for name, param in self.model.backbone.named_parameters():
                if "adapter" not in name:
                    param.requires_grad = False
        else:
            print("Not freezing dinov2 backbone")

        # Initialize adapter (same logic as encoder.py)
        if not sela_args.resume:
            for n, m in self.named_modules():
                if "adapter" in n:
                    for n2, m2 in m.named_modules():
                        if "D_fc2" in n2:
                            if isinstance(m2, nn.Linear):
                                nn.init.constant_(m2.weight, 0.0)
                                nn.init.constant_(m2.bias, 0.0)

        # Optionally load pretrained adapter checkpoint (same key-munging as encoder.py)
        if sela_args.pretrained_adapter_path is not None:
            print("Loading pretrained adapter from: ", sela_args.pretrained_adapter_path)
            ckpt = torch.load(sela_args.pretrained_adapter_path, map_location="cpu")

            # encoder.py expects ckpt["model"]; keep same behavior
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

            # Debug: print "top-level" keys (no periods) like you did
            try:
                for key in state_dict.keys():
                    if "." not in key:
                        print(key)
            except Exception:
                pass

            new_state_dict = {}
            for key, value in state_dict.items():
                # Same prefix stripping you already use:
                # "backbone_network.model." -> ""
                if key.startswith("backbone_network.model."):
                    new_key = key[len("backbone_network.model.") :]
                    new_state_dict[new_key] = value
                else:
                    # Be permissive: also allow direct keys that already match
                    new_state_dict[key] = value

            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            if missing:
                print(f"[RichOutputSelaEncoder] Missing keys when loading adapter: {len(missing)}")
            if unexpected:
                print(f"[RichOutputSelaEncoder] Unexpected keys when loading adapter: {len(unexpected)}")

            if sela_args.freeze_pre_sela:
                print("Freezing pretrained Sela adapter")
                for _, param in self.model.named_parameters():
                    param.requires_grad = False
            else:
                print("Not freezing pretrained Sela adapter")

    def forward(self, x: torch.Tensor) -> Union[dict, torch.Tensor]:
        """
        Return the rich dict output required by attention_based_predictor.
        """
        out = self.model(x)
        return out

    def getOutputDim(self) -> int:
        return int(self.features_dim)
