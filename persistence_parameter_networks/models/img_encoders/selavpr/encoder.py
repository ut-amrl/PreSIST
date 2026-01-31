import torch
import torch.nn as nn

from .network import GeoLocalizationNet


class SelaEncoderArgs:
    def __init__(
        self,
        foundation_model_path,
        registers=False,
        resume=False,
        image_size=518,
        freeze_dino=True,
        use_adapter=True,
        pretrained_adapter_path=None,
        freeze_pre_sela=False,
    ):
        self.foundation_model_path = foundation_model_path
        self.registers = registers
        self.resume = resume
        self.image_size = image_size
        self.freeze_dino = freeze_dino
        self.use_adapter = use_adapter
        self.pretrained_adapter_path = pretrained_adapter_path
        self.freeze_pre_sela = freeze_pre_sela


class SelaEncoder(nn.Module):
    def __init__(self, sela_args: SelaEncoderArgs):
        super().__init__()
        self.model = GeoLocalizationNet(sela_args)

        if sela_args.freeze_dino:
            print("Freezing dinov2 backbone")
            for name, param in self.model.backbone.named_parameters():
                if "adapter" not in name:
                    param.requires_grad = False
        else:
            print("Not freezing dinov2 backbone")

        ## initialize Adapter
        if not sela_args.resume:
            for n, m in self.named_modules():
                if "adapter" in n:
                    for n2, m2 in m.named_modules():
                        if "D_fc2" in n2:
                            if isinstance(m2, nn.Linear):
                                nn.init.constant_(m2.weight, 0.0)
                                nn.init.constant_(m2.bias, 0.0)

        if sela_args.pretrained_adapter_path is not None:
            print("Loading pretrained adapter from: ", sela_args.pretrained_adapter_path)
            ckpt = torch.load(sela_args.pretrained_adapter_path, map_location="cpu")
            state_dict = ckpt["model"]
            # print(state_dict.keys())
            # print the keys in the state dict that dont contain a period
            for key in state_dict.keys():
                if "." not in key:
                    print(key)
            # Load the
            # self.model.load_state_dict(torch.load(sela_args.pretrained_adapter_path

            new_state_dict = {}
            for key, value in state_dict.items():
                # Check if the key starts with "backbone." and remove it
                if key.startswith("backbone_network.model."):
                    new_key = key[len("backbone_network.model.") :]  # Remove the "backbone." prefix
                    new_state_dict[new_key] = value
            self.model.load_state_dict(new_state_dict)

            if sela_args.freeze_pre_sela:
                print("Freezing pretrained Sela adapter")
                for name, param in self.model.named_parameters():
                    param.requires_grad = False
            else:
                print("Not freezing pretrained Sela adapter")

    def forward(self, x):
        return self.model(x)[1]

    def getOutputDim(self):
        return 1024
