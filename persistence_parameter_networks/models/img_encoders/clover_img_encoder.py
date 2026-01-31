import torch
import torch.nn.functional as F
from torch import nn

from persistence_parameter_networks.img_encoders.selavpr.encoder import SelaEncoder, SelaEncoderArgs
from persistence_parameter_networks.img_encoders.abs_img_encoder import AbsImgEncoder


class CLOVER_Net(AbsImgEncoder):
    def __init__(
        self,
        sela_args: SelaEncoderArgs,
        post_process_mlp_layers: int = 1,
        feat_length: int = 384,
        post_process_mlp_change_len_last_layer: bool = False,
        use_post_process_head: bool = True,
        **kwargs,
    ):
        super().__init__()
        base_encoder = SelaEncoder(sela_args)

        if not use_post_process_head:
            assert feat_length == base_encoder.getOutputDim(), (
                "Base encoder output length must be equal to feat_length if use_post_process_head is False"
            )

        self.base_encoder = base_encoder

        self.feat_dim = feat_length

        self.base_encoder_out_dim = base_encoder.getOutputDim()

        self.post_process_head = nn.Identity()

        if use_post_process_head:
            post_process_head_layers = []

            if post_process_mlp_change_len_last_layer:
                for _ in range(post_process_mlp_layers):
                    post_process_head_layers.append(
                        nn.Linear(self.base_encoder_out_dim, self.base_encoder_out_dim)
                    )
                    post_process_head_layers.append(nn.ReLU(inplace=True))
                post_process_head_layers.append(nn.Linear(self.base_encoder_out_dim, feat_length))
            else:
                post_process_head_layers.append(nn.Linear(self.base_encoder_out_dim, feat_length))
                for _ in range(post_process_mlp_layers):
                    post_process_head_layers.append(nn.ReLU(inplace=True))
                    post_process_head_layers.append(nn.Linear(feat_length, feat_length))

            # Initialize the layers in the heads
            # Use Kaiming normal initialization for the weights and 0 for the biases
            for layer in post_process_head_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    nn.init.zeros_(layer.bias)

            self.post_process_head = nn.Sequential(*post_process_head_layers)

        self.default_frozen_params = {}

        for name, param in self.post_process_head.named_parameters():
            if not param.requires_grad:
                self.default_frozen_params["post_process_head"] = name

        for name, param in self.base_encoder.named_parameters():
            if not param.requires_grad:
                self.default_frozen_params["base_encoder"] = name

    def freezeWeights(self, freeze_config):
        if freeze_config.freeze_base_encoder:
            print("Freezing base encoder ")
        else:
            print("Unfreezing base encoder (maintaining default frozen weights)")
        for name, param in self.base_encoder.named_parameters():
            if freeze_config.freeze_base_encoder:
                param.requires_grad = False
            else:
                if ("base_encoder" not in self.default_frozen_params) or (
                    name not in self.default_frozen_params["base_encoder"]
                ):
                    param.requires_grad = True

        if freeze_config.freeze_post_process_head:
            print("Freezing post process head")
            for name, param in self.post_process_head.named_parameters():
                param.requires_grad = False
        else:
            print("Unfreezing post process head (maintaining default frozen weights)")
            for name, param in self.post_process_head.named_parameters():
                if ("post_process_head" not in self.default_frozen_params) or (
                    name not in self.default_frozen_params["post_process_head"]
                ):
                    param.requires_grad = True


    def getOutputDim(self):
        return self.feat_dim


    def forward(self, x):

        content_vec = self.post_process_head(self.base_encoder(x))

        return content_vec