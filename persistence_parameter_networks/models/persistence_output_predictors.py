import torch
import torch.nn as nn
from typing import Tuple, Optional, Union

from abc import ABC, abstractmethod
from persistence_parameter_networks.models.abs_raw_predictor import AbsRawPredictor
from persistence_parameter_networks.models.attention_based_predictor import MultiMaskObjectConditionedCrossAttnPredictor, build_pooler, PoolerConfig
from persistence_parameter_networks.models.detr_ish_based_predictor import DETRLikeMaskConditionedPersistencePredictor
from persistence_parameter_networks.models.dual_encoder_framework import DualEncoderModel, InstancePreprocessConfig, GenericCombinerFactory
from persistence_parameter_networks.models.img_encoders.selavpr.rich_img_encoder import RichOutputSelaEncoder, RichOutputSelaEncoderArgs



# import optional
from typing import Optional


# Abstract base class for raw parameter predictors
class PersistenceOutputPredictor(nn.Module, ABC):
    def __init__(self, baseModel: AbsRawPredictor, name: str, confidenceFromLogits: bool = False):
        super(PersistenceOutputPredictor, self).__init__()
        self.baseModel = baseModel
        # Assert that the base model's output legnth is 3 (y50, yr, yc)
        if self.baseModel.outputLength != 3:
            raise ValueError(f"Base model outputLength must be 3 (y50, yr, yc). Got {self.baseModel.outputLength}")
        self.confidence_from_logits = confidenceFromLogits
        self.model_name = name


    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Abstract method to predict raw parameters from image and mask
        # Should be implemented by subclasses
        # Should output a tensor of shape (batch_size, outputLength)
        # Output tensor should be real-valued without activation functions applied
        rawPredictorOutputs = self.baseModel(image, mask)

        y50_hat, yr_hat, yc_hat = rawPredictorOutputs[:, 0], rawPredictorOutputs[:, 1], rawPredictorOutputs[:, 2]

        if (not self.confidence_from_logits):
            yc_hat = torch.sigmoid(yc_hat)

        return y50_hat, yr_hat, yc_hat


class PersistenceObjConditionedCrossAttn(PersistenceOutputPredictor):
    def __init__(
        self,
        backbone_cfg: RichOutputSelaEncoderArgs,
        pooler_cfg: PoolerConfig,
        attn_dim: int | None = None,
        num_heads: int = 8,
        ff_hidden: int = 512,
        confidenceFromLogits: bool = False,
        name: str = "obj_conditioned_cross_attn",
    ):

        # Build backbone from config
        backbone = RichOutputSelaEncoder(backbone_cfg)

        # Build pooler
        if pooler_cfg is not None:
            pooler = build_pooler(pooler_cfg)
            composed_name = f"{name}__{pooler_cfg.kind}"
        else:
            composed_name = name

        super().__init__(
            baseModel=MultiMaskObjectConditionedCrossAttnPredictor(
                backbone=backbone,
                outputLength=3,
                pooler=pooler,
                attn_dim=attn_dim,
                num_heads=num_heads,
                ff_hidden=ff_hidden,
            ),
            confidenceFromLogits=confidenceFromLogits,
            name=composed_name,
        )


class PersistenceOutDetrLikePredictor(PersistenceOutputPredictor):
    """
    - mask: (B,1,H,W) or (B,K,1,H,W)
    - output: (B,outputLength) or (B,K,outputLength)
    """
    def __init__(
        self,
        backbone_cfg: RichOutputSelaEncoderArgs,
        d_model: int | None = None,
        nhead: int = 8,
        num_layers: int = 3,
        dim_ff: int = 1024,
        dropout: float = 0.0,
        interact_queries: bool = True,
        mask_resize_mode: str = "bilinear",
        init_from_mask_pool: bool = True,
        learned_query_fallback: bool = True,
        max_queries: int = 32,
        eps: float = 1e-6,
        confidenceFromLogits: bool = False,
        name: str = "detr_based",
    ):
        # Build backbone from config (same class as other predictor)
        backbone = RichOutputSelaEncoder(backbone_cfg)

        super().__init__(
            baseModel=DETRLikeMaskConditionedPersistencePredictor(
                backbone=backbone,
                outputLength=3,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_ff=dim_ff,
                dropout=dropout,
                interact_queries=interact_queries,
                mask_resize_mode=mask_resize_mode,
                init_from_mask_pool=init_from_mask_pool,
                learned_query_fallback=learned_query_fallback,
                max_queries=max_queries,
                eps=eps,
            ),
            confidenceFromLogits=confidenceFromLogits,
            name=name,
        )


class PersistenceOutDualEncoderModel(PersistenceOutputPredictor):


    def __init__(
        self,
        context_encoder_args: RichOutputSelaEncoderArgs,
        instance_encoder_args: RichOutputSelaEncoderArgs,
        combiner_factory: GenericCombinerFactory,
        use_unified_encoder: bool = False,
        downstream_dim: int = 512,
        preprocess_cfg: Optional[InstancePreprocessConfig] = None,
        confidenceFromLogits: bool = False,
        name: str = "dual_encoder",
    ):
        context_encoder, instance_encoder = self.buildEncoders(context_encoder_args, instance_encoder_args, use_unified_encoder)
        super().__init__(
            baseModel=DualEncoderModel(
                context_encoder=context_encoder,
                instance_encoder=instance_encoder,
                use_unified_encoder=use_unified_encoder,
                outputLength=3,
                combiner_factory=combiner_factory,
                downstream_dim=downstream_dim,
                preprocess_cfg=preprocess_cfg,
            ),
            confidenceFromLogits=confidenceFromLogits,
            name=name,
        )

    @staticmethod
    def buildEncoders(
        context_encoder_args: RichOutputSelaEncoderArgs,
        instance_encoder_args: RichOutputSelaEncoderArgs,
        use_unified_encoder: bool = False,
    ) -> Tuple[RichOutputSelaEncoder, RichOutputSelaEncoder]:
        if (not context_encoder_args.return_global_only):
            # Override to ensure return_global_only is True for encoders used in this predictor, since the predictor relies on global features. This is a bit hacky but allows us to reuse the same encoder class with a simple flag.
            print("Warning: Overriding context_encoder_args.return_global_only to True for use in PersistenceOutDualEncoderModel since the predictor relies on global features. Original value was:", context_encoder_args.return_global_only)
            context_encoder_args.return_global_only = True
        if use_unified_encoder:
            # If using a unified encoder, we ignore instance_encoder_args and build one encoder for both
            encoder = RichOutputSelaEncoder(context_encoder_args)  # Use context args for the unified encoder
            return encoder, encoder

        if (not instance_encoder_args.return_global_only):
            print("Warning: Overriding instance_encoder_args.return_global_only to True for use in PersistenceOutDualEncoderModel since the predictor relies on global features. Original value was:", instance_encoder_args.return_global_only)
            instance_encoder_args.return_global_only = True

        context_encoder = RichOutputSelaEncoder(context_encoder_args)
        instance_encoder = RichOutputSelaEncoder(instance_encoder_args)
        return context_encoder, instance_encoder

