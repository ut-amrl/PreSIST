import torch
import torch.nn as nn
from typing import Tuple

from abc import ABC, abstractmethod
from persistence_parameter_networks.models.abs_raw_predictor import AbsRawPredictor
from persistence_parameter_networks.models.attention_based_predictor import MultiMaskObjectConditionedCrossAttnPredictor
from persistence_parameter_networks.models.detr_ish_based_predictor import DETRLikeMaskConditionedPersistencePredictor
from persistence_parameter_networks.models.dual_encoder_framework import DualEncoderModel, FeatureCombiner, InstancePreprocessConfig

# import optional
from typing import Optional


# Abstract base class for raw parameter predictors
class PersistenceOutputPredictor(nn.Module, ABC):
    def __init__(self, baseModel: AbsRawPredictor, confidenceFromLogits: bool = False):
        super(PersistenceOutputPredictor, self).__init__()
        self.baseModel = baseModel
        # Assert that the base model's output legnth is 3 (y50, yr, yc)
        if self.baseModel.outputLength != 3:
            raise ValueError(f"Base model outputLength must be 3 (y50, yr, yc). Got {self.baseModel.outputLength}")
        self.confidence_from_logits = confidenceFromLogits


    @abstractmethod
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
        backbone: nn.Module, # TODO replace with geoloc backbone parameters
        pooler: nn.Module,
        attn_dim: int | None = None,
        num_heads: int = 8,
        ff_hidden: int = 512,
        confidenceFromLogits: bool = False,
    ):
        super().__init__(
            baseModel=MultiMaskObjectConditionedCrossAttnPredictor(
                backbone=backbone, # TODO
                outputLength=3,
                pooler=pooler,
                attn_dim=attn_dim,
                num_heads=num_heads,
                ff_hidden=ff_hidden,
            ),
            confidenceFromLogits=confidenceFromLogits)


class PersistenceOutDetrLikePredictor(PersistenceOutputPredictor):
    """
    - mask: (B,1,H,W) or (B,K,1,H,W)
    - output: (B,outputLength) or (B,K,outputLength)
    """
    def __init__(
        self,
        backbone: nn.Module, # TODO replace with geoloc backbone parameters
        d_model: int | None = None,
        nhead: int = 8,
        num_layers: int = 3,
        dim_ff: int = 1024,
        dropout: float = 0.0,
        interact_queries: bool = True,       # <-- constructor argument you asked for
        mask_resize_mode: str = "bilinear",
        init_from_mask_pool: bool = True,
        learned_query_fallback: bool = True,
        max_queries: int = 32,
        eps: float = 1e-6,
        confidenceFromLogits: bool = False,
    ):
        super().__init__(
            baseModel=DETRLikeMaskConditionedPersistencePredictor(
                backbone=backbone, # TODO
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
            confidenceFromLogits=confidenceFromLogits
        )


class PersistenceOutDualEncoderModel(PersistenceOutputPredictor):


    def __init__(
        self,
        context_encoder: nn.Module, # TODO replace with geoloc backbone parameters
        instance_encoder: nn.Module,# TODO replace with geoloc backbone parameters
        combiner: FeatureCombiner | None = None,
        downstream_dim: int = 512,
        preprocess_cfg: Optional[InstancePreprocessConfig] = None,
        head: Optional[nn.Module] = None,
        head_hidden: int = 256,
        head_dropout: float = 0.0,
        confidenceFromLogits: bool = False,
    ):
        super().__init__(
            baseModel=DualEncoderModel(
                context_encoder=context_encoder, # TODO
                instance_encoder=instance_encoder, # TODO
                outputLength=3,
                combiner=combiner,
                downstream_dim=downstream_dim,
                preprocess_cfg=preprocess_cfg,
                head=head,
                head_hidden=head_hidden,
                head_dropout=head_dropout,
            ),
            confidenceFromLogits=confidenceFromLogits
        )

