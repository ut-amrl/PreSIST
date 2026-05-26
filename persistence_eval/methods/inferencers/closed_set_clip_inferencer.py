from typing import Optional

from persistence_eval.core.domain import ImageMaskGroup, PersistenceParameterEstimate
from persistence_eval.methods.inferencers.base import AbsPersistenceParameterInferencer
from persistence_eval.methods.inferencers.closed_set_clip_align.closed_set_clip_embedder import CLIPAligner
from persistence_eval.methods.inferencers.closed_set_semantics.semantic_class_persistence_param_lookup import SemanticClassPersistenceParamLookup


class ClosedSetCLIPInferencer(AbsPersistenceParameterInferencer):
    def __init__(
        self,
        clipAligner: CLIPAligner,
        semanticClassPersistenceParamLookup: SemanticClassPersistenceParamLookup,
        name: str | None = None,
    ) -> None:
        self.clipAligner = clipAligner
        self.semanticClassPersistenceParamLookup = semanticClassPersistenceParamLookup

    def infer(
        self, group: ImageMaskGroup, *, one_off: bool = False,
    ) -> tuple[list[PersistenceParameterEstimate], Optional[float]]:
        id_to_class = self.clipAligner.class_list.id_to_class
        estimates = []
        for i in range(len(group.masks)):
            class_id = self.clipAligner.find_closest_class(group.image, group.masks[i])
            shared = self.semanticClassPersistenceParamLookup.get_params_for_id(class_id)
            # Copy so we don't mutate the cached lookup entry shared across episodes.
            estimate = PersistenceParameterEstimate(
                params=shared.params,
                confidence=shared.confidence,
                metadata={
                    **shared.metadata,
                    "anchor_class_id": int(class_id),
                    "anchor_class_name": id_to_class.get(class_id, str(class_id)),
                },
            )
            estimates.append(estimate)
        return estimates, None
