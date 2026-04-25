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

    def infer(self, group: ImageMaskGroup) -> list[PersistenceParameterEstimate]:
        estimates = []
        for i in range(len(group.masks)):
            class_id = self.clipAligner.find_closest_class(group.image, group.masks[i])
            params = self.semanticClassPersistenceParamLookup.get_params_for_id(class_id)
            estimates.append(params)
        return estimates
