"""Class-keyed lookup inferencer with a fixed class id (no CLIP).

For datasets where every track belongs to a single semantic class (e.g. PKLot
where every tracked object is a parking-spot car), this inferencer skips the
CLIP alignment step and returns the params for a single hardcoded class id
read from a class-keyed JSON cache.
"""

from __future__ import annotations

from typing import Optional

from persistence_eval.core.domain import ImageMaskGroup, PersistenceParameterEstimate
from persistence_eval.methods.inferencers.base import AbsPersistenceParameterInferencer
from persistence_eval.methods.inferencers.closed_set_semantics.semantic_class_persistence_param_lookup import (
    SemanticClassPersistenceParamLookup,
)


class ClassKeyedLookupInferencer(AbsPersistenceParameterInferencer):
    def __init__(
        self,
        class_lookup_file: str,
        class_id: int = 0,
        class_name: str | None = None,
        name: str | None = None,
    ) -> None:
        self._lookup = SemanticClassPersistenceParamLookup(params_file=class_lookup_file)
        self._class_id = class_id
        self._class_name = class_name

    def infer(
        self, group: ImageMaskGroup, *, one_off: bool = False,
    ) -> tuple[list[PersistenceParameterEstimate], Optional[float]]:
        shared = self._lookup.get_params_for_id(self._class_id)
        meta_extra = {"anchor_class_id": int(self._class_id)}
        if self._class_name is not None:
            meta_extra["anchor_class_name"] = self._class_name
        estimates = [
            PersistenceParameterEstimate(
                params=shared.params,
                confidence=shared.confidence,
                metadata={**shared.metadata, **meta_extra},
            )
            for _ in group.track_ids
        ]
        return estimates, None
