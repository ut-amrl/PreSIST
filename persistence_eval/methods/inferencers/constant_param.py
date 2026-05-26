from __future__ import annotations

from typing import Optional

from persistence_eval.core.domain import ImageMaskGroup, PersistenceParameterEstimate
from persistence_eval.methods.inferencers.base import AbsPersistenceParameterInferencer


class ConstantParamInferencer(AbsPersistenceParameterInferencer):
    """Emits the same fixed parameter dict for every track — no image/mask processing."""

    def __init__(self, **params: float) -> None:
        self._params = dict(params)

    def infer(
        self, group: ImageMaskGroup, *, one_off: bool = False,
    ) -> tuple[list[PersistenceParameterEstimate], Optional[float]]:
        return [PersistenceParameterEstimate(params=self._params) for _ in group.track_ids], None
