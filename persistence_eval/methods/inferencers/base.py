from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from persistence_eval.core.domain import ImageMaskGroup, PersistenceParameterEstimate


class AbsPersistenceParameterInferencer(ABC):
    """
    Estimates persistence parameters for all objects in an ImageMaskGroup.
    Returns one PersistenceParameterEstimate per mask, in the same order as
    group.track_ids. group.masks is (K, H, W) bool.
    """

    @abstractmethod
    def infer(self, group: ImageMaskGroup) -> list[PersistenceParameterEstimate]: ...

    def set_run_dir(self, run_dir: Path) -> None:
        """Called by InferenceRunner before iteration begins. Override to opt in."""
