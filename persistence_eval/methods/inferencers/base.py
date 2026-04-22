from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from persistence_eval.core.domain import AnchorObservation, PersistenceParameterEstimate


class AbsPersistenceParameterInferencer(ABC):
    @abstractmethod
    def infer(self, anchor: AnchorObservation) -> PersistenceParameterEstimate: ...

    def infer_batch(
        self,
        image: np.ndarray,
        masks: list[np.ndarray],
        anchor_time: float,
        track_ids: list[str],
    ) -> list[PersistenceParameterEstimate]:
        """
        Infer for multiple masks on the same image in one pass.
        Default implementation calls infer() sequentially; subclasses may override
        to exploit shared image encoding and reduce overhead.
        """
        results = []
        for mask, track_id in zip(masks, track_ids):
            anchor = AnchorObservation(image=image, mask=mask, anchor_time=anchor_time, track_id=track_id)
            results.append(self.infer(anchor))
        return results
