"""
DirectVLMPredictor: alternative to the survival-function/persistence-filter pair
that asks a VLM directly for P(object exists at anchor + dt) for each query
time. Pure feedforward — no observation history is consumed.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from persistence_eval.core.domain import BinaryObservationEvent, InferenceArtifact
from persistence_eval.datasets._mask_io import load_mask_bool
from persistence_eval.methods.predictors.base import AbsPersistencePredictor
from dataset_generation.llm_persistence_querying.abs_direct_future_existence_vlm_querier import (
    AbsDirectFutureExistenceVLMQuerier,
)
from utils.image_utils import loadImageForDatasetGeneration


class DirectVLMPredictor(AbsPersistencePredictor):
    """
    Reads cached per-instance VLM context out of `artifact.estimate.metadata["vlm_context"]`,
    issues a single VLM call per artifact via the wrapped querier, and returns the
    parsed per-time probabilities.

    The querier abstracts how the VLM is asked (prompt template, backend, image overlays);
    the predictor only handles I/O of cached image + mask paths and exposes the
    AbsPersistencePredictor interface.
    """

    def __init__(
        self,
        querier: AbsDirectFutureExistenceVLMQuerier,
        name: str | None = None,
    ) -> None:
        self._querier = querier
        self.name = name

    def predict(
        self,
        artifact: InferenceArtifact,
        query_times: torch.Tensor,
        observations: list[BinaryObservationEvent] | None = None,
    ) -> torch.Tensor:
        del observations  # pure feedforward variant — observations are ignored
        ctx = artifact.estimate.metadata.get("vlm_context")
        if not ctx:
            raise ValueError(
                "DirectVLMPredictor requires artifact.estimate.metadata['vlm_context']; "
                "did you pair it with DirectVLMQueryInferencer at inference time?"
            )

        image_tensor = self._load_image(ctx["image_path"])
        mask_tensor = self._load_mask(ctx["mask_path"])

        probs = self._querier.query(
            image=image_tensor,
            mask=mask_tensor,
            query_times_seconds=query_times.tolist(),
        )
        return torch.tensor(probs, dtype=query_times.dtype)

    @staticmethod
    def _load_image(path: str) -> torch.Tensor:
        return loadImageForDatasetGeneration(path)

    @staticmethod
    def _load_mask(path: str) -> torch.Tensor:
        return load_mask_bool(path)
