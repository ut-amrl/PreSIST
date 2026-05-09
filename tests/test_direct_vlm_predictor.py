"""Unit tests for DirectVLMPredictor + PromptedDirectFutureExistenceVLMQuerier with a mocked VLM."""
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset_generation.intermediate_results_logger.debug_data import DebugPayload
from dataset_generation.llm_persistence_querying.prompted_direct_future_existence_vlm_querier import (
    PromptedDirectFutureExistenceVLMQuerier,
)
from dataset_generation.llm_persistence_querying.vlm_querying.abs_vlm_querier import AbsVLMQuerier
from persistence_eval.core.domain import InferenceArtifact, PersistenceParameterEstimate
from persistence_eval.methods.predictors.direct_vlm import DirectVLMPredictor


class MockVLM(AbsVLMQuerier):
    def __init__(self, response: str):
        super().__init__(name="mock")
        self._response = response
        self.last_prompt = None
        self.last_images = None

    def queryVLM(self, prompt, images):
        self.last_prompt = prompt
        self.last_images = images
        return self._response, DebugPayload.empty()


def _make_artifact_with_context(tmpdir: Path) -> InferenceArtifact:
    image_path = tmpdir / "img.png"
    Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8)).save(image_path)

    mask_path = tmpdir / "mask.npz"
    mask = np.zeros((32, 48), dtype=bool)
    mask[8:16, 12:24] = True
    np.savez_compressed(mask_path, mask=mask)

    return InferenceArtifact(
        track_id="t0",
        anchor_time=0.0,
        inference_latency_ms=0.0,
        estimate=PersistenceParameterEstimate(
            params={},
            metadata={
                "track_id": "t0",
                "vlm_context": {
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                },
            },
        ),
    )


def _make_predictor(response: str) -> tuple[DirectVLMPredictor, MockVLM]:
    vlm = MockVLM(response)
    querier = PromptedDirectFutureExistenceVLMQuerier(
        vlmQuerier=vlm,
        prompt_template_file="prompt_files/llm_querier/direct_future_existence.txt",
        mask_style="blur_bg_outline_fg",
    )
    return DirectVLMPredictor(querier=querier), vlm


def test_direct_vlm_predictor_in_order_response():
    response = json.dumps([
        {"dt_seconds": 60.0, "prob": 0.9},
        {"dt_seconds": 600.0, "prob": 0.7},
        {"dt_seconds": 3600.0, "prob": 0.4},
    ])
    predictor, vlm = _make_predictor(response)

    with tempfile.TemporaryDirectory() as tmp:
        artifact = _make_artifact_with_context(Path(tmp))
        query_times = torch.tensor([60.0, 600.0, 3600.0], dtype=torch.float64)
        probs = predictor.predict(artifact, query_times=query_times)

    assert probs.shape == (3,)
    assert torch.allclose(probs, torch.tensor([0.9, 0.7, 0.4], dtype=torch.float64))
    assert "60.000" in vlm.last_prompt and "3600.000" in vlm.last_prompt
    assert len(vlm.last_images) == 2


def test_direct_vlm_predictor_dt_match_when_lengths_differ():
    response = json.dumps([
        {"dt_seconds": 600.0, "prob": 0.65},
        {"dt_seconds": 60.0, "prob": 0.95},
    ])
    predictor, _vlm = _make_predictor(response)

    with tempfile.TemporaryDirectory() as tmp:
        artifact = _make_artifact_with_context(Path(tmp))
        # Three horizons but VLM returned two — third must come back NaN, others
        # must align by closest dt despite reversed order in the response.
        query_times = torch.tensor([60.0, 600.0, 3600.0], dtype=torch.float64)
        probs = predictor.predict(artifact, query_times=query_times)

    assert math.isclose(probs[0].item(), 0.95)
    assert math.isclose(probs[1].item(), 0.65)
    assert math.isnan(probs[2].item())
