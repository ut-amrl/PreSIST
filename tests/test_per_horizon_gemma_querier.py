"""Unit test for PerHorizonGemmaDirectFutureExistenceVLMQuerier with a mocked chat engine."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from dataset_generation.llm_persistence_querying.gemma.gemma_chat_engine import GemmaChatResult


def _build_querier(per_horizon_prompt_overrides=None):
    # Patch GemmaChatEngine so the constructor doesn't try to talk to vLLM.
    fake_engine = MagicMock()
    fake_engine.use_vllm_server = False
    with patch(
        "dataset_generation.llm_persistence_querying.gemma_direct_future_existence_vlm_querier.GemmaChatEngine",
        return_value=fake_engine,
    ):
        from dataset_generation.llm_persistence_querying.per_horizon_gemma_direct_future_existence_vlm_querier import (
            PerHorizonGemmaDirectFutureExistenceVLMQuerier,
        )
        kwargs = dict(
            name="test_per_horizon",
            model_name="google/gemma-4-31B-it",
            use_vllm_server=False,
            enable_thinking=False,
            mask_style=None,           # skip overlay rendering — keeps test cheap
            include_full_img_last=False,
            max_concurrency=4,
        )
        if per_horizon_prompt_overrides:
            kwargs.update(per_horizon_prompt_overrides)
        querier = PerHorizonGemmaDirectFutureExistenceVLMQuerier(**kwargs)
    return querier, fake_engine


def _make_image_and_mask():
    image = torch.zeros((3, 24, 32), dtype=torch.uint8)
    mask = torch.zeros((24, 32), dtype=torch.bool)
    mask[6:12, 8:16] = True
    return image, mask


def test_per_horizon_one_request_per_horizon_and_aligned():
    querier, engine = _build_querier()
    # Server returns a JSON object per request — order matches input order.
    engine.chat_many.return_value = [
        GemmaChatResult(text=json.dumps({"prob": 0.95})),
        GemmaChatResult(text=json.dumps({"prob": 0.6})),
        GemmaChatResult(text=json.dumps({"prob": 0.2})),
    ]

    image, mask = _make_image_and_mask()
    probs = querier.query(image, mask, query_times_seconds=[60.0, 600.0, 3600.0])

    assert probs == [0.95, 0.6, 0.2]

    # Exactly one request per horizon, prompts include the rendered horizon.
    args, kwargs = engine.chat_many.call_args
    requests = args[0] if args else kwargs["requests"]
    assert len(requests) == 3
    assert "60.000" in requests[0].prompt and "1.0min" in requests[0].prompt
    assert "600.000" in requests[1].prompt
    assert "3600.000" in requests[2].prompt and "1.00h" in requests[2].prompt
    assert kwargs.get("max_concurrency", None) == querier.max_concurrency


def test_per_horizon_unparseable_response_yields_nan():
    querier, engine = _build_querier()
    engine.chat_many.return_value = [
        GemmaChatResult(text="not json at all"),
        GemmaChatResult(text=json.dumps({"prob": 0.7})),
    ]

    image, mask = _make_image_and_mask()
    probs = querier.query(image, mask, query_times_seconds=[60.0, 600.0])

    import math
    assert math.isnan(probs[0])
    assert probs[1] == 0.7


def test_per_horizon_clamps_and_handles_thinking_prefix():
    querier, engine = _build_querier()
    engine.chat_many.return_value = [
        GemmaChatResult(text="<think>internal reasoning…</think>\n{\"prob\": 1.5}"),
        GemmaChatResult(text="<think>more</think>\n{\"prob\": -0.2}"),
    ]

    image, mask = _make_image_and_mask()
    probs = querier.query(image, mask, query_times_seconds=[60.0, 600.0])

    assert probs[0] == 1.0   # clamped
    assert probs[1] == 0.0   # clamped
