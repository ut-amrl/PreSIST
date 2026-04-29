"""
Gemma 4 VLM querier. Thin wrapper over GemmaChatEngine; sends one
multi-image user message and returns the reply (plus optional reasoning
trace when thinking is enabled).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from dataset_generation.intermediate_results_logger.debug_data import DebugPayload
from dataset_generation.llm_persistence_querying.gemma.gemma_chat_engine import (
    GemmaChatEngine,
)
from dataset_generation.llm_persistence_querying.vlm_querying.abs_vlm_querier import AbsVLMQuerier

logging.getLogger("httpx").setLevel(logging.WARNING)


class GemmaVLMQuerier(AbsVLMQuerier):
    def __init__(
        self,
        *,
        name: str,
        model_name: str = "google/gemma-4-27b-it",
        use_vllm_server: bool = False,
        enable_thinking: bool = False,
        device_map: Union[str, Dict[str, Any]] = "auto",
        torch_dtype: Union[torch.dtype, str] = "auto",
        vllm_url_root: str = "http://localhost",
        vllm_port: int = 9001,
        vllm_url_suffix: str = "/v1",
        vllm_api_key: str = "EMPTY",
        vllm_request_timeout_s: float = 120.0,
        vllm_extra_request_params: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        if name is None:
            raise ValueError("name must be provided for GemmaVLMQuerier")
        super().__init__(name=name)

        self.use_vllm_server = use_vllm_server
        self.enable_thinking = bool(enable_thinking)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        print(
            f"Using GemmaVLMQuerier with use_vllm_server={self.use_vllm_server}, "
            f"enable_thinking={self.enable_thinking}"
        )

        self._engine = GemmaChatEngine(
            model_name=model_name,
            use_vllm_server=use_vllm_server,
            enable_thinking=self.enable_thinking,
            device_map=device_map,
            torch_dtype=torch_dtype,
            vllm_url_root=vllm_url_root,
            vllm_port=vllm_port,
            vllm_url_suffix=vllm_url_suffix,
            vllm_api_key=vllm_api_key,
            vllm_request_timeout_s=vllm_request_timeout_s,
            vllm_extra_request_params=vllm_extra_request_params,
        )

    def queryVLM(
        self, prompt: str, images: List[Image.Image]
    ) -> Tuple[str, DebugPayload]:
        if not images:
            raise ValueError("GemmaVLMQuerier.queryVLM requires at least one image.")
        result = self._engine.chat(
            prompt=prompt,
            images=images,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        debug = DebugPayload()
        if result.reasoning:
            debug.add("reasoning_trace", result.reasoning, kind="text")
        return result.text, debug
