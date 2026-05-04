from __future__ import annotations

from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from dataset_generation.intermediate_results_logger.debug_data import DebugPayload
from dataset_generation.llm_persistence_querying.gemma.gemma_chat_engine import (
    GemmaChatEngine,
)
from dataset_generation.llm_persistence_querying.llm_querying.abs_llm_querier import AbsLLMQuerier


class GemmaLLMQuerier(AbsLLMQuerier):
    """Gemma 4 text-only querier. Thin wrapper over GemmaChatEngine.

    `enable_thinking=True` requires the vLLM server to be launched with
    `--reasoning-parser gemma4` (see start_gemma_vllm_server.sh --thinking).
    """

    def __init__(
        self,
        *,
        use_vllm_server: bool = True,
        model_name: str = "google/gemma-4-31B-it",
        name: Optional[str] = None,
        enable_thinking: bool = False,
        device_map: Union[str, Dict[str, Any]] = "auto",
        torch_dtype: Union[torch.dtype, str] = "auto",
        vllm_url_root: str = "http://localhost",
        vllm_port: int = 9008,
        vllm_url_suffix: str = "/v1",
        vllm_api_key: str = "EMPTY",
        vllm_request_timeout_s: float = 120.0,
        vllm_extra_request_params: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        if name is None:
            raise ValueError("name must be provided for GemmaLLMQuerier")
        super().__init__(name=name)

        self.model_name = model_name
        self.use_vllm_server = use_vllm_server
        self.enable_thinking = bool(enable_thinking)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        self._engine_kwargs: Dict[str, Any] = dict(
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
        self._engine: Optional[GemmaChatEngine] = None

        self.totalQueryTime = 0.0
        self.numQueries = 0
        print(
            f"Using GemmaLLMQuerier with use_vllm_server={self.use_vllm_server}, "
            f"enable_thinking={self.enable_thinking}"
        )

    def initialize(self):
        if self._engine is None:
            self._engine = GemmaChatEngine(**self._engine_kwargs)

    def queryLLM(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[str, DebugPayload]:
        self.initialize()
        assert self._engine is not None

        start_time = time()
        result = self._engine.chat(
            prompt=prompt,
            images=None,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p if top_p is not None else self.top_p,
            stop=stop,
            seed=seed,
        )
        query_time = time() - start_time
        self.totalQueryTime += query_time
        self.numQueries += 1

        debug = DebugPayload()
        debug.add("prompt", prompt, kind="text")
        debug.add("raw_response", result.text, kind="text")
        if result.reasoning:
            debug.add("reasoning_trace", result.reasoning, kind="text")
        return result.text, debug
