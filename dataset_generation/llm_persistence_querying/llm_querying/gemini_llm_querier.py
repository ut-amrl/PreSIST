"""Gemini text-only LLM querier. Thin wrapper over ``GeminiChatEngine``; the
shared singleton in ``gemini._shared`` owns the API client / cumulative token
tracker / snapshot file, so this class only carries per-querier knobs (model,
sampling, thinking config).

Drop-in replacement for ``GemmaLLMQuerier`` from the persistence-querier
caller's perspective: same ``AbsLLMQuerier.queryLLM`` signature, same
``(text, DebugPayload)`` return shape, so it plugs straight into
``ChainedPromptQueryPersistenceQuerier`` without changes to that class.
"""
from __future__ import annotations

from time import time
from typing import Any, Dict, List, Optional, Tuple

from dataset_generation.intermediate_results_logger.debug_data import DebugPayload
from dataset_generation.llm_persistence_querying.gemini.gemini_chat_engine import (
    GeminiChatEngine,
)
from dataset_generation.llm_persistence_querying.llm_querying.abs_llm_querier import (
    AbsLLMQuerier,
)


class GeminiLLMQuerier(AbsLLMQuerier):
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        model_name: str = "gemini-2.5-pro",
        thinking_config: Optional[Any] = None,
        disable_thinking: bool = False,
        extra_generation_config: Optional[Dict[str, Any]] = None,
        allow_unvetted_model: bool = False,
        max_new_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        if name is None:
            raise ValueError("name must be provided for GeminiLLMQuerier")
        super().__init__(name=name)

        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        # Process-wide Gemini state (client, API key, usage tracker, snapshot
        # file) is owned by gemini._shared. Configure that once at process
        # startup via ``configure_gemini_shared(...)`` — if not configured by
        # then, this constructor lazily applies env-var defaults.
        self._engine = GeminiChatEngine(
            model_name=model_name,
            thinking_config=thinking_config,
            disable_thinking=disable_thinking,
            extra_generation_config=extra_generation_config,
            allow_unvetted_model=allow_unvetted_model,
        )

        self.totalQueryTime = 0.0
        self.numQueries = 0
        print(f"Using GeminiLLMQuerier model={self.model_name}")

    # Compatibility no-op: GemmaLLMQuerier exposes initialize() and some
    # callers may invoke it; for Gemini the engine is constructed eagerly.
    def initialize(self) -> None:
        return None

    def wants_warmup(self) -> bool:
        """Gemini is HTTP-only; the first call only pays for a TCP/TLS
        handshake (~hundreds of ms) — negligible vs. the seconds-long Gemini
        decode latency, and not worth one real billed API call to amortize.
        """
        return False

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
        self.totalQueryTime += time() - start_time
        self.numQueries += 1

        debug = DebugPayload()
        debug.add("prompt", prompt, kind="text")
        debug.add("raw_response", result.text, kind="text")
        if result.reasoning:
            debug.add("reasoning_trace", result.reasoning, kind="text")
        return result.text, debug
