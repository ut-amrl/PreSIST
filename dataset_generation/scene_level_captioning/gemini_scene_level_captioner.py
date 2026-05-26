"""
Gemini scene-level captioner. Thin wrapper over GeminiChatEngine. Mirrors
GemmaVLSceneLevelCaptioner: tensor -> PIL -> one chat call -> caption, with a
one-shot downscale retry when ``finish_reason == "MAX_TOKENS"``.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple, Union

import torch
from PIL import Image

from dataset_generation.intermediate_results_logger.debug_data import (
    DebugPayload,
    DebugPayloadFactory,
)
from dataset_generation.llm_persistence_querying.gemini.gemini_chat_engine import (
    GeminiChatEngine,
    TokenUsage,
)
from dataset_generation.scene_level_captioning.abs_scene_level_captioner import (
    AbstractSceneLevelCaptioner,
)
from utils.timing import timer


_LENGTH_FINISH_REASONS = frozenset({"MAX_TOKENS", "LENGTH"})


# Inlined copies of helpers from qwen_object_discoverer to avoid pulling in
# the qwen-vl-utils dep at import time. Identical behavior.
def _chw_uint8_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 3 or image.shape[0] not in (3, 4):
        raise ValueError("Expected image (C,H,W) with C in {3,4}.")
    if image.dtype != torch.uint8:
        raise ValueError("Expected image dtype=torch.uint8.")
    img3 = image[:3].permute(1, 2, 0).contiguous().cpu().numpy()
    return Image.fromarray(img3)


def _downscale_for_retry(pil: Image.Image, max_side: int = 768) -> Image.Image:
    w, h = pil.size
    scale = min(max_side / max(w, h), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if (new_w, new_h) == (w, h):
        return pil
    return pil.resize((new_w, new_h), Image.Resampling.LANCZOS)


class GeminiSceneLevelCaptioner(AbstractSceneLevelCaptioner):
    """Gemini scene-level captioner. Wraps GeminiChatEngine for a single
    text+image chat per scene. Retries once on token-budget truncation with a
    downscaled image.
    """

    def __init__(
        self,
        *,
        name: str,
        model_name: str = "gemini-2.5-pro",
        thinking_config: Optional[Any] = None,
        disable_thinking: bool = False,
        extra_generation_config: Optional[Dict[str, Any]] = None,
        allow_unvetted_model: bool = False,
        prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        max_new_tokens: int = 16384,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        super().__init__(name=name)

        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        if (prompt is not None) and (prompt_file is not None):
            raise ValueError("Only one of prompt or prompt_file should be provided.")
        if prompt_file is not None:
            with open(prompt_file, "r") as f:
                self.prompt = f.read()
        elif prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = "Describe the scene in detail."

        self._per_query_timer_name = "Timer::gemini_scene_captioner_vlm_query"

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

    def wants_warmup(self) -> bool:
        """Gemini is HTTP-only; the first call only pays for a TCP/TLS
        handshake (~hundreds of ms) — negligible vs. the seconds-long Gemini
        decode latency, and not worth one real billed API call to amortize.
        """
        return False

    def _caption_once(self, pil: Image.Image):
        with timer.time_block(self._per_query_timer_name):
            return self._engine.chat(
                prompt=self.prompt,
                images=[pil],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

    def generateCaption(
        self, image: torch.Tensor, image_id: str
    ) -> Tuple[str, DebugPayload]:
        if image.ndim == 3:
            pil = _chw_uint8_to_pil(image)
        elif image.ndim == 4:
            pil = _chw_uint8_to_pil(image[0])
        else:
            raise ValueError(f"Expected image with 3 or 4 dims, got {image.ndim}")

        result = self._caption_once(pil)

        if result.finish_reason in _LENGTH_FINISH_REASONS:
            max_img_size = max(512, max(pil.width, pil.height) // 4)
            retry_pil = _downscale_for_retry(pil, max_side=max_img_size)
            print("[gemini_scene_captioner] output truncated; retrying with downscaled image.")
            result = self._caption_once(retry_pil)
            if result.finish_reason in _LENGTH_FINISH_REASONS:
                raise RuntimeError(
                    "Scene caption generation was cut off even after retrying with a smaller image."
                )

        print(f"GeminiSceneLevelCaptioner caption: {result.text[:200]}")
        debug = DebugPayloadFactory.instance().create("scene_caption", image_id)
        if result.reasoning:
            debug.add("reasoning_trace", result.reasoning, kind="text")
        debug.add("token_usage", json.dumps(asdict(result.usage)), kind="text")
        return result.text, debug
