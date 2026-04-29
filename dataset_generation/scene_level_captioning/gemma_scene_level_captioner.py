from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
from PIL import Image

from dataset_generation.intermediate_results_logger.debug_data import DebugPayload
from dataset_generation.llm_persistence_querying.gemma.gemma_chat_engine import (
    GemmaChatEngine,
)
from dataset_generation.scene_level_captioning.abs_scene_level_captioner import AbstractSceneLevelCaptioner
from instance_segmentation.object_discovery.qwen_object_discoverer import (
    _chw_uint8_to_pil,
    _downscale_for_retry,
)
from utils.timing import querier_timer_suffix, timer


class GemmaVLSceneLevelCaptioner(AbstractSceneLevelCaptioner):
    """
    Gemma 4 scene-level captioner. Thin wrapper over GemmaChatEngine.

    `enable_thinking=True` requires the vLLM server to be launched with
    `--reasoning-parser gemma4` (see start_gemma_vllm_server.sh --thinking).
    On vLLM truncation (`finish_reason == "length"`), retries once with the
    image downscaled. Retains this captioner-local retry path because the
    engine is a single-call helper.
    """

    def __init__(
        self,
        *,
        name: str,
        model_name: str = "google/gemma-4-31B-it",
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        use_vllm_server: bool = True,
        enable_thinking: bool = False,
        torch_dtype: Union[torch.dtype, str] = "auto",
        vllm_url_root: str = "http://localhost",
        vllm_port: int = 9008,
        vllm_url_suffix: str = "/v1",
        vllm_api_key: str = "EMPTY",
        vllm_request_timeout_s: float = 240.0,
        vllm_extra_request_params: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        max_new_tokens: int = 16384,
        num_beams: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        super().__init__(name=name)

        if (torch_dtype == "auto") and use_vllm_server:
            torch_dtype = "bfloat16" if torch.cuda.is_available() else "float32"

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_vllm_server = use_vllm_server
        self.enable_thinking = bool(enable_thinking)
        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = int(num_beams)
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

        if device_map is None:
            resolved_device_map = "auto" if self.device.startswith("cuda") else None
        else:
            resolved_device_map = device_map

        self._per_query_timer_name = (
            "Timer::gemma_scene_captioner_vlm_query"
            + querier_timer_suffix(
                use_vllm_server=use_vllm_server,
                enable_thinking=self.enable_thinking,
            )
        )

        self._engine = GemmaChatEngine(
            model_name=model_name,
            use_vllm_server=use_vllm_server,
            enable_thinking=self.enable_thinking,
            device_map=resolved_device_map if resolved_device_map is not None else "auto",
            torch_dtype=torch_dtype,
            vllm_url_root=vllm_url_root,
            vllm_port=vllm_port,
            vllm_url_suffix=vllm_url_suffix,
            vllm_api_key=vllm_api_key,
            vllm_request_timeout_s=vllm_request_timeout_s,
            vllm_extra_request_params=vllm_extra_request_params,
        )

    def _caption_once(self, pil: Image.Image):
        with timer.time_block(self._per_query_timer_name):
            return self._engine.chat(
                prompt=self.prompt,
                images=[pil],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

    def generateCaption(self, image: torch.Tensor) -> Tuple[str, DebugPayload]:
        if image.ndim == 3:
            pil = _chw_uint8_to_pil(image)
        elif image.ndim == 4:
            pil = _chw_uint8_to_pil(image[0])
        else:
            raise ValueError(f"Expected image with 3 or 4 dims, got {image.ndim}")

        result = self._caption_once(pil)

        if result.finish_reason == "length":
            max_img_size = max(512, max(pil.width, pil.height) // 4)
            retry_pil = _downscale_for_retry(pil, max_side=max_img_size)
            print("[gemma_scene_captioner] output truncated; retrying with downscaled image.")
            result = self._caption_once(retry_pil)
            if result.finish_reason == "length":
                raise RuntimeError(
                    "Scene caption generation was cut off even after retrying with a smaller image."
                )

        print(f"GemmaVLSceneLevelCaptioner caption: {result.text[:200]}")
        debug = DebugPayload()
        if result.reasoning:
            debug.add("reasoning_trace", result.reasoning, kind="text")
        return result.text, debug
