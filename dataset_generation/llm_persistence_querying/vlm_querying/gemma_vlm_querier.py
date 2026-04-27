"""
Gemma VLM querier for vision-language persistence prompts.
Parallel to QwenVLMQuerier; uses Gemma 4 (google/gemma-4-*-it) with multiple images per call.

Two modes:
1) In-memory (Transformers): loads model + processor, uses generate().
2) vLLM server (OpenAI-compatible): sends chat completions with image_url parts to vLLM.
Switch via use_vllm_server.
"""
from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Union

from dataset_generation.llm_persistence_querying.utils.llm_querying_utils import (
    derive_served_model_name,
    resolve_served_model_name,
    resolve_torch_dtype,
)
import torch
from PIL import Image

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from dataset_generation.llm_persistence_querying.vlm_querying.abs_vlm_querier import AbsVLMQuerier


def _pil_to_data_url_jpeg(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class GemmaVLMQuerier(AbsVLMQuerier):
    """
    VLM querier using Gemma 4 (google/gemma-4-9b-it or google/gemma-4-27b-it).
    Accepts a prompt and a list of PIL images; sends one multi-image user message and returns the reply.

    Two modes:
    1) use_vllm_server=False (default): load model locally with Transformers.
    2) use_vllm_server=True: use OpenAI-compatible vLLM server (no local model load).
    """

    def __init__(
        self,
        *,
        name: str,
        model_name: str = "google/gemma-4-27b-it",
        # --- mode ---
        use_vllm_server: bool = False,
        # --- local (Transformers) runtime ---
        device_map: Union[str, Dict[str, Any]] = "auto",
        torch_dtype: Union[torch.dtype, str] = "auto",
        # --- vLLM server runtime ---
        vllm_url_root: str = "http://localhost",
        vllm_port: int = 9001,
        vllm_url_suffix: str = "/v1",
        vllm_api_key: str = "EMPTY",
        vllm_request_timeout_s: float = 120.0,
        vllm_extra_request_params: Optional[Dict[str, Any]] = None,
        # --- generation / sampling (both modes) ---
        max_new_tokens: int = 4096,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        if name is None:
            raise ValueError("name must be provided for GemmaVLMQuerier")
        super().__init__(name=name)

        self.model_name = model_name
        self.use_vllm_server = use_vllm_server
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = temperature
        self.top_p = top_p

        self._default_sampling: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        self.vllm_extra_request_params = vllm_extra_request_params or {}

        self.model = None
        self.processor = None
        self.device_map = device_map
        self.vllm_client = None
        self.vllm_base_url = ""
        self._name_resolved = False

        print("Using GemmaVLMQuerier with use_vllm_server =", self.use_vllm_server)

        if self.use_vllm_server:
            self.vllm_served_model_name = derive_served_model_name(
                self.model_name, torch_dtype
            )
            base_url = f"{vllm_url_root}:{vllm_port}{vllm_url_suffix}".rstrip("/")
            self.vllm_base_url = base_url
            from openai import OpenAI
            self.vllm_client = OpenAI(
                base_url=self.vllm_base_url,
                api_key=vllm_api_key,
                timeout=vllm_request_timeout_s,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            resolved_dtype = resolve_torch_dtype(torch_dtype)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=resolved_dtype,
                device_map=device_map,
            )
            self.model.eval()

    def queryVLM(self, prompt: str, images: List[Image.Image]) -> str:
        if not images:
            raise ValueError("GemmaVLMQuerier.queryVLM requires at least one image.")
        if self.use_vllm_server:
            return self._query_vllm(prompt, images)
        return self._query_local(prompt, images)

    def _query_vllm(self, prompt: str, images: List[Image.Image]) -> str:
        assert self.vllm_client is not None
        assert self.vllm_served_model_name is not None
        if not self._name_resolved:
            self.vllm_served_model_name = resolve_served_model_name(
                self.vllm_client,
                model_name=self.model_name,
                derived_name=self.vllm_served_model_name,
                base_url=self.vllm_base_url,
            )
            self._name_resolved = True
        content: List[Dict[str, Any]] = []
        for im in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": _pil_to_data_url_jpeg(im)},
            })
        content.append({"type": "text", "text": prompt})
        params: Dict[str, Any] = {
            "max_tokens": self._default_sampling["max_new_tokens"],
            "temperature": self._default_sampling["temperature"],
            "top_p": self._default_sampling["top_p"],
        }
        resp = self.vllm_client.chat.completions.create(
            model=self.vllm_served_model_name,
            messages=[{"role": "user", "content": content}],
            **params,
            **self.vllm_extra_request_params,
        )
        print(f"vLLM response: {resp.choices[0]}")
        return (resp.choices[0].message.content or "").strip()

    @torch.no_grad()
    def _query_local(self, prompt: str, images: List[Image.Image]) -> str:
        assert self.model is not None
        assert self.processor is not None
        content = []
        for im in images:
            content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
        ).to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
        )
        generated_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
