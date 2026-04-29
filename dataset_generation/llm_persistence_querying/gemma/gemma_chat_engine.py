"""
Shared Gemma 4 chat engine: a single class used by both the LLM querier,
VLM querier, scene captioner, and object describer to talk to a Gemma 4
model in either vLLM-server mode or local Transformers mode.

Returns (main_text, reasoning) from `chat(...)` — the caller is responsible
for placing reasoning into a DebugPayload (the engine does not know the
naming convention or per-call context).

Thinking mode is enabled per-call via `enable_thinking` set at ctor time:
- vLLM mode passes `chat_template_kwargs={"enable_thinking": ...}` in
  `extra_body`; reasoning arrives on `message.reasoning`. Requires the
  server to be launched with `--reasoning-parser gemma4` (see
  start_gemma_vllm_server.sh's --thinking flag).
- Local mode passes `enable_thinking` to `apply_chat_template`, decodes
  with `skip_special_tokens=False`, and uses `processor.parse_response`
  to split thinking from content (per Gemma's official docs).
"""
from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from dataset_generation.llm_persistence_querying.utils.llm_querying_utils import (
    derive_served_model_name,
    resolve_served_model_name,
    resolve_torch_dtype,
)


def _pil_to_data_url_jpeg(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# Fallback regex for Gemma 4 thinking delimiters when processor.parse_response
# is unavailable. Per the model card the thinking content is wrapped as
# `<|channel>thought ... <channel|>`.
_GEMMA_THOUGHT_RE = re.compile(
    r"<\|channel\|?>\s*thought\s*(.*?)\s*<\|?channel\|?>",
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass
class GemmaChatResult:
    text: str
    reasoning: Optional[str] = None
    finish_reason: Optional[str] = None  # vLLM only; None for local mode


class GemmaChatEngine:
    def __init__(
        self,
        *,
        model_name: str = "google/gemma-4-31B-it",
        use_vllm_server: bool = True,
        enable_thinking: bool = False,
        # local (Transformers)
        device_map: Union[str, Dict[str, Any]] = "auto",
        torch_dtype: Union[torch.dtype, str] = "auto",
        # vLLM server
        vllm_url_root: str = "http://localhost",
        vllm_port: int = 9008,
        vllm_url_suffix: str = "/v1",
        vllm_api_key: str = "EMPTY",
        vllm_request_timeout_s: float = 240.0,
        vllm_extra_request_params: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.use_vllm_server = use_vllm_server
        self.enable_thinking = bool(enable_thinking)
        self.vllm_extra_request_params = dict(vllm_extra_request_params or {})

        self.model = None
        self.processor = None
        self.vllm_client = None
        self.vllm_base_url = ""
        self.vllm_served_model_name: Optional[str] = None
        self._name_resolved = False

        if self.use_vllm_server:
            self.vllm_served_model_name = derive_served_model_name(
                self.model_name, torch_dtype
            )
            base_url = f"{vllm_url_root}:{vllm_port}{vllm_url_suffix}".rstrip("/")
            self.vllm_base_url = base_url
            from openai import OpenAI
            self.vllm_client = OpenAI(
                base_url=base_url,
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

    def chat(
        self,
        *,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> GemmaChatResult:
        if self.use_vllm_server:
            return self._chat_vllm(
                prompt=prompt,
                images=images,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                seed=seed,
            )
        return self._chat_local(
            prompt=prompt,
            images=images,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def _resolve_served_name(self) -> None:
        if self._name_resolved:
            return
        assert self.vllm_client is not None
        assert self.vllm_served_model_name is not None
        self.vllm_served_model_name = resolve_served_model_name(
            self.vllm_client,
            model_name=self.model_name,
            derived_name=self.vllm_served_model_name,
            base_url=self.vllm_base_url,
        )
        self._name_resolved = True

    def _build_user_content(
        self, prompt: str, images: Optional[List[Image.Image]]
    ) -> Union[str, List[Dict[str, Any]]]:
        if not images:
            return prompt
        content: List[Dict[str, Any]] = []
        for im in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": _pil_to_data_url_jpeg(im)},
            })
        content.append({"type": "text", "text": prompt})
        return content

    def _merged_extra_body(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = dict(self.vllm_extra_request_params)
        chat_kwargs = dict(extra.get("chat_template_kwargs") or {})
        chat_kwargs.setdefault("enable_thinking", self.enable_thinking)
        extra["chat_template_kwargs"] = chat_kwargs
        return extra

    def _chat_vllm(
        self,
        *,
        prompt: str,
        images: Optional[List[Image.Image]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        seed: Optional[int],
    ) -> GemmaChatResult:
        self._resolve_served_name()
        assert self.vllm_client is not None
        content = self._build_user_content(prompt, images)
        params: Dict[str, Any] = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop is not None:
            params["stop"] = stop
        if seed is not None:
            params["seed"] = seed
        resp = self.vllm_client.chat.completions.create(
            model=self.vllm_served_model_name,
            messages=[{"role": "user", "content": content}],
            extra_body=self._merged_extra_body(),
            **params,
        )
        choice = resp.choices[0]
        msg = choice.message
        main = (msg.content or "").strip()
        raw_reasoning = (
            getattr(msg, "reasoning", None)
            or getattr(msg, "reasoning_content", None)
        )
        reasoning = (raw_reasoning or "").strip() or None
        return GemmaChatResult(
            text=main,
            reasoning=reasoning,
            finish_reason=choice.finish_reason,
        )

    @torch.no_grad()
    def _chat_local(
        self,
        *,
        prompt: str,
        images: Optional[List[Image.Image]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GemmaChatResult:
        assert self.model is not None
        assert self.processor is not None

        user_content: List[Dict[str, Any]] = []
        if images:
            for im in images:
                user_content.append({"type": "image", "image": im})
        user_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": user_content}]

        chat_template_kwargs: Dict[str, Any] = {}
        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            chat_template_kwargs["enable_thinking"] = self.enable_thinking

        proc_kwargs: Dict[str, Any] = {
            "text": text,
            "return_tensors": "pt",
        }
        if images:
            proc_kwargs["images"] = images
        inputs = self.processor(**proc_kwargs).to(self.model.device)

        do_sample = temperature is not None and temperature > 0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        decoded = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]

        main, reasoning = self._split_local_response(decoded)
        return GemmaChatResult(text=main, reasoning=reasoning, finish_reason=None)

    def _split_local_response(self, decoded: str) -> Tuple[str, Optional[str]]:
        parse_response = getattr(self.processor, "parse_response", None)
        if callable(parse_response):
            try:
                parsed = parse_response(decoded)
                if isinstance(parsed, dict):
                    main = (parsed.get("content") or "").strip()
                    thinking = (parsed.get("thinking") or "").strip() or None
                    return main, thinking
            except Exception:
                pass

        match = _GEMMA_THOUGHT_RE.search(decoded)
        if match:
            reasoning = match.group(1).strip() or None
            main = (decoded[: match.start()] + decoded[match.end():]).strip()
            return main, reasoning

        return decoded.strip(), None
