"""
Qwen VLM querier for vision-language persistence prompts.
Parallel to QwenLLMQuerier; uses Qwen2-VL or Qwen3-VL with multiple images per call.

Two modes (like QwenLLMQuerier):
1) In-memory (Transformers): loads model + processor + tokenizer, uses generate().
2) vLLM server (OpenAI-compatible): sends chat completions with image_url parts to vLLM.
Switch via use_vllm_server.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional, Type, Union

from dataset_generation.llm_persistence_querying.utils.llm_querying_utils import derive_served_model_name, resolve_served_model_name, resolve_torch_dtype
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

# qwen_vlm_querier.py  (or whatever the file is called)
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)



from dataset_generation.llm_persistence_querying.vlm_querying.abs_vlm_querier import AbsVLMQuerier


def _pil_to_data_url_jpeg(image: Image.Image) -> str:
    """Encode a PIL Image to a data URL (JPEG base64) for OpenAI-compatible API."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# def _derive_served_model_name(model_name: str, torch_dtype: str, quantization: bool = False) -> str:
#     """Match start_vllm_server.sh: basename lower + dtype + optional quant."""
#     model_basename = os.path.basename(model_name)
#     model_normalized = model_basename.lower()
#     dtype_str = torch_dtype.split(".")[-1].lower() if isinstance(torch_dtype, str) else str(torch_dtype).split(".")[-1].lower()
#     name = f"{model_normalized}__{dtype_str}"
#     if quantization:
#         name = f"{name}__{quantization}"
#     return name


def _build_qwenvl_prompt_n_images(user_text: str, num_images: int) -> str:
    """Build Qwen VL prompt with N vision placeholders (one per image)."""
    placeholders = "\n".join(
        "<|vision_start|><|image_pad|><|vision_end|>" for _ in range(num_images)
    )
    return (
        "<|im_start|>user\n"
        f"{placeholders}\n"
        f"{user_text.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _infer_qwen_family(model_name: str) -> str:
    s = model_name.lower()
    if "qwen3" in s:
        return "qwen3"
    if "qwen2" in s:
        return "qwen2"
    raise ValueError(
        f"Could not infer Qwen family from model_name={model_name!r}. "
        "Expected to find 'Qwen2' or 'Qwen3' in the name."
    )


def _get_qwen_vl_model_cls(family: str) -> Type[torch.nn.Module]:
    if family == "qwen2":
        from transformers import Qwen2VLForConditionalGeneration as Cls
        return Cls
    if family == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration as Cls
        return Cls
    raise ValueError(f"Unknown family: {family!r}")


def _from_pretrained_with_dtype_fallback(model_cls, model_name: str, device_map, torch_dtype) -> torch.nn.Module:
    try:
        return model_cls.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    except TypeError:
        return model_cls.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map=device_map,
        )


class QwenVLMQuerier(AbsVLMQuerier):
    """
    VLM querier using Qwen2-VL or Qwen3-VL.
    Accepts a prompt and a list of PIL images; sends one multi-image user message and returns the reply.

    Two modes:
    1) use_vllm_server=False (default): load model locally with Transformers.
    2) use_vllm_server=True: use OpenAI-compatible vLLM server (no local model load).
    """

    def __init__(
        self,
        *,
        name: str,
        model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
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
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ):
        if name is None:
            raise ValueError("name must be provided for QwenVLMQuerier")
        super().__init__(name=name)

        self.model_name = model_name
        self.use_vllm_server = use_vllm_server
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        self._default_sampling: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }
        self.vllm_extra_request_params = vllm_extra_request_params or {}

        self.model = None
        self.processor = None
        self.tokenizer = None
        # self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_map = device_map
        self.qwen_family = _infer_qwen_family(self.model_name)
        self.vllm_client = None
        self.vllm_base_url = ""
        self._name_resolved = False

        print("Using QwenVLMQuerier with use_vllm_server =", self.use_vllm_server)

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
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if getattr(self.processor, "chat_template", None) in (None, ""):
                tok_template = getattr(self.tokenizer, "chat_template", None)
                if tok_template:
                    self.processor.chat_template = tok_template
            model_cls = _get_qwen_vl_model_cls(self.qwen_family)
            resolved_dtype = resolve_torch_dtype(torch_dtype)
            self.model = _from_pretrained_with_dtype_fallback(
                model_cls, self.model_name, self.device_map, resolved_dtype,
            )
            self.model.eval()

    def queryVLM(self, prompt: str, images: List[Image.Image]) -> str:
        if not images:
            raise ValueError("QwenVLMQuerier.queryVLM requires at least one image.")
        if self.use_vllm_server:
            return self._query_vllm(prompt, images)
        return self._query_local(prompt, images)

    def _query_vllm(self, prompt: str, images: List[Image.Image]) -> str:
        """Send images + text to vLLM via OpenAI chat completions (content: image_url + text)."""
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
        return (resp.choices[0].message.content or "").strip()

    @torch.no_grad()
    def _query_local(self, prompt: str, images: List[Image.Image]) -> str:
        """Local Transformers path: processor + model.generate()."""
        assert self.model is not None
        assert self.processor is not None
        content = []
        for im in images:
            content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = _build_qwenvl_prompt_n_images(prompt, len(images))
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # target_device = getattr(self.model, "device", None) or torch.device(self.device)
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            do_sample=self.do_sample,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )
        generated_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        text_out = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return text_out
