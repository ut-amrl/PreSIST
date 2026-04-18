from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Type, Union

import torch
from PIL import Image

from dataset_generation.llm_persistence_querying.utils.llm_querying_utils import (
    derive_served_model_name,
    resolve_served_model_name,
    resolve_torch_dtype,
)
from dataset_generation.scene_level_captioning.abs_scene_level_captioner import AbstractSceneLevelCaptioner
from instance_segmentation.object_discovery.qwen_object_discoverer import (
    _chw_uint8_to_pil,
    _downscale_for_retry,
    _from_pretrained_with_dtype_fallback,
    _get_qwen_vl_model_cls,
    _infer_qwen_family,
    _pil_to_data_url_jpeg,
)


class QwenVLSceneLevelCaptioner(AbstractSceneLevelCaptioner):
    def __init__(
        self,
        *,
        name: str,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        use_vllm_server: bool = True,
        torch_dtype: Union[torch.dtype, str] = "auto",
        vllm_url_root: str = "http://localhost",
        vllm_port: int = 9011,
        vllm_url_suffix: str = "/v1",
        vllm_api_key: str = "EMPTY",
        vllm_request_timeout_s: float = 240.0,
        vllm_extra_request_params: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        max_new_tokens: int = 512,
        num_beams: int = 2,
    ):
        super().__init__(name=name)

        if (torch_dtype == "auto") and use_vllm_server:
            torch_dtype = "bfloat16" if torch.cuda.is_available() else "float32"

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_vllm_server = use_vllm_server
        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = int(num_beams)

        if (prompt is not None) and (prompt_file is not None):
            raise ValueError("Only one of prompt or prompt_file should be provided.")

        default_prompt = "Describe the scene in detail."

        if prompt is None and prompt_file is None:
            self.prompt = default_prompt
        elif prompt_file is not None:
            with open(prompt_file, "r") as f:
                self.prompt = f.read()
        else:
            self.prompt = prompt  # type: ignore[assignment]

        if device_map is None:
            self.device_map = "auto" if self.device.startswith("cuda") else None
        else:
            self.device_map = device_map

        self.qwen_family = _infer_qwen_family(self.model_name)

        self.vllm_client = None
        self.vllm_served_model_name = None
        self.vllm_extra_request_params = vllm_extra_request_params or {}
        self.processor = None
        self.model = None

        if self.use_vllm_server:
            self.vllm_served_model_name = derive_served_model_name(self.model_name, torch_dtype)
            base_url = f"{vllm_url_root}:{vllm_port}{vllm_url_suffix}".rstrip("/")
            self.vllm_base_url = base_url
            from openai import OpenAI
            self.vllm_client = OpenAI(
                base_url=base_url,
                api_key=vllm_api_key,
                timeout=vllm_request_timeout_s,
            )
            self.vllm_served_model_name = resolve_served_model_name(
                self.vllm_client,
                model_name=self.model_name,
                derived_name=self.vllm_served_model_name,
                base_url=self.vllm_base_url,
            )
        else:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            model_cls = _get_qwen_vl_model_cls(self.qwen_family)
            resolved_dtype = resolve_torch_dtype(torch_dtype)
            self.model = _from_pretrained_with_dtype_fallback(
                model_cls, self.model_name, self.device_map, resolved_dtype
            )
            self.model.eval()

    def _make_vllm_request(
        self,
        content,
        enable_thinking: Optional[bool] = None,
        inject_no_think: bool = False,
    ):
        extra_body = None
        if enable_thinking is not None:
            extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

        user_content = content
        if inject_no_think:
            user_content = []
            injected = False
            for item in content:
                if (
                    not injected
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                ):
                    user_content.append({**item, "text": "/no_think\n" + item["text"]})
                    injected = True
                else:
                    user_content.append(item)
            if not injected:
                user_content.append({"type": "text", "text": "/no_think"})

        return self.vllm_client.chat.completions.create(
            model=self.vllm_served_model_name,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            extra_body=extra_body,
            **self.vllm_extra_request_params,
        )

    def _caption_vllm(self, pil: Image.Image) -> str:
        content = [
            {"type": "image_url", "image_url": {"url": _pil_to_data_url_jpeg(pil)}},
            {"type": "text", "text": self.prompt},
        ]

        resp = self._make_vllm_request(content, enable_thinking=None)
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()

        reasoning = (getattr(choice.message, "reasoning_content", "") or "").strip()
        if reasoning:
            print(f"[vLLM DEBUG] Reasoning length (first attempt): {len(reasoning)} chars")

        if choice.finish_reason == "length":
            print(f"[vLLM DEBUG] Generation hit max length ({self.max_new_tokens}), retrying with thinking disabled.")
            resp = self._make_vllm_request(content, enable_thinking=False, inject_no_think=True)
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()

            if choice.finish_reason == "length":
                max_img_size = max(512, max(pil.width, pil.height) // 4)
                retry_pil = _downscale_for_retry(pil, max_side=max_img_size)
                print("[vLLM DEBUG] Still truncated; retrying with downscaled image.")
                resized_content = [
                    {"type": "image_url", "image_url": {"url": _pil_to_data_url_jpeg(retry_pil)}},
                    {"type": "text", "text": self.prompt},
                ]
                resp = self._make_vllm_request(resized_content, enable_thinking=False, inject_no_think=True)
                choice = resp.choices[0]
                text = (choice.message.content or "").strip()

                if choice.finish_reason == "length":
                    raise RuntimeError(
                        "Scene caption generation was cut off even after retrying with thinking disabled and smaller image."
                    )

        print("vLLM response finish reason: ", choice.finish_reason)
        print("----------------- Output caption: -----------------")
        print(text)

        print(f"[vLLM DEBUG] Final reasoning length: {len(reasoning)} chars")

        return text

    @torch.no_grad()
    def _caption_local(self, pil: Image.Image) -> str:
        assert self.model is not None
        assert self.processor is not None

        from qwen_vl_utils import process_vision_info
        from instance_segmentation.object_discovery.qwen_object_discoverer import _build_qwenvl_prompt

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        text = _build_qwenvl_prompt(self.prompt)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        target_device = getattr(self.model, "device", None) or torch.device(self.device)
        inputs = inputs.to(target_device)

        generated_ids = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=1.15,
        )

        generated_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        return self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def generateCaption(self, image: torch.Tensor) -> str:
        if image.ndim == 3:
            pil = _chw_uint8_to_pil(image)
        elif image.ndim == 4:
            pil = _chw_uint8_to_pil(image[0])
        else:
            raise ValueError(f"Expected image with 3 or 4 dims, got {image.ndim}")

        if self.use_vllm_server:
            caption = self._caption_vllm(pil)
        else:
            caption = self._caption_local(pil)

        print(f"QwenVLSceneLevelCaptioner ({self.qwen_family}) caption: {caption[:200]}")
        return caption
