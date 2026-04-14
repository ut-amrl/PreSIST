from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Type, Union
import re

import torch
from PIL import Image

from dataset_generation.llm_persistence_querying.utils.llm_querying_utils import derive_served_model_name, resolve_torch_dtype
from instance_segmentation.object_discovery.abs_object_discoverer import AbsObjectDiscoverer

from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info


def _build_qwen2vl_prompt(user_text: str) -> str:
    # Minimal Qwen chat framing + required vision placeholder tokens
    # This is what the chat template would normally insert.
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        f"{user_text.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _downscale_for_retry(pil: Image.Image, max_side: int = 768) -> Image.Image:
    w, h = pil.size
    scale = min(max_side / max(w, h), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if (new_w, new_h) == (w, h):
        return pil
    return pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

def _pil_to_data_url_jpeg(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

def _normalize_phrase(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\.\;\:\(\)\[\]\{\}\"]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_listish_text(text: str, max_items: int) -> List[str]:
    # Update to allow either new line separated or comma separated lists

    raw = text.strip()
    parts: List[str] = []

    # Split by lines (primary), commas (fallback)
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove bullets / numbering
        line = re.sub(r"^\s*[\-\*\d]+\s*[\)\.\:]\s*", "", line)
        if line:
            parts.append(line)

    if not parts or len(parts) == 1:
        # Fallback: comma-split
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if chunk:
                parts.append(chunk)

    seen = set()
    out: List[str] = []
    for p in parts:
        p2 = _normalize_phrase(p)
        if not p2:
            continue
        if p2 in {"scene", "image", "photo", "picture", "background", "sky", "lighting"}:
            continue
        if p2 not in seen:
            seen.add(p2)
            out.append(p2)
            if len(out) >= max_items:
                break

    return out


def _chw_uint8_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 3 or image.shape[0] not in (3, 4):
        raise ValueError("Expected image (C,H,W) with C in {3,4}.")
    if image.dtype != torch.uint8:
        raise ValueError("Expected image dtype=torch.uint8.")

    img3 = image[:3].permute(1, 2, 0).contiguous().cpu().numpy()
    return Image.fromarray(img3)



def _build_qwenvl_prompt(user_text: str) -> str:
    # Minimal Qwen chat framing + required vision placeholder tokens
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        f"{user_text.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def _infer_qwen_family(model_name: str) -> str:
    """
    Returns: "qwen2" or "qwen3"
    Decides by searching for 'qwen2' or 'qwen3' anywhere in the model_name.
    """
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
    """
    Lazy import so your code can run even if only one of the classes exists
    in the installed transformers version.
    """
    if family == "qwen2":
        try:
            from transformers import Qwen2VLForConditionalGeneration as Cls
        except Exception as e:
            raise ImportError(
                "Failed to import Qwen2VLForConditionalGeneration. "
                "Your transformers version may be too old/new."
            ) from e
        return Cls

    if family == "qwen3":
        try:
            from transformers import Qwen3VLForConditionalGeneration as Cls
        except Exception as e:
            raise ImportError(
                "Failed to import Qwen3VLForConditionalGeneration. "
                "Install/upgrade transformers to a version that supports Qwen3-VL."
            ) from e
        return Cls

    raise ValueError(f"Unknown family: {family!r}")

def _from_pretrained_with_dtype_fallback(model_cls, model_name: str, device_map, torch_dtype):
    """
    Some releases accept torch_dtype=..., others show dtype=...
    Try torch_dtype first, then dtype.
    """
    try:
        return model_cls.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    except TypeError:
        # Fallback for variants/docs that use dtype=...
        return model_cls.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map=device_map,
        )


class QwenVLObjectDiscoverer(AbsObjectDiscoverer):
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
        # vllm_port: int = 9008,
        vllm_port: int = 9010,
        vllm_url_suffix: str = "/v1",
        vllm_api_key: str = "EMPTY",
        vllm_request_timeout_s: float = 240.0,
        vllm_extra_request_params: Optional[Dict[str, Any]] = None,
        max_objects: int = 20,
        prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        max_new_tokens: int = 2048,
        # max_new_tokens: int = 2048*16,
        num_beams: int = 2,
    ):
        super().__init__(name=name)

        # model_name = "Qwen/Qwen3-VL-8B-Instruct"
        vllm_port = 9011


        if (torch_dtype == "auto") and use_vllm_server:
            torch_dtype = "bfloat16" if torch.cuda.is_available() else "float32"

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_objects = int(max_objects)
        self.use_vllm_server = use_vllm_server

        if (prompt is not None) and (prompt_file is not None):
            raise ValueError("Only one of prompt or prompt_file should be provided.")

        default_prompt = (
            "List all visible objects and structures in the image that someone might want to segment.\n"
            "Output as short noun phrases, one per line, lowercase, 1–4 words, no extra text.\n"
            "Include both foreground items and large static things (e.g., “road”, “building”, “tree”).\n"
            "Do not include background/scene/sky/lighting."
        )

        if prompt is None and prompt_file is None:
            self.prompt = default_prompt
        elif prompt_file is not None:
            with open(prompt_file, "r") as f:
                self.prompt = f.read()
        else:
            self.prompt = prompt  # type: ignore[assignment]

        self.max_new_tokens = int(max_new_tokens)
        # self.max_new_tokens = 8192
        # print("Max new tokens: ", self.max_new_tokens)
        # exit(1)
        self.num_beams = int(num_beams)

        if device_map is None:
            self.device_map = "auto" if self.device.startswith("cuda") else None
        else:
            self.device_map = device_map

        # Infer family from model_name (Qwen2 vs Qwen3)
        self.qwen_family = _infer_qwen_family(self.model_name)

        self.vllm_client = None
        self.vllm_served_model_name = None
        self.vllm_extra_request_params = vllm_extra_request_params or {}
        self.processor = None
        self.tokenizer = None
        self.model = None

        if self.use_vllm_server:
            self.vllm_served_model_name = derive_served_model_name(self.model_name, torch_dtype)
            base_url = f"{vllm_url_root}:{vllm_port}{vllm_url_suffix}".rstrip("/")
            from openai import OpenAI
            self.vllm_client = OpenAI(
                base_url=base_url,
                api_key=vllm_api_key,
                timeout=vllm_request_timeout_s,
            )
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # If processor lacks a chat template, copy it from the tokenizer if available.
            if getattr(self.processor, "chat_template", None) in (None, ""):
                tok_template = getattr(self.tokenizer, "chat_template", None)
                if tok_template:
                    self.processor.chat_template = tok_template

            # Load the right model class based on name
            model_cls = _get_qwen_vl_model_cls(self.qwen_family)
            resolved_dtype = resolve_torch_dtype(torch_dtype)
            self.model = _from_pretrained_with_dtype_fallback(
                model_cls, self.model_name, self.device_map, resolved_dtype
            )
            self.model.eval()

    @torch.no_grad()
    def identify(self, image: torch.Tensor) -> List[str]:
        if self.use_vllm_server:
            text_out = self._identify_vllm(image)
        else:
            text_out = self._identify_local(image)
        print(f"QwenVLObjectDiscoverer ({self.qwen_family}) identified text:", text_out)
        return _parse_listish_text(text_out, max_items=self.max_objects)

    def _make_vllm_request(
        self,
        content,
        enable_thinking: Optional[bool] = None,
        inject_no_think: bool = False,
    ):
        """
        Send a request to the vLLM server.

        inject_no_think:
            Prepends '/no_think' to the first text chunk of the user message.
        """
        extra_body = None

        if enable_thinking is not None:
            extra_body = {
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking
                }
            }

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
                    user_content.append({
                        **item,
                        "text": "/no_think\n" + item["text"],
                    })
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


    def _identify_vllm(self, image: torch.Tensor) -> str:
        assert self.vllm_client is not None
        assert self.vllm_served_model_name is not None

        pil = _chw_uint8_to_pil(image)

        content = [
            {
                "type": "image_url",
                "image_url": {"url": _pil_to_data_url_jpeg(pil)},
            },
            {"type": "text", "text": self.prompt},
        ]

        # First request (normal)
        print("Making first request to vLLM server with default thinking behavior...")
        resp = self._make_vllm_request(content, enable_thinking=None)

        choice = resp.choices[0]
        text = (choice.message.content or "").strip()


        reasoning = (getattr(choice.message, "reasoning_content", "") or "").strip()
        # print("Reasoning (if provided by model):\n", reasoning)
        # Print a truncated version of the reasoning if provided, to avoid overwhelming the logs
        if reasoning:
            # Print the full reasoning length
            print(f"[vLLM DEBUG] Reasoning length (first attempt): {len(reasoning)} chars")
            print(
                "[vLLM DEBUG] Reasoning from first attempt (truncated to 1000 chars):\n",
                reasoning[:1000] + ("...[truncated]" if len(reasoning) > 1000 else "")
            )


        if choice.finish_reason == "length":
            print("\n[vLLM DEBUG] Generation hit max length: ", self.max_new_tokens)

            print(f"[vLLM DEBUG] Image size: {pil.width}x{pil.height}")
            print(f"[vLLM DEBUG] Prompt chars: {len(self.prompt)}")
            print(f"[vLLM DEBUG] Response chars (first attempt): {len(text)}")

            # Print the reasoning if provided by the model (vLLM feature)


            if hasattr(resp, "usage") and resp.usage is not None:
                print(
                    "[vLLM DEBUG] Token usage (first attempt) "
                    f"(prompt={resp.usage.prompt_tokens}, "
                    f"completion={resp.usage.completion_tokens}, "
                    f"total={resp.usage.total_tokens})"
                )

            print("[vLLM DEBUG] Retrying with thinking disabled")

            resp = self._make_vllm_request(
                content,
                enable_thinking=False,
                inject_no_think=True,
            )

            choice = resp.choices[0]
            text = (choice.message.content or "").strip()

            # Print the first 1000 lines of text or 4000 characters, whichever is hit first, to avoid overwhelming the logs
            text_to_print = text[:4000]
            text_lines = text_to_print.split('\n')
            text_lines = text_lines[:1000]
            print(f"[vLLM DEBUG] Response from retry (truncated to 1000 lines or 4000 chars):\n", "\n".join(text_lines) + ("...[truncated]" if len(text) > 4000 else ""))


            reasoning = (getattr(choice.message, "reasoning_content", "") or "").strip()
            # Print a truncated version of the reasoning if provided, to avoid overwhelming the logs
            if reasoning:
                # Print the full reasoning length
                print(f"[vLLM DEBUG] Reasoning length (retry): {len(reasoning)} chars")
                print(
                    "[vLLM DEBUG] Reasoning from retry (truncated to 1000 chars):\n",
                    reasoning[:1000] + ("...[truncated]" if len(reasoning) > 1000 else "")
                )
            # print("Reasoning after retry (if provided by model):\n", reasoning)

            if choice.finish_reason == "length":

                # Make a subsequent retry with a smaller image
                # Try cutting the image in 1/3 or to a max side of 512, whichever is larger
                max_img_size = max(512, max(pil.width, pil.height) // 4)
                retry_pil = _downscale_for_retry(pil, max_side=max_img_size)
                print("[vLLM DEBUG] Generation was cut off. Retrying again with thinking disabled and downscaled image (max side 512).")


                if hasattr(resp, "usage") and resp.usage is not None:
                    print(
                        "[vLLM DEBUG] Token usage (first attempt) "
                        f"(prompt={resp.usage.prompt_tokens}, "
                        f"completion={resp.usage.completion_tokens}, "
                        f"total={resp.usage.total_tokens})"
                    )

                resized_content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": _pil_to_data_url_jpeg(retry_pil)},
                    },
                    {"type": "text", "text": self.prompt},
                ]

                resp = self._make_vllm_request(
                    resized_content,
                    enable_thinking=False,
                    inject_no_think=True,
                )

                choice = resp.choices[0]
                text = (choice.message.content or "").strip()

                # Print the first 1000 lines of text or 4000 characters, whichever is hit first, to avoid overwhelming the logs
                text_to_print = text[:4000]
                text_lines = text_to_print.split('\n')
                text_lines = text_lines[:1000]
                print(f"[vLLM DEBUG] Response from retry (truncated to 1000 lines or 4000 chars):\n", "\n".join(text_lines) + ("...[truncated]" if len(text) > 4000 else ""))


                reasoning = (getattr(choice.message, "reasoning_content", "") or "").strip()
                # Print a truncated version of the reasoning if provided, to avoid overwhelming the logs
                if reasoning:
                    # Print the full reasoning length
                    print(f"[vLLM DEBUG] Reasoning length (retry): {len(reasoning)} chars")
                    print(
                        "[vLLM DEBUG] Reasoning from retry (truncated to 1000 chars):\n",
                        reasoning[:1000] + ("...[truncated]" if len(reasoning) > 1000 else "")
                    )
                # print("Reasoning after retry (if provided by model):\n", reasoning)

                if choice.finish_reason == "length":

                    if hasattr(resp, "usage") and resp.usage is not None:
                        print(
                            "[vLLM DEBUG] Token usage (first attempt) "
                            f"(prompt={resp.usage.prompt_tokens}, "
                            f"completion={resp.usage.completion_tokens}, "
                            f"total={resp.usage.total_tokens})"
                        )
                    raise RuntimeError(
                        "Generation was cut off even after retrying with thinking disabled and smaller image."
                    )


        return text


    def _identify_local(self, image: torch.Tensor) -> str:
        assert self.model is not None
        assert self.processor is not None
        pil = _chw_uint8_to_pil(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        # Keep using required vision placeholder tokens (works for Qwen2-VL / Qwen3-VL)
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
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        text_out = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return text_out

