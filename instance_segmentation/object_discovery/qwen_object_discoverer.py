from __future__ import annotations

from typing import List, Optional
import re

import torch
from PIL import Image

from instance_segmentation.object_discovery.abs_object_discoverer import AbsObjectDiscoverer

# from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
# from qwen_vl_utils import process_vision_info

from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ... keep your helpers unchanged ...

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

def _normalize_phrase(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\.\;\:\(\)\[\]\{\}\"]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_listish_text(text: str, max_items: int) -> List[str]:
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

    if not parts:
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



class QwenVLObjectDiscoverer(AbsObjectDiscoverer):
    def __init__(
        self,
        *,
        name: str,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        max_objects: int = 20,
        prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        max_new_tokens: int = 256,
        num_beams: int = 2,
    ):
        super().__init__(name=name)

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_objects = int(max_objects)

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
        self.num_beams = int(num_beams)

        if device_map is None:
            self.device_map = "auto" if self.device.startswith("cuda") else None
        else:
            self.device_map = device_map

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # If processor lacks a chat template, copy it from the tokenizer if available.
        if getattr(self.processor, "chat_template", None) in (None, ""):
            tok_template = getattr(self.tokenizer, "chat_template", None)
            if tok_template:
                self.processor.chat_template = tok_template

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map=self.device_map,
        )
        self.model.eval()



    @torch.no_grad()
    def identify(self, image: torch.Tensor) -> List[str]:
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

        # ✅ Include required vision placeholder tokens in the text prompt
        text = _build_qwen2vl_prompt(self.prompt)

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

        text_out = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        print("QwenVLObjectDiscoverer identified text:", text_out)
        return _parse_listish_text(text_out, max_items=self.max_objects)


    # @torch.no_grad()
    # def identify(self, image: torch.Tensor) -> List[str]:
    #     pil = _chw_uint8_to_pil(image)

    #     messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "image", "image": pil},
    #                 {"type": "text", "text": self.prompt},
    #             ],
    #         }
    #     ]

    #     # ✅ Avoid chat templates entirely (fixes your error)
    #     text = "User: " + self.prompt.strip() + "\nAssistant:"

    #     image_inputs, video_inputs = process_vision_info(messages)

    #     inputs = self.processor(
    #         text=[text],
    #         images=image_inputs,
    #         videos=video_inputs,
    #         padding=True,
    #         return_tensors="pt",
    #     )

    #     target_device = getattr(self.model, "device", None) or torch.device(self.device)
    #     inputs = inputs.to(target_device)

    #     generated_ids = self.model.generate(
    #         **inputs,
    #         do_sample=False,
    #         num_beams=self.num_beams,
    #         max_new_tokens=self.max_new_tokens,
    #         repetition_penalty=1.15,
    #     )

    #     # Since we did not use apply_chat_template, trimming by input length is still correct:
    #     generated_trimmed = [
    #         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    #     ]

    #     text_out = self.processor.batch_decode(
    #         generated_trimmed,
    #         skip_special_tokens=True,
    #         clean_up_tokenization_spaces=False,
    #     )[0].strip()

    #     print("QwenVLObjectDiscoverer identified text:", text_out)
    #     return _parse_listish_text(text_out, max_items=self.max_objects)


    # def identify(self, image: torch.Tensor) -> List[str]:
    #     pil = _chw_uint8_to_pil(image)

    #     messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "image", "image": pil},
    #                 {"type": "text", "text": self.prompt},
    #             ],
    #         }
    #     ]

    #     # --- IMPORTANT CHANGE: apply_chat_template via tokenizer first (robust) ---
    #     try:
    #         text = self.processor.apply_chat_template(
    #             messages, tokenize=False, add_generation_prompt=True
    #         )
    #     except Exception:
    #         text = self.tokenizer.apply_chat_template(
    #             messages, tokenize=False, add_generation_prompt=True
    #         )

    #     image_inputs, video_inputs = process_vision_info(messages)

    #     inputs = self.processor(
    #         text=[text],
    #         images=image_inputs,
    #         videos=video_inputs,
    #         padding=True,
    #         return_tensors="pt",
    #     )

    #     target_device = getattr(self.model, "device", None) or torch.device(self.device)
    #     inputs = inputs.to(target_device)

    #     generated_ids = self.model.generate(
    #         **inputs,
    #         do_sample=False,
    #         num_beams=self.num_beams,
    #         max_new_tokens=self.max_new_tokens,
    #         repetition_penalty=1.15,
    #     )

    #     generated_trimmed = [
    #         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    #     ]

    #     text_out = self.processor.batch_decode(
    #         generated_trimmed,
    #         skip_special_tokens=True,
    #         clean_up_tokenization_spaces=False,
    #     )[0].strip()

    #     print("QwenVLObjectDiscoverer identified text:", text_out)
    #     return _parse_listish_text(text_out, max_items=self.max_objects)




# class QwenVLObjectDiscoverer(AbsObjectDiscoverer):
#     """
#     Qwen2-VL-based object discoverer.
#     Uses a newline-list output format (most in-domain for Qwen).
#     """

#     def __init__(
#         self,
#         *,
#         name: str,
#         model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
#         device: Optional[str] = None,
#         device_map: Optional[str] = None,
#         max_objects: int = 20,
#         prompt: Optional[str] = None,
#         prompt_file: Optional[str] = None,
#         max_new_tokens: int = 256,
#         num_beams: int = 2,
#     ):
#         if name is None:
#             raise ValueError("name must be provided for QwenVLObjectDiscoverer")
#         super().__init__(name=name)

#         self.model_name = model_name
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.max_objects = int(max_objects)

#         if (prompt is not None) and (prompt_file is not None):
#             raise ValueError("Only one of prompt or prompt_file should be provided.")

#         # === Updated default prompt (yours) ===
#         default_prompt = (
#             "List all visible objects and structures in the image that someone might want to segment.\n"
#             "Output as short noun phrases, one per line, lowercase, 1–4 words, no extra text.\n"
#             "Include both foreground items and large static things (e.g., “road”, “building”, “tree”).\n"
#             "Do not include background/scene/sky/lighting."
#         )

#         if prompt is None and prompt_file is None:
#             self.prompt = default_prompt
#         elif prompt_file is not None:
#             with open(prompt_file, "r") as f:
#                 self.prompt = f.read()
#         else:
#             self.prompt = prompt  # type: ignore[assignment]

#         self.max_new_tokens = int(max_new_tokens)
#         self.num_beams = int(num_beams)

#         if device_map is None:
#             self.device_map = "auto" if self.device.startswith("cuda") else None
#         else:
#             self.device_map = device_map

#         self.processor = AutoProcessor.from_pretrained(self.model_name)

#         self.model = Qwen2VLForConditionalGeneration.from_pretrained(
#             self.model_name,
#             torch_dtype="auto",
#             device_map=self.device_map,
#         )
#         self.model.eval()

#     @torch.no_grad()
#     def identify(self, image: torch.Tensor) -> List[str]:
#         pil = _chw_uint8_to_pil(image)

#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": pil},
#                     {"type": "text", "text": self.prompt},
#                 ],
#             }
#         ]

#         text = self.processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )

#         image_inputs, video_inputs = process_vision_info(messages)

#         inputs = self.processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )

#         target_device = getattr(self.model, "device", None) or torch.device(self.device)
#         inputs = inputs.to(target_device)

#         generated_ids = self.model.generate(
#             **inputs,
#             do_sample=False,
#             num_beams=self.num_beams,
#             max_new_tokens=self.max_new_tokens,
#             repetition_penalty=1.15,
#         )

#         generated_trimmed = [
#             out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]

#         text_out = self.processor.batch_decode(
#             generated_trimmed,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False,
#         )[0].strip()

#         print("QwenVLObjectDiscoverer identified text:", text_out)
#         return _parse_listish_text(text_out, max_items=self.max_objects)
