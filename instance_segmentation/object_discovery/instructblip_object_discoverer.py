from __future__ import annotations

from typing import List, Optional
import re
import torch
from instance_segmentation.object_discovery.abs_object_discoverer import AbsObjectDiscoverer
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration



def _normalize_phrase(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\.\;\:\(\)\[\]\{\}\"]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_listish_text(text: str, max_items: int) -> List[str]:
    """
    Parse outputs like:
      "person, dog, red car"
    or
      "1) person\n2) dog\n3) red car"
    """
    raw = text.strip()
    parts: List[str] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*[\-\*\d]+\s*[\)\.\:]\s*", "", line)
        for chunk in line.split(","):
            chunk = chunk.strip()
            if chunk:
                parts.append(chunk)

    seen = set()
    out: List[str] = []
    for p in parts:
        p2 = _normalize_phrase(p)
        if not p2:
            continue
        if p2 in {"scene", "image", "photo", "picture", "background"}:
            continue
        if p2 not in seen:
            seen.add(p2)
            out.append(p2)
        if len(out) >= max_items:
            break

    # fallback if it returns a sentence
    if not out and raw:
        rough = re.split(r"[,\n;/]+", raw)
        out = [_normalize_phrase(x) for x in rough if _normalize_phrase(x)]
        out = out[:max_items]

    return out


class InstructBLIPObjectDiscoverer(AbsObjectDiscoverer):
    """
    InstructBLIP-based object discoverer.

    Supports `prompt` or `prompt_file` (mutually exclusive), like your captioner example.
    Uses a default prompt if neither is provided.
    """

    def __init__(
        self,
        *,
        name: str,
        model_name: str = "Salesforce/instructblip-flan-t5-xl",
        device: Optional[str] = None,
        max_objects: int = 8,
        prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        max_new_tokens: int = 64,
        num_beams: int = 5,
    ):
        if name is None:
            raise ValueError("name must be provided for InstructBLIPObjectIdentifier")
        super().__init__(name=name)

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_objects = int(max_objects)

        if (prompt is not None) and (prompt_file is not None):
            raise ValueError("Only one of prompt or prompt_file should be provided.")

        # default_prompt = (
        #     "List the most notable objects in the image as short noun phrases. "
        #     f"Return up to {self.max_objects} items, separated by commas. "
        #     "No extra words."
        # )

        # default_prompt = (
        #     "Identify the objects in the image as short noun phrases. "
        #     f"Return up to {self.max_objects} items, separated by commas. "
        #     "No extra words."
        # )

        default_prompt = (
            "List all the distinct visible objects in the image."
            "Return the list as short noun phrases, separated by commas, no extra text"
        )


#         default_prompt = (
#             "List all distinct visible objects in the image.
# Output ONLY a comma separated list of strings.
# Each string must be a short noun phrase (1–4 words), lowercase.
# No extra keys, no explanations, no markdown.
# Prefer specific phrases helpful for segmentation (e.g., “red car” not “car” if multiple cars).
# Exclude “background”, “scene”, “sky”, “lighting”, and any uncertain objects."
#             "Identify the objects in the image as short noun phrases. "
#             f"Return up to {self.max_objects} items, separated by commas. "
#             "No extra words."
#         )



        if prompt is None and prompt_file is None:
            self.prompt = default_prompt
        elif prompt_file is not None:
            with open(prompt_file, "r") as f:
                self.prompt = f.read()
        else:
            self.prompt = prompt  # type: ignore[assignment]

        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = int(num_beams)

        self.processor = InstructBlipProcessor.from_pretrained(model_name)
        dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def identify(self, image: torch.Tensor) -> List[str]:
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor")
        if image.ndim != 3 or image.shape[0] not in (3, 4):
            raise ValueError("Expected image (C,H,W) with C in {3,4}.")
        if image.dtype != torch.uint8:
            raise ValueError("Expected image dtype=torch.uint8.")

        img = image[:3].unsqueeze(0)  # (1,3,H,W)

        inputs = self.processor(images=img, text=self.prompt, return_tensors="pt")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    inputs[k] = v.to(self.device, dtype=self.model.dtype)
                else:
                    inputs[k] = v.to(self.device)

        out_ids = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=1.2,
        )

        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        # print("InstructBLIPObjectDiscoverer identified text:", text)
        return _parse_listish_text(text, max_items=self.max_objects)
