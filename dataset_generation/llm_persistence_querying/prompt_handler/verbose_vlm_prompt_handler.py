"""
Per-object VLM prompt handler: one prompt per object, no object id in prompt.
Response is a single JSON object; the integrating class adds id when aggregating.
Parsing mirrors VerbosePersistencePromptHandler (same three cases, adapted for one object).
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from instance_segmentation.abs_instance_segmenter import SegmentationResults
from utils.visualization_utils import mask_highlight_image, blur_bkg_outline_fg

from dataset_generation.llm_persistence_querying.prompt_handler.abs_vlm_prompt_handler import AbsVLMPromptHandler
from utils.image_utils import (
    image_tensor_to_pil,
    numpy_bgr_to_pil,
)


DEFAULT_SINGLE_OBJECT_INSTRUCTION = """You are estimating persistence for one object in a robot scene.

You are shown two images: (1) the full scene, and (2) a view that indicates the object of interest—either by overlaying it in color on the scene, or by showing the object region on a dark or black background (with the object in full color or as a white shape). In all cases, the second image identifies which object to evaluate.

For this object only, respond with a single JSON object (no array, no extra text) with exactly these keys:
- "expected_persistence_hours": positive float (hours the object is likely to stay in place)
- "persistence_uncertainty": one of "low", "medium", "high"
- "static_likelihood": float in [0.0, 1.0]
- "notes": one short sentence

Object description: {object_description}

Scene: {scene_caption}
Time/date: {time_str}

If other objects are listed below, use them only for context (e.g. "next to the table").
{other_objects_block}

Respond with ONLY the JSON object, no markdown or explanation."""


class VerboseVLMPersistencePromptHandler(AbsVLMPromptHandler):
    def __init__(
        self,
        name: str,
        instruction_template: Optional[str] = None,
        prompt_template_file: Optional[str] = None,
        mask_style: str = "mask_overlay",
        mask_highlight_color: Tuple[int, int, int] = (0, 0, 255),
        include_other_descriptions: bool = True,
    ):
        super().__init__(name=name)
        if (instruction_template is not None) and (prompt_template_file is not None):
            raise ValueError(
                "Only one of instruction_template or prompt_template_file should be provided."
            )
        if prompt_template_file is not None:
            with open(prompt_template_file, "r", encoding="utf-8") as f:
                instruction_template = f.read()
        self.instruction_template = instruction_template or DEFAULT_SINGLE_OBJECT_INSTRUCTION
        self.mask_style = mask_style
        self.mask_highlight_color = tuple(int(x) for x in mask_highlight_color)
        self.include_other_descriptions = bool(include_other_descriptions)

    def generatePromptForObject(
        self,
        object_description: str,
        scene_caption: str,
        time_and_date: Optional[str],
        *,
        other_descriptions: Optional[Dict[Any, str]] = None,
    ) -> str:
        time_str = time_and_date if time_and_date is not None else "unknown"
        other_descriptions = other_descriptions or {}
        other_lines = [f"- {v}" for v in other_descriptions.values()]
        other_objects_block = "\n".join(other_lines) if other_lines else "No other objects listed."

        return self.instruction_template.format(
            object_description=object_description,
            scene_caption=scene_caption,
            time_str=time_str,
            other_objects_block=other_objects_block,
        )

    def buildPromptAndImagesForObject(
        self,
        scene_caption: str,
        object_descriptions: Dict[Any, str],
        relationship_descriptions: Dict[Any, Any],
        time_and_date: Optional[str],
        image: torch.Tensor,
        instances: SegmentationResults,
        object_id: int,
    ) -> Tuple[str, List[Image.Image]]:
        _ = relationship_descriptions  # currently unused, reserved for future prompt variants

        matches = (instances.ids == object_id).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            raise ValueError(f"Object id {object_id} not found in instances.ids")
        object_index = int(matches[0].item())

        # Try to get the object description using the object_id directly; if that fails, use the string version of the object_id (in case keys are strings)
        try:
            object_description = object_descriptions[object_id]
        except KeyError:
            object_description = object_descriptions[str(object_id)]

        other_descriptions: Optional[Dict[Any, str]] = None
        if self.include_other_descriptions and object_descriptions:
            other_descriptions = {k: v for k, v in object_descriptions.items() if k != object_id}

        prompt = self.generatePromptForObject(
            object_description,
            scene_caption,
            time_and_date,
            other_descriptions=other_descriptions,
        )

        original_pil = image_tensor_to_pil(image)
        masked_np = mask_highlight_image(
            image,
            instances.masks[object_index],
            style=self.mask_style,
            mask_highlight_color=self.mask_highlight_color,
        )
        masked_pil = numpy_bgr_to_pil(masked_np)

        return prompt, [original_pil, masked_pil]

    # def parseSingleObjectPersistenceResponse(self, response_text: str) -> Dict[str, Any]:
    #     """
    #     Parse a single-object persistence response. Mirrors the existing
    #     VerbosePersistencePromptHandler.parsePersistenceParamsFromLLMResponse logic
    #     (same three cases), but returns one dict instead of a list.
    #     """
    #     cleaned = response_text.strip()

    #     # Case 1: Direct JSON object
    #     if cleaned.startswith("{") and cleaned.endswith("}"):
    #         try:
    #             return json.loads(cleaned)
    #         except json.JSONDecodeError:
    #             pass

    #     # Case 2: Try to find a JSON object inside the text
    #     match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    #     if match:
    #         try:
    #             obj = json.loads(match.group(0))
    #             if isinstance(obj, dict):
    #                 return obj
    #             if isinstance(obj, list) and len(obj) == 1:
    #                 return obj[0] if isinstance(obj[0], dict) else obj
    #         except json.JSONDecodeError:
    #             pass

    #     # Case 3: Try to find a JSON array (model returned [...] with one element)
    #     match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    #     if match:
    #         try:
    #             arr = json.loads(match.group(0))
    #             if isinstance(arr, list) and len(arr) >= 1 and isinstance(arr[0], dict):
    #                 return arr[0]
    #         except json.JSONDecodeError:
    #             pass

    #     raise ValueError(
    #         f"Could not parse valid JSON (single object) from model response:\n{response_text}"
    #     )



    def parseSingleObjectPersistenceResponse(self, response_text: str) -> Dict[str, Any]:
        """
        Parse a single-object persistence response.

        Behavior:
        - If there is a `</think>` tag, prefer JSON that appears *after* the last `</think>`.
        - Within the chosen region, collect all JSON-like snippets and use the **last**
          one that can be coerced to a dict.
        - Accept either:
          - a JSON object (`{...}`), or
          - a singleton JSON array of one dict (`[{...}]`), or
          - a JSON array where the first element is a dict (we take that dict).

        Raises ValueError if no suitable JSON object can be found.
        """
        cleaned = response_text.strip()

        def extract_last_single_object(text: str) -> Dict[str, Any]:
            text = text.strip()
            candidates: list[Dict[str, Any]] = []

            # Case A: whole string is a JSON object or array
            if (text.startswith("{") and text.endswith("}")) or (
                text.startswith("[") and text.endswith("]")
            ):
                try:
                    obj = json.loads(text)
                    # Normalize to a dict if possible
                    if isinstance(obj, dict):
                        candidates.append(obj)
                    elif isinstance(obj, list):
                        # singleton list-of-dict
                        if len(obj) == 1 and isinstance(obj[0], dict):
                            candidates.append(obj[0])
                        # array-of-dicts: take first dict
                        elif len(obj) >= 1 and isinstance(obj[0], dict):
                            candidates.append(obj[0])
                except json.JSONDecodeError:
                    pass  # fall through to more flexible parsing

            # Case B: find all JSON objects `{...}` in the text (non-greedy)
            for m in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
                candidate = m.group(0)
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    continue

                if isinstance(obj, dict):
                    candidates.append(obj)
                elif isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
                    candidates.append(obj[0])

            # Case C: find JSON arrays `[...]` and pull dicts from them (non-greedy)
            for m in re.finditer(r"\[.*?\]", text, flags=re.DOTALL):
                candidate = m.group(0)
                try:
                    arr = json.loads(candidate)
                except json.JSONDecodeError:
                    continue

                if isinstance(arr, list) and len(arr) >= 1 and isinstance(arr[0], dict):
                    candidates.append(arr[0])

            if not candidates:
                raise ValueError("No valid single-object JSON found in text")

            # Use the **last** candidate that fit the expected shape
            return candidates[-1]

        # 1) Prefer anything after the last </think> tag
        think_end_idx = cleaned.rfind("</think>")
        if think_end_idx != -1:
            after_think = cleaned[think_end_idx + len("</think>") :]
            try:
                return extract_last_single_object(after_think)
            except ValueError:
                # Fall back to scanning the whole response if nothing usable after </think>
                pass

        # 2) Fallback: use the whole text
        try:
            return extract_last_single_object(cleaned)
        except ValueError:
            raise ValueError(
                f"Could not parse valid JSON (single object) from model response:\n{response_text}"
            )