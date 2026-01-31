from dataset_generation.llm_persistence_querying.prompt_handler.abs_prompt_handler import AbsPromptHandler
from string import Template
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class VerbosePersistencePromptHandler(AbsPromptHandler):
    """
    Prompt/response handler for estimating persistence parameters.
    """

    def __init__(
        self,
        prompt_template_file: Optional[str] = None,
        prompt_template_str: Optional[str] = None,
        name: Optional[str] = None,
    ):
        if (name is None):
            raise ValueError("name must be provided for VerbosePersistencePromptHandler")
        super().__init__(name)
        if (prompt_template_str is not None) and (prompt_template_file is not None):
            raise ValueError("Only one of prompt_template or prompt_template_file should be provided.")

        if prompt_template_file is not None:
            with open(prompt_template_file, 'r') as f:
                prompt_template_str = f.read()


        self.prompt_template = Template(prompt_template_str)


    # ----------------------------------------------------------------------
    # PROMPT GENERATION
    # ----------------------------------------------------------------------
    def generatePrompt(
        self,
        sceneCaption: str,
        objectDescriptions: Dict[int, str],
        relationshipDescriptions: Dict[str, str],
        timeAndDate: Optional[str],
    ) -> str:

        time_str = timeAndDate if timeAndDate is not None else "unknown"

        # Format objects
        obj_lines = []
        for obj_id, desc in objectDescriptions.items():
            obj_lines.append(f"- id: {obj_id}\n  description: {desc}")
        objects_block = "\n".join(obj_lines)

        # Format relationships (not used in current prompt, but could be added later)
        rel_lines = []
        for rel_key, rel_desc in relationshipDescriptions.items():
            rel_lines.append(f"- {rel_key}: {rel_desc}")
        relationships_block = "\n".join(rel_lines)

        # Fill in template placeholders
        prompt = self.prompt_template.substitute(
            scene_caption=sceneCaption,
            time_str=time_str,
            objects_block=objects_block,
            relationships_block=relationships_block,
        )

        #         # Build the detailed instruction prompt
        #         prompt = f"""You are an expert in robot perception and probabilistic tracking.

        # The robot uses a persistence filter to decide how long to trust that an object stays where it was last seen.
        # You will receive:
        # - A scene-level description
        # - A list of objects with IDs and natural-language descriptions
        # - The approximate time and date

        # For EACH object, estimate how long it is likely to remain in roughly the same location, and how uncertain you are.

        # Return a JSON array where each element includes:
        # - "id": the object id (integer or string)
        # - "expected_persistence_hours": positive float — expected time (in hours) the object stays put
        # - "persistence_uncertainty": one of ["low", "medium", "high"]
        # - "static_likelihood": float in [0.0, 1.0] — probability that the object is effectively static
        # - "notes": short sentence explaining reasoning (max 1 sentence)

        # Guidelines:
        # - If the object is structural or immovable (walls, cabinets), use very large expected_persistence_hours (e.g., 1000+).
        # - If the object is carried, handheld, easily moved, or actively used, use small expected_persistence_hours that align with typical patterns in that sort of environment (ex. minutes for occupied desk, 3-24 hours for unoccupied kitchen)
        # - If uncertain, set "persistence_uncertainty" to "high".
        # - Respond with ONLY valid JSON. No commentary or explanation outside the JSON.

        # Scene description:
        # {sceneCaption}

        # Time and date:
        # {time_str}

        # Objects:
        # {objects_block}

        # JSON answer:
        # """
        return prompt

    # ----------------------------------------------------------------------
    # RESPONSE PARSING
    # ----------------------------------------------------------------------
    def parsePersistenceParamsFromLLMResponse(
        self,
        responseText: str
    ) -> List[Dict[str, Any]]:
        """
        Extract and parse the JSON array produced by the LLM.
        Includes fallback logic in case the model surrounds JSON with extra text.
        """

        cleaned = responseText.strip()

        # Case 1: Direct JSON array
        if cleaned.startswith("[") and cleaned.endswith("]"):
            return json.loads(cleaned)

        # Case 2: Try to find a JSON array inside the text
        match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Case 3: Try to find a single JSON object (rare but possible)
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                return obj if isinstance(obj, list) else [obj]
            except json.JSONDecodeError:
                pass

        # Case 4: Total failure → raise error to caller
        raise ValueError(f"Could not parse valid JSON from model response:\n{responseText}")