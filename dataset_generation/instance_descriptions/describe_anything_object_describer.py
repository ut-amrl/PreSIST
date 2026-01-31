from utils.image_utils import tensor_image_to_pil, mask_to_pil
from instance_segmentation.abs_instance_segmenter import SegmentationResults
from dataset_generation.instance_descriptions.abs_object_describer import AbstractObjectDescriber
import torch
import numpy as np
from typing import Optional, Dict


from PIL import Image
from dam import DescribeAnythingModel, disable_torch_init  # from the repo


class DescribeAnythingObjectDescriber(AbstractObjectDescriber):
    def __init__(
        self,
        model_path: str = "nvidia/DAM-3B",
        device: Optional[torch.device] = None,
        prompt_file: Optional[str] = None,
        prompt: Optional[str] = None,
        conv_mode: str = "v1",
        prompt_mode: str = "focal_prompt",
        *,
        temperature: float = 0.2,
        top_p: float = 0.5,
        num_beams: int = 1,
        max_new_tokens: int = 512,
        streaming: bool = False,
        name: Optional[str] = None,
    ):
        """
        Thin wrapper around NVidia's DescribeAnythingModel.

        Args:
            model_path:
                Path or Hugging Face ID for the DAM checkpoint.
                In their example they use "nvidia/DAM-3B". :contentReference[oaicite:0]{index=0}

            device:
                Torch device to run DAM on. If None, picks CUDA if available, else CPU.

            prompt:
                Text prompt given to the model, e.g.
                "<image>\\nDescribe the masked region in detail."

            conv_mode:
                Conversation template. The examples use "v1". :contentReference[oaicite:1]{index=1}

            prompt_mode:
                High-level prompt mode string. In the example script they accept
                "focal_prompt" and internally map it to "full+focal_crop". :contentReference[oaicite:2]{index=2}
                We preserve that behavior below.

            temperature, top_p, num_beams, max_new_tokens, streaming:
                Generation parameters forwarded to `get_description`.
        """

        if name is None:
            raise ValueError("name must be provided for DescribeAnythingObjectDescriber")
        super().__init__(name=name)
        disable_torch_init()  # same as in examples to speed up init

        if (prompt is not None) and (prompt_file is not None):
            raise ValueError("Only one of prompt or prompt_file should be provided.")

        if (prompt_file is None) and (prompt is None):
            prompt = "<image>\nDescribe the masked region in detail."
        elif prompt_file is not None:
            with open(prompt_file, 'r') as f:
                prompt = f.read()


        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Mirror the CLI mapping in examples/dam_with_sam.py
        prompt_modes = {
            "focal_prompt": "full+focal_crop",
        }
        resolved_prompt_mode = prompt_modes.get(prompt_mode, prompt_mode)

        # Create the DAM model exactly like the example script
        self.dam = DescribeAnythingModel(
            model_path=model_path,
            conv_mode=conv_mode,
            prompt_mode=resolved_prompt_mode,
        ).to(self.device)

        self.prompt = prompt
        self.gen_kwargs = dict(
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            streaming=streaming,
        )



    # ---------- main API ----------

    def describeObjects(
        self,
        image: torch.Tensor,
        instances: SegmentationResults,
    ) -> Dict[int, str]:
        """
        Describe objects in the image based on the provided instances.

        Args:
            image:
                Input image tensor of shape (C, H, W).
            instances:
                SegmentationResults with:
                    - instances.masks: iterable of per-instance masks (H, W) or (1, H, W)
                    - instances.ids: tensor/list of instance IDs
                    - instances.boxes: tensor of (N, 4) in XYXY format (currently unused)

        Returns:
            dict[obj_id -> description string]
        """
        pil_image = tensor_image_to_pil(image)
        descriptions: Dict[int, str] = {}

        for i, mask in enumerate(instances.masks):
            obj_id = int(instances.ids[i].item())
            # obj_bb = instances.boxes[i]  # available if you later want to use it

            pil_mask = mask_to_pil(mask)

            # Call Describe Anything Model
            out = self.dam.get_description(
                pil_image,
                pil_mask,
                self.prompt,
                **self.gen_kwargs,
            )

            # If streaming=False, `out` is a full string;
            # If streaming=True, it's an iterator of tokens, like in the example script.
            if isinstance(out, str):
                description = out.strip()
            else:
                description = "".join(out).strip()

            descriptions[obj_id] = description

        return descriptions
