from dataset_generation.scene_level_captioning.abs_scene_level_captioner import AbstractSceneLevelCaptioner
from dataset_generation.intermediate_results_logger.debug_data import DebugPayload
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from typing import Optional, Tuple

from utils.threading_utils import ForwardSerializeLock



class InstructBLIPSceneLevelCaptioner(AbstractSceneLevelCaptioner):
    def __init__(
        self,
        model_name: str = "Salesforce/instructblip-flan-t5-xl",
        device: str | None = None,
        prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        max_new_tokens: int = 256,
        num_beams: int = 5,
        name: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model_name : str
            Hugging Face model id for InstructBLIP.
            Examples: "Salesforce/instructblip-flan-t5-xl",
                      "Salesforce/instructblip-flan-t5-xxl",
                      "Salesforce/instructblip-vicuna-7b", etc.
        device : str | None
            "cuda", "cpu", or "cuda:0", etc. If None, auto-detect.
        instruction : str
            Instruction prompt passed to InstructBLIP, e.g.
            "Describe the scene in detail."
        max_new_tokens : int
            Maximum number of tokens to generate.
        num_beams : int
            Beam search width for generation.
        """
        if (name is None):
            raise ValueError("name must be provided for InstructBLIPSceneLevelCaptioner")
        super().__init__(name)

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if (prompt is not None) and (prompt_file is not None):
            raise ValueError("Only one of prompt or prompt_file should be provided.")
        if (prompt_file is None) and (prompt is None):
            prompt = "Describe the scene in detail."
        elif prompt_file is not None:
            with open(prompt_file, 'r') as f:
                prompt = f.read()
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.initialized = False
        # Defense-in-depth: PyTorch-based local model. Single-threaded today;
        # the lock keeps it correct under future cross-scene threading and
        # warns once if that ever happens.
        self._forward_lock = ForwardSerializeLock(
            label=f"{type(self).__name__}.generateCaption",
            advice="InstructBLIP is local-only; can't be parallelized across "
                   "host threads on one GPU. Run from a single-threaded driver.",
        )
        # self.initialize()



    def initialize(self):
        self.initialized = True

        # Load processor and model from HF Hub
        self.processor = InstructBlipProcessor.from_pretrained(self.model_name)

        # Use fp16 on GPU, fp32 on CPU
        dtype = torch.float16 if "cuda" in self.device else torch.float32
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generateCaption(self, image: torch.Tensor, image_id: str) -> Tuple[str, DebugPayload]:
        """
        Generate a scene-level caption for the given image using InstructBLIP.
        Takes in a torch tensor image (C, H, W) or (B, C, H, W) and returns (caption, DebugPayload).
        """
        if not self.initialized:
            self.initialize()

        if image.dim() == 3:
            # (C, H, W) -> (1, C, H, W)
            image = image.unsqueeze(0)

        print("Prompt: ", self.prompt)

        # InstructBlipProcessor can take torch tensors directly
        inputs = self.processor(
            images=image,
            text=self.prompt,
            return_tensors="pt",
            # do_rescale=False,
        )

        # Move to device and match dtype for float tensors
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    inputs[k] = v.to(self.device, dtype=self.model.dtype)
                else:
                    inputs[k] = v.to(self.device)

        with self._forward_lock.acquire():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1.0,
            )

        caption = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

        return caption, DebugPayload.empty()



# class InstructBLIPSceneLevelCaptioner(AbstractSceneLevelCaptioner):
#     def __init__(self, model_name="Salesforce/instructblip-flan-t5-xl", device="cuda", prompt="Describe this scene."):
#         """
#         Initialize an InstructBLIP captioner.
#         The constructor accepts only configuration parameters, not the model itself.
#         """
#         super().__init__()
#         self.device = device
#         self.prompt = prompt

#         # Load processor + model internally (not passed in)
#         self.processor = InstructBlipProcessor.from_pretrained(model_name)
#         self.model = InstructBlipForConditionalGeneration.from_pretrained(model_name)
#         self.model.to(device)

#     def generateCaption(self, image):
#         """
#         Generate a scene-level caption for the given image using InstructBLIP.
#         Takes in a torch tensor image and returns a string caption.
#         """

#         inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)

#         with torch.no_grad():
#             output = self.model.generate(
#                 **inputs,
#                 max_new_tokens=64,
#                 do_sample=False
#             )

#         caption = self.processor.batch_decode(output, skip_special_tokens=True)[0]
#         return caption


