from dataset_generation.scene_level_captioning.abs_scene_level_captioner import AbstractSceneLevelCaptioner
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from typing import Optional


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




class DummySceneLevelCaptioner(AbstractSceneLevelCaptioner):
    def __init__(
        self
    ):
        super().__init__("DummySceneLevelCaptioner")

    @torch.no_grad()
    def generateCaption(self, image: torch.Tensor) -> str:
        return ""