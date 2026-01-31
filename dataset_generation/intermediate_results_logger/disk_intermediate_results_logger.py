from dataset_generation.intermediate_results_logger.abs_intermediate_results_logger import AbsIntermediateResultsLogger
import os
from PIL import Image
import json

from utils.visualization_utils import overlay_single_mask

class DiskIntermediateResultsLogger(AbsIntermediateResultsLogger):

    def __init__(self, logDirectory):
        self.logDirectory = logDirectory
        self.imgBaseName = None
        self.image = None


    def generateLogDirectory(self, subDirNames):
        dirPath = self.logDirectory
        for subDirName in subDirNames:
            dirPath = os.path.join(dirPath, subDirName)
            if not os.path.exists(dirPath):
                os.mkdir(dirPath)
        return dirPath


    def logInputData(self, image, timeAndDate, inputBatchInfo):
        self.image = image

        fileBaseName = "unknown_image"
        if inputBatchInfo and 'filename' in inputBatchInfo:
            filename = inputBatchInfo['filename']

            # Extract base name without extension

            baseName = os.path.basename(filename)
            fileBaseName, _ = os.path.splitext(baseName)

        self.imgBaseName = fileBaseName


    def logSegmentationResults(self, segmentationResults):
        segmentationResultsDir = self.generateLogDirectory([self.imgBaseName, "segmentation_results"])
        # Make sure directory exists
        if not os.path.exists(segmentationResultsDir):
            os.makedirs(segmentationResultsDir)
        # Save each mask overlaied on the original image
        for idx, mask in enumerate(segmentationResults.masks):
            overlay_np = overlay_single_mask(
                self.image,
                mask,
                alpha_mask=0.4,
                alpha_img=0.6,
                color_bgr=(0, 0, 255),
            )
            overlay_path = os.path.join(segmentationResultsDir, f"mask_{idx:03d}.png")
            overlay_pil = Image.fromarray(overlay_np)
            overlay_pil.save(overlay_path)

    def logSceneCaption(self, caption):
        # Save caption to a text file
        captionDir = self.generateLogDirectory([self.imgBaseName])
        # Make sure directory exists
        if not os.path.exists(captionDir):
            os.makedirs(captionDir)
        captionPath = os.path.join(captionDir, "scene_caption.txt")
        with open(captionPath, 'w', encoding='utf-8') as f:
            f.write(caption + "\n")

    def logObjectDescriptions(self, objectDescriptions):
        # Save object descriptions to a JSON file

        descriptionsDir = self.generateLogDirectory([self.imgBaseName])
        # Make sure directory exists
        if not os.path.exists(descriptionsDir):
            os.makedirs(descriptionsDir)
        descriptionsPath = os.path.join(descriptionsDir, "object_descriptions.json")
        with open(descriptionsPath, 'w', encoding='utf-8') as f:
            json.dump(objectDescriptions, f, indent=2, ensure_ascii=False)

    def logRelationshipDescriptions(self, relationshipDescriptions):
        pass

    def logPrompts(self, prompts):

        if (not prompts) or (len(prompts) == 0):
            return

        # Save prompts to a text file
        promptsDir = self.generateLogDirectory([self.imgBaseName])

        # If there is only one prompt, save as llm_prompt.txt
        if (len(prompts) == 1):
            promptsPath = os.path.join(promptsDir, "llm_prompt.txt")
            with open(promptsPath, 'w', encoding='utf-8') as f:
                f.write(prompts[0] + "\n\n")
            return

        # If there are multiple prompts, save each with an index
        if not os.path.exists(promptsDir):
            os.makedirs(promptsDir)
        for idx, prompt in enumerate(prompts):
            promptsPath = os.path.join(promptsDir, f"llm_prompt_{idx:03d}.txt")
            with open(promptsPath, 'w', encoding='utf-8') as f:
                f.write(prompt + "\n\n")

    def logPersistenceParameters(self, persistenceParameters):
        # Convert to JSON and save to file
        parametersDir = self.generateLogDirectory([self.imgBaseName])
        # Make sure directory exists
        if not os.path.exists(parametersDir):
            os.makedirs(parametersDir)
        parametersPath = os.path.join(parametersDir, "persistence_parameters.json")
        with open(parametersPath, 'w', encoding='utf-8') as f:
            json.dump(persistenceParameters, f, indent=2, ensure_ascii=False)