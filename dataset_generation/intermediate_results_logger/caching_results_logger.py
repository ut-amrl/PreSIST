from dataset_generation.intermediate_results_logger.abs_intermediate_results_logger import AbsIntermediateResultsLogger
import os
from PIL import Image
import json

from utils.visualization_utils import saveMasksOnImages

from instance_segmentation.abs_instance_segmenter import SegmentationResults
import torch
import numpy as np

class CachingResultsLogger:

    FULL_RESULTS_DIR = "full_results"
    SEGMENTER_RESULTS_DIR = "segmentation_results"
    SCENE_CAPTIONER_RESULTS_DIR = "scene_caption_results"
    OBJECT_DESCRIBER_RESULTS_DIR = "object_description_results"
    RELATIONSHIP_DESCRIBER_RESULTS_DIR = "relationship_description_results"
    LLM_QUERIER_RESULTS_DIR = "llm_querier_results"

    def __init__(
        self,
        baseLogDirectory,
        saveDebugImages=False,
        forceRegen=False,
        imageDirRoot="/workspace/data/input/"
    ):
        self.baseLogDirectory = baseLogDirectory
        self.saveDebugImages = saveDebugImages
        self.imageDirRoot = imageDirRoot

        self.configFileName = None
        self.imgBaseName = None
        self.imgExtendedName = None
        self.imgFileName = None
        self.image = None
        self.segmenterName = None
        self.sceneCaptionerName = None
        self.objectDescriberName = None
        self.relationshipDescriberName = None
        self.llmQuerierName = None
        self.forceRegen = forceRegen

    def configureComponentNames(
        self,
        segmenterName,
        sceneCaptionerName,
        objectDescriberName,
        relationshipDescriberName,
        llmQuerierName
    ):
        self.segmenterName = segmenterName
        self.sceneCaptionerName = sceneCaptionerName
        self.objectDescriberName = objectDescriberName
        self.relationshipDescriberName = relationshipDescriberName
        self.llmQuerierName = llmQuerierName


    def createDirectoryIfNotExists(self, dirPath):
        # For each part of the path, create directory if it does not exist
        os.makedirs(dirPath, exist_ok=True)

    def generateLogDirectory(self, subDirNames):
        dirPath = self.baseLogDirectory
        for subDirName in subDirNames:
            dirPath = os.path.join(dirPath, subDirName)
        return dirPath


    def getFinalOutputDirectory(self):
        # return self.generateLogDirectory([self.FULL_RESULTS_DIR, self.configFileName, self.imgBaseName])
        return self.generateLogDirectory([self.FULL_RESULTS_DIR, self.configFileName, self.imgExtendedName])


    def getSegmenterResultsDirectory(self):
        # return self.generateLogDirectory([self.SEGMENTER_RESULTS_DIR, self.segmenterName, self.imgBaseName])
        return self.generateLogDirectory([self.SEGMENTER_RESULTS_DIR, self.segmenterName, self.imgExtendedName])


    def getSceneCaptionerResultsDirectory(self):
        # return self.generateLogDirectory([self.SCENE_CAPTIONER_RESULTS_DIR, self.sceneCaptionerName, self.imgBaseName])
        return self.generateLogDirectory([self.SCENE_CAPTIONER_RESULTS_DIR, self.sceneCaptionerName, self.imgExtendedName])


    def getObjectDescriberResultsDirectory(self):
        # return self.generateLogDirectory([self.OBJECT_DESCRIBER_RESULTS_DIR, self.segmenterName, self.objectDescriberName, self.imgBaseName])
        return self.generateLogDirectory([self.OBJECT_DESCRIBER_RESULTS_DIR, self.segmenterName, self.objectDescriberName, self.imgExtendedName])


    def getRelationshipDescriberResultsDirectory(self):
        # return self.generateLogDirectory([self.RELATIONSHIP_DESCRIBER_RESULTS_DIR, self.segmenterName, self.objectDescriberName, self.relationshipDescriberName, self.imgBaseName])
        return self.generateLogDirectory([self.RELATIONSHIP_DESCRIBER_RESULTS_DIR, self.segmenterName, self.objectDescriberName, self.relationshipDescriberName, self.imgExtendedName])


    def logInputData(self, image, timeAndDate, inputBatchInfo):

        # Check that the configured component names are set
        assert self.segmenterName is not None, "Segmenter name not configured"
        assert self.sceneCaptionerName is not None, "Scene captioner name not configured"
        assert self.objectDescriberName is not None, "Object describer name not configured"
        assert self.relationshipDescriberName is not None, "Relationship describer name not configured"
        assert self.llmQuerierName is not None, "LLM querier name not configured"
        assert self.configFileName is not None, "Config file name not configured"

        self.image = image

        fileBaseName = "unknown_image"
        if inputBatchInfo and 'filename' in inputBatchInfo:
            filename = inputBatchInfo['filename']

            # Extract base name without extension

            baseName = os.path.basename(filename)
            fileBaseName, _ = os.path.splitext(baseName)

            self.imgFileName = filename
        self.imgBaseName = fileBaseName

        # Create the imageExtendedName by taking off the imageDirRoot from the filename and replacing os separators with underscores
        if inputBatchInfo and 'filename' in inputBatchInfo:
            filename = inputBatchInfo['filename']
            if filename.startswith(self.imageDirRoot):
                relativePath = os.path.relpath(filename, self.imageDirRoot)
            else:
                relativePath = filename
            self.imgExtendedName = relativePath.replace(os.sep, "_")
        else:
            self.imgExtendedName = fileBaseName


    def getOrGenerateSegmentationResults(self, segmentationLambda):
        segmentationResultsDir = self.getSegmenterResultsDirectory()

        # Segmentation results file is stored as segmentation_results.npz
        segmentationResultsFilePath = os.path.join(segmentationResultsDir, "segmentation_results.npz")

        # If the segmentation results already exist, and forceRegen is False, load and return them
        if not self.forceRegen and os.path.exists(segmentationResultsFilePath):
            print("Retrieving cached segmentation results from: ", segmentationResultsFilePath)

            # Load existing segmentation results
            segmentationResults, _ = SegmentationResults.loadFromFile(segmentationResultsFilePath)
            return segmentationResults

        # Otherwise, compute segmentation results
        segmentationResults = segmentationLambda()

        # Make sure directory exists
        # Create directory if it does not exist
        self.createDirectoryIfNotExists(segmentationResultsDir)

        if (self.saveDebugImages):
            saveMasksOnImages(
                segmentationResults.masks,
                self.image,
                segmentationResultsDir)

        # Save segmentation results to file
        segmentationResults.saveToFile(segmentationResultsFilePath, imagePath=self.imgFileName)
        return segmentationResults


    def getOrGenerateSceneCaption(self, captionLambda):
        captionDir = self.getSceneCaptionerResultsDirectory()
        primaryCaptionFilePath = os.path.join(captionDir, "scene_caption.json")
        alternateCaptionFilePath = os.path.join(captionDir, "scene_caption.txt")

        # If the caption already exists, and forceRegen is False, load and return it
        if not self.forceRegen and os.path.exists(primaryCaptionFilePath):
            print("Retrieving cached scene caption from: ", primaryCaptionFilePath)
            with open(primaryCaptionFilePath, 'r', encoding='utf-8') as f:
                captionData = json.load(f)
            caption = captionData.get("caption", "").strip()

            return caption

        if not self.forceRegen and os.path.exists(alternateCaptionFilePath):
            print("Retrieving cached scene caption from: ", alternateCaptionFilePath)
            with open(alternateCaptionFilePath, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            return caption

        # Otherwise, compute caption
        caption = captionLambda()

        # Save caption to a JSON file
        # Make sure directory exists
        self.createDirectoryIfNotExists(captionDir)

        with open(primaryCaptionFilePath, 'w', encoding='utf-8') as f:
            json.dump({
                "caption": caption,
                "imagePath": self.imgFileName
            }, f, indent=2, ensure_ascii=False)

        return caption


    def getOrGenerateObjectDescriptions(self, descriptionsLambda):
        descriptionsDir = self.getObjectDescriberResultsDirectory()
        descriptionsFilePath = os.path.join(descriptionsDir, "object_descriptions.json")

        # If the descriptions already exist, and forceRegen is False, load and return them
        if not self.forceRegen and os.path.exists(descriptionsFilePath):
            print("Retrieving cached object descriptions from: ", descriptionsFilePath)
            with open(descriptionsFilePath, 'r', encoding='utf-8') as f:
                objectDescriptions = json.load(f)

            # Check if there are two keys ("descriptions" and "imagePath"), and if so, extract descriptions
            if ("descriptions" in objectDescriptions) and ("imagePath" in objectDescriptions):
                objectDescriptions = objectDescriptions["descriptions"]
            return objectDescriptions

        # Otherwise, compute descriptions
        objectDescriptions = descriptionsLambda()

        # Save object descriptions to a JSON file
        # Make sure directory exists
        self.createDirectoryIfNotExists(descriptionsDir)

        with open(descriptionsFilePath, 'w', encoding='utf-8') as f:
            descriptionsOutputData = {
                "descriptions": objectDescriptions,
                "imagePath": self.imgFileName
            }
            json.dump(descriptionsOutputData, f, indent=2, ensure_ascii=False)

        return objectDescriptions

    def getOrGenerateRelationshipDescriptions(self, relationshipLambda):

        # For now, just return the computed relationships without caching
        relationshipDescriptions = relationshipLambda()
        return relationshipDescriptions


    def getOrGenerateLLMPromptsAndOutput(self, llmLambda):

        llmDataDir = self.getFinalOutputDirectory()

        promptsFilePath = os.path.join(llmDataDir, "llm_prompt.txt")
        persistenceParamsPath = os.path.join(llmDataDir, "persistence_parameters.json")
        rawResponsePath = os.path.join(llmDataDir, "llm_raw_response.txt")

        # If both the prompts and persistence parameters already exist, and forceRegen is False, load and return them
        if (not self.forceRegen) and os.path.exists(promptsFilePath) and os.path.exists(persistenceParamsPath) and os.path.exists(rawResponsePath):
            print("Retrieving cached LLM prompts and persistence parameters from: ", llmDataDir)
            with open(promptsFilePath, 'r', encoding='utf-8') as f:
                prompts = [f.read().strip()]

            with open(rawResponsePath, 'r', encoding='utf-8') as f:
                rawResponse = f.read().strip()

            with open(persistenceParamsPath, 'r', encoding='utf-8') as f:
                persistenceParameters = json.load(f)

                # If the persistenceParameters is a dict and has a key "imagePath" and "persistenceParams", extract the value of "persistenceParameters"
                if isinstance(persistenceParameters, dict) and "persistenceParams" in persistenceParameters:
                    persistenceParameters = persistenceParameters["persistenceParams"]
            return prompts, rawResponse, persistenceParameters

        # Otherwise, compute prompts and persistence parameters
        prompts, rawResponse, persistenceParameters = llmLambda()

        # Save prompts and persistence parameters to their respective files
        self.createDirectoryIfNotExists(llmDataDir)

        with open(promptsFilePath, 'w', encoding='utf-8') as f:
            f.write(prompts[0] + "\n\n")
        with open(rawResponsePath, 'w', encoding='utf-8') as f:
            f.write(rawResponse + "\n\n")
        with open(persistenceParamsPath, 'w', encoding='utf-8') as f:
            persistenceDataOut = {
                "persistenceParams": persistenceParameters,
                "imagePath": self.imgFileName
            }
            json.dump(persistenceDataOut, f, indent=2, ensure_ascii=False)
            # json.dump(persistenceParameters, f, indent=2, ensure_ascii=False)

        return prompts, rawResponse, persistenceParameters

    # def logObjectDescriptions(self, objectDescriptions):
    #     # Save object descriptions to a JSON file

    #     descriptionsDir = self.generateLogDirectory([self.imgBaseName])
    #     # Make sure directory exists
    #     if not os.path.exists(descriptionsDir):
    #         os.makedirs(descriptionsDir)
    #     descriptionsPath = os.path.join(descriptionsDir, "object_descriptions.json")
    #     with open(descriptionsPath, 'w', encoding='utf-8') as f:
    #         json.dump(objectDescriptions, f, indent=2, ensure_ascii=False)

    # def logRelationshipDescriptions(self, relationshipDescriptions):
    #     pass

    # def logPrompts(self, prompts):

    #     if (not prompts) or (len(prompts) == 0):
    #         return

    #     # Save prompts to a text file
    #     promptsDir = self.generateLogDirectory([self.imgBaseName])

    #     # If there is only one prompt, save as llm_prompt.txt
    #     if (len(prompts) == 1):
    #         promptsPath = os.path.join(promptsDir, "llm_prompt.txt")
    #         with open(promptsPath, 'w', encoding='utf-8') as f:
    #             f.write(prompts[0] + "\n\n")
    #         return

    #     # If there are multiple prompts, save each with an index
    #     if not os.path.exists(promptsDir):
    #         os.makedirs(promptsDir)
    #     for idx, prompt in enumerate(prompts):
    #         promptsPath = os.path.join(promptsDir, f"llm_prompt_{idx:03d}.txt")
    #         with open(promptsPath, 'w', encoding='utf-8') as f:
    #             f.write(prompt + "\n\n")

    # def logPersistenceParameters(self, persistenceParameters):
    #     # Convert to JSON and save to file
    #     parametersDir = self.generateLogDirectory([self.imgBaseName])
    #     # Make sure directory exists
    #     if not os.path.exists(parametersDir):
    #         os.makedirs(parametersDir)
    #     parametersPath = os.path.join(parametersDir, "persistence_parameters.json")
    #     with open(parametersPath, 'w', encoding='utf-8') as f:
    #         json.dump(persistenceParameters, f, indent=2, ensure_ascii=False)