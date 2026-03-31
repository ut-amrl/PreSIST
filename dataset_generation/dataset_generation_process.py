import numpy as np

from abc import ABC, abstractmethod

from persistence_param_dataset.persistence_dataset_types import DatasetResult, ImageWithMetadata
from dataset_generation.images_dataset.query_image_dataset import AbsQueryImageDataset
from utils.image_utils import loadImageAsTensor
from instance_segmentation.abs_instance_segmenter import AbsInstanceSegmenter
from dataset_generation.scene_level_captioning.abs_scene_level_captioner import AbstractSceneLevelCaptioner
from dataset_generation.instance_descriptions.abs_object_describer import AbstractObjectDescriber
from dataset_generation.llm_persistence_querying.abs_llm_persistence_querier import AbsLLMPersistenceQuerier
from dataset_generation.intermediate_results_logger.abs_intermediate_results_logger import AbsIntermediateResultsLogger

class AbstractDatasetGenerator(ABC):

    @abstractmethod
    def processImage(self, image):
        pass

    def generateDataset(self,
        queryImageDataset: AbsQueryImageDataset
    ):
        imagesWithMetadata = queryImageDataset.getImagesWithMetadata()
        dataset = []
        for imageWithMetadata in imagesWithMetadata:
            imageFileName = imageWithMetadata.filename
            # Load the image from file
            image = loadImageAsTensor(imageFileName)
            data_entry = self.processImage(image)
            datasetResult = DatasetResult(
                imageWithMetadata=imageWithMetadata,
                instanceMasks=data_entry[0],
                persistenceParameterEstimates=data_entry[1]
            )
            dataset.append(datasetResult)
        return dataset

    def saveDataset(self, dataset, file_path):
        # Throw an error for now (not implemented)
        raise NotImplementedError("saveDataset not yet implemented")

class SceneInstanceAndRelationalDatasetGenerator(AbstractDatasetGenerator):

    def __init__(
            self,
            instanceSegmenter: AbsInstanceSegmenter,
            sceneCaptioner: AbstractSceneLevelCaptioner,
            objectDescriber: AbstractObjectDescriber,
            relationshipDescriber,
            llmPersistenceQuerier: AbsLLMPersistenceQuerier,
            intermediateLogger: AbsIntermediateResultsLogger,
            clipClassifier,
    ):
        self.instanceSegmenter = instanceSegmenter
        self.sceneCaptioner = sceneCaptioner
        self.objectDescriber = objectDescriber
        self.relationshipDescriber = relationshipDescriber
        self.llmPersistenceQuerier = llmPersistenceQuerier
        self.intermediateLogger = intermediateLogger
        self.clipClassifier = clipClassifier 

    def processImage(self, image, timeAndDate=None, inputBatchInfo=None):

        print("Processing new image...")
        self.intermediateLogger.logInputData(image, timeAndDate, inputBatchInfo)
        print("Generating scene caption and descriptions...")
        sceneCaption = self.sceneCaptioner.generateCaption(image)
        self.intermediateLogger.logSceneCaption(sceneCaption)
        print("Segmenting instances...")
        instances = self.instanceSegmenter.segmentInstances(image)
        self.intermediateLogger.logSegmentationResults(instances)
        print("Classifying semantic classes...")
        image_np = np.array(image)
        masks = [inst.mask for inst in instances]  
        clip_results = self.clipClassifier.classify_all_instances(image_np, masks)
        clip_by_id = {r["instance_id"]: r for r in clip_results}
        print("Generating object descriptions...")
        objectDescriptions = self.objectDescriber.describeObjects(image, instances)
        self.intermediateLogger.logObjectDescriptions(objectDescriptions)
        print("Generating relationship descriptions...")
        relationshipDescriptions = self.relationshipDescriber.describeRelationships(image, instances, objectDescriptions)
        self.intermediateLogger.logRelationshipDescriptions(relationshipDescriptions)
        print("Estimating persistence parameters...")
        prompts, rawResponse, persistenceParameterEstimates = self.llmPersistenceQuerier.estimatePersistenceParameters(sceneCaption, objectDescriptions, relationshipDescriptions, timeAndDate)
        self.intermediateLogger.logPrompts(prompts)
        self.intermediateLogger.logPersistenceParameters(persistenceParameterEstimates)

        return instances, persistenceParameterEstimates