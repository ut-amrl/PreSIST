
from dataset_generation.dataset_generation_process import AbstractDatasetGenerator

from persistence_param_dataset.persistence_dataset_types import DatasetResult, ImageWithMetadata
from dataset_generation.images_dataset.query_image_dataset import AbsQueryImageDataset
from utils.image_utils import loadImageAsTensor
from instance_segmentation.abs_instance_segmenter import AbsInstanceSegmenter
from dataset_generation.scene_level_captioning.abs_scene_level_captioner import AbstractSceneLevelCaptioner
from dataset_generation.instance_descriptions.abs_object_describer import AbstractObjectDescriber
from dataset_generation.llm_persistence_querying.abs_llm_persistence_querier import AbsLLMPersistenceQuerier
from dataset_generation.intermediate_results_logger.abs_intermediate_results_logger import AbsIntermediateResultsLogger
from dataset_generation.intermediate_results_logger.caching_results_logger import CachingResultsLogger


class CachingSceneInstanceAndRelationalDatasetGenerator(AbstractDatasetGenerator):

    def __init__(
            self,
            instanceSegmenter: AbsInstanceSegmenter,
            sceneCaptioner: AbstractSceneLevelCaptioner,
            objectDescriber: AbstractObjectDescriber,
            relationshipDescriber,
            llmPersistenceQuerier: AbsLLMPersistenceQuerier,
            intermediateLogger: CachingResultsLogger
    ):
        self.instanceSegmenter = instanceSegmenter
        self.sceneCaptioner = sceneCaptioner
        self.objectDescriber = objectDescriber
        self.relationshipDescriber = relationshipDescriber
        self.llmPersistenceQuerier = llmPersistenceQuerier
        self.intermediateLogger = intermediateLogger

        self.intermediateLogger.configureComponentNames(
            segmenterName=self.instanceSegmenter.getName(),
            sceneCaptionerName=self.sceneCaptioner.getName(),
            objectDescriberName=self.objectDescriber.getName(),
            relationshipDescriberName=self.relationshipDescriber.getName(),
            llmQuerierName=self.llmPersistenceQuerier.getName()
        )


    def processImage(self, image, timeAndDate=None, inputBatchInfo=None):

        print("Processing new image...")
        self.intermediateLogger.logInputData(image, timeAndDate, inputBatchInfo)

        # print("Generating scene caption and descriptions...")
        # # sceneCaption = self.sceneCaptioner.generateCaption(image)
        # # self.intermediateLogger.logSceneCaption(sceneCaption)
        # captionLambda = lambda: self.sceneCaptioner.generateCaption(image)
        # sceneCaption = self.intermediateLogger.getOrGenerateSceneCaption(captionLambda)

        print("Segmenting instances...")
        # instances = self.instanceSegmenter.segmentInstances(image)
        # self.intermediateLogger.logSegmentationResults(instances)
        segmentationLambda = lambda: self.instanceSegmenter.segmentInstances(image)
        instances = self.intermediateLogger.getOrGenerateSegmentationResults(segmentationLambda)

        return instances, None

        print("Generating object descriptions...")
        # objectDescriptions = self.objectDescriber.describeObjects(image, instances)
        # self.intermediateLogger.logObjectDescriptions(objectDescriptions)
        objectDescriptionLambda = lambda: self.objectDescriber.describeObjects(image, instances)
        objectDescriptions = self.intermediateLogger.getOrGenerateObjectDescriptions(objectDescriptionLambda)

        print("Generating relationship descriptions...")
        # relationshipDescriptions = self.relationshipDescriber.describeRelationships(image, instances, objectDescriptions)
        # self.intermediateLogger.logRelationshipDescriptions(relationshipDescriptions)
        relationshipDescriberLambda = lambda: self.relationshipDescriber.describeRelationships(image, instances, objectDescriptions)
        relationshipDescriptions = self.intermediateLogger.getOrGenerateRelationshipDescriptions(relationshipDescriberLambda)

        print("Estimating persistence parameters...")
        # prompts, persistenceParameterEstimates = self.llmPersistenceQuerier.estimatePersistenceParameters(sceneCaption, objectDescriptions, relationshipDescriptions, timeAndDate)
        # self.intermediateLogger.logPrompts(prompts)
        # self.intermediateLogger.logPersistenceParameters(persistenceParameterEstimates)
        persistenceParamLambda = lambda: self.llmPersistenceQuerier.estimatePersistenceParameters(sceneCaption, objectDescriptions, relationshipDescriptions, timeAndDate)
        prompts, rawResponse, persistenceParameterEstimates = self.intermediateLogger.getOrGenerateLLMPromptsAndOutput(persistenceParamLambda)

        return instances, persistenceParameterEstimates
