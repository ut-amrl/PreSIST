


from instance_segmentation.abs_instance_segmenter import AbsInstanceSegmenter
from dataset_generation.scene_level_captioning.abs_scene_level_captioner import AbstractSceneLevelCaptioner
from dataset_generation.instance_descriptions.abs_object_describer import AbstractObjectDescriber
from dataset_generation.llm_persistence_querying.abs_llm_persistence_querier import AbsLLMPersistenceQuerier
from dataset_generation.intermediate_results_logger.abs_intermediate_results_logger import AbsIntermediateResultsLogger
from dataset_generation.intermediate_results_logger.caching_results_logger2 import CachingResultsLoggerRevised

from abc import ABC, abstractmethod
import torch

from tqdm import tqdm

class AbstractDatasetGeneratorFromDataloader(ABC):


    def generateDataset(self,
        dataloader: torch.utils.data.DataLoader
    ):
        """
        Iterate over the dataloader (which yields dict batches from make_collate_fn),
        unbatch into per-item imageInfo dicts, call processImage on each, and return
        a list of outputs.
        """
        # outputs = []

        # for batch in dataloader:
        for batch in tqdm(dataloader, desc="Processing batches from dataloader"):
            if not isinstance(batch, dict):
                raise TypeError(
                    f"Expected dataloader to yield dict batches, got: {type(batch)}"
                )

            # Determine batch size from a reliable field
            if "index" in batch and torch.is_tensor(batch["index"]):
                bs = int(batch["index"].shape[0])
            elif "path" in batch and isinstance(batch["path"], list):
                bs = len(batch["path"])
            else:
                raise KeyError("Batch is missing 'index' tensor or 'path' list to infer batch size.")

            # Sanity check: all fields should have length bs (for list fields) or shape[0]==bs (for tensor fields)
            for k, v in batch.items():
                if torch.is_tensor(v):
                    if v.shape[0] != bs:
                        raise ValueError(f"Tensor field batch['{k}'] has shape[0]={v.shape[0]} != bs={bs}")
                elif isinstance(v, list):
                    if len(v) != bs:
                        raise ValueError(f"List field batch['{k}'] has len={len(v)} != bs={bs}")
                else:
                    # Your collate_fn shouldn't produce scalars here; fail early if it does
                    raise TypeError(f"Unsupported batch field type for key '{k}': {type(v)}")

            # Unbatch -> per-item dicts -> process
            for i in range(bs):
                imageInfo = {}
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        imageInfo[k] = v[i].item() if v.ndim == 1 else v[i]
                    else:  # list
                        imageInfo[k] = v[i]

                try:
                    out = self.processImage(imageInfo)

                    # print("Done processing image: ", imageInfo.get("image_id", "") + " " + imageInfo.get("path", imageInfo.get("relpath", "unknown")))
                    # if out is not None:
                        # outputs.append(out)
                except Exception as e:
                    print(f"Error processing image {imageInfo.get('image_id', '')} at path {imageInfo.get('path', imageInfo.get('relpath', 'unknown'))}: {e}")
                    # Optionally, you could log this error to a file or logger instead of printing
                # out = self.processImage(imageInfo)
                # print("Done processing image: ", imageInfo.get("image_id", "") + " " + imageInfo.get("path", imageInfo.get("relpath", "unknown")))
                # if out is not None:
                    # outputs.append(out)
        # return outputs


    @abstractmethod
    def processImage(self, imageInfo):
        # imageInfo is dictionary with entries:
        # return {
        #     "index": index,
        #     "component": it.component,
        #     "relpath": it.relpath,
        #     "path": str(abspath),
        #     "image": image,
        #     "image_id": image_id,
        #     "file_size": file_size,
        #     "mtime_ns": mtime_ns,
        # }
        # Should return a dictionary with output entries (can be whatever you want, but should be consistent across calls):

        pass





# Default: run all phases. Subset to e.g. ["segmentation"] or ["segmentation", "object_descriptions"].
DEFAULT_RUN_PHASES = [
    "segmentation",
    "scene_caption",
    "object_descriptions",
    "relationships",
    "persistence",
]

RUN_PHASE_DEPENDENCIES_REVERSE = {
    "segmentation": ["segmentation"],
    "scene_caption": ["scene_caption"],
    "object_descriptions": ["segmentation", "object_descriptions"],
    "relationships": ["segmentation", "object_descriptions", "relationships"],
    "persistence": ["segmentation", "scene_caption", "object_descriptions", "relationships", "  persistence"],
}

class CachingSceneInstanceAndRelationalDatasetGenerator(AbstractDatasetGeneratorFromDataloader):

    def __init__(
            self,
            instanceSegmenter: AbsInstanceSegmenter,
            sceneCaptioner: AbstractSceneLevelCaptioner,
            objectDescriber: AbstractObjectDescriber,
            relationshipDescriber,
            llmPersistenceQuerier: AbsLLMPersistenceQuerier,
            intermediateLogger: CachingResultsLoggerRevised,
            run_phases: list = None,
    ):
        self.instanceSegmenter = instanceSegmenter
        self.sceneCaptioner = sceneCaptioner
        self.objectDescriber = objectDescriber
        self.relationshipDescriber = relationshipDescriber
        self.llmPersistenceQuerier = llmPersistenceQuerier
        self.intermediateLogger = intermediateLogger


        if (run_phases is None) or (len(run_phases) == 0):
            phases_to_run = DEFAULT_RUN_PHASES
        else:
            phases_to_run = []
            for phase in run_phases:
                phases_to_run.extend(RUN_PHASE_DEPENDENCIES_REVERSE[phase])
        self.run_phases = list(set(phases_to_run))

        self.intermediateLogger.configureComponentNames(
            segmenterName=self.instanceSegmenter.getName(),
            sceneCaptionerName=self.sceneCaptioner.getName(),
            objectDescriberName=self.objectDescriber.getName(),
            relationshipDescriberName=self.relationshipDescriber.getName(),
            llmQuerierName=self.llmPersistenceQuerier.getName()
        )


    def processImage(self, imageInfo):
        image = imageInfo["image"]
        timeAndDate = None
        run = self.run_phases
        img_id = imageInfo.get("image_id", imageInfo.get("path", imageInfo.get("relpath", "unknown")))

        self.intermediateLogger.logInputData(image, timeAndDate, imageInfo)

        if self.intermediateLogger.checkIfAlreadyProcessed():
            return

        instances = None
        sceneCaption = None
        objectDescriptions = None
        relationshipDescriptions = None

        # --- Segmentation (run when needed by any later phase) ---
        if ("segmentation" in run):
            try:
                segmentationLambda = lambda: self.instanceSegmenter.segmentInstances(image)
                instances = self.intermediateLogger.getOrGenerateSegmentationResults(segmentationLambda)
            except Exception as e:
                print(f"Skipping image {img_id}: segmentation failed or skipped — {e}")
                return

        if ("scene_caption" in run):
            try:
                captionLambda = lambda: self.sceneCaptioner.generateCaption(image)
                sceneCaption = self.intermediateLogger.getOrGenerateSceneCaption(captionLambda)
            except Exception as e:
                print(f"Skipping image {img_id}: scene caption failed or skipped — {e}")
                return


        if ("object_descriptions" in run):
            try:
                objectDescriptionLambda = lambda: self.objectDescriber.describeObjects(image, instances)
                objectDescriptions = self.intermediateLogger.getOrGenerateObjectDescriptions(objectDescriptionLambda)
            except Exception as e:
                print(f"Skipping image {img_id}: object descriptions failed or skipped — {e}")
                return

        if ("relationships" in run):
            try:
                relationshipDescriberLambda = lambda: self.relationshipDescriber.describeRelationships(image, instances, objectDescriptions)
                relationshipDescriptions = self.intermediateLogger.getOrGenerateRelationshipDescriptions(relationshipDescriberLambda)
            except Exception as e:
                print(f"Skipping image {img_id}: relationship descriptions failed or skipped — {e}")
                return

        if ("persistence" in run):
            try:
                persistenceParamLambda = lambda: self.llmPersistenceQuerier.estimatePersistenceParameters(
                    sceneCaption,
                    objectDescriptions,
                    relationshipDescriptions,
                    timeAndDate,
                    image=image,
                    instances=instances
                )
                prompts, rawResponse, persistenceParameterEstimates = self.intermediateLogger.getOrGenerateLLMPromptsAndOutput(persistenceParamLambda)
                return instances, persistenceParameterEstimates
            except Exception as e:
                print(f"Skipping image {img_id}: persistence estimation failed or skipped — {e}")
                return

        return
