


from instance_segmentation.abs_instance_segmenter import AbsInstanceSegmenter
from dataset_generation.scene_level_captioning.abs_scene_level_captioner import AbstractSceneLevelCaptioner
from dataset_generation.instance_descriptions.abs_object_describer import AbstractObjectDescriber
from dataset_generation.llm_persistence_querying.abs_llm_persistence_querier import AbsLLMPersistenceQuerier
from dataset_generation.intermediate_results_logger.abs_intermediate_results_logger import AbsIntermediateResultsLogger
from dataset_generation.intermediate_results_logger.caching_results_logger2 import CachingResultsLoggerRevised
from utils.image_utils import LazyImage
from utils.timing import timer

from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import time
from typing import Optional

import torch

from tqdm import tqdm

class AbstractDatasetGeneratorFromDataloader(ABC):

    def __init__(
        self,
        periodic_report_every_n_images: int = 50,
        periodic_report_every_minutes: float = 20.0,
        periodic_report_path: Optional[str] = None,
        scene_concurrency: int = 4,
    ):
        """
        periodic_report_every_n_images: dump timing report after this many
            processed images. Set to 0 to disable the count trigger.
        periodic_report_every_minutes: dump timing report after this many wall-
            clock minutes since the last dump. Set to 0 to disable.
        periodic_report_path: where to write the periodic snapshot. None means
            stdout only (the default). When a path is given, periodic dumps go
            to that file (overwriting) and stdout is suppressed for them.
        scene_concurrency: how many scenes to process concurrently *within one
            worker process* via a per-batch ThreadPoolExecutor. Default 1 =
            current behavior (strictly serial). Threading is only correct now
            that the IntermediateResultsLogger holds no per-image state — the
            ImageContext refactor (see caching_results_logger2.py) is what
            makes >1 safe. Set higher to fan out per-scene VLM/HTTP calls
            (the captioning pass benefits the most: each scene is one HTTP
            call, so cross-scene threading is the only way to parallelize it
            within a worker).
        """
        self._periodic_n = periodic_report_every_n_images
        self._periodic_minutes = periodic_report_every_minutes
        self._periodic_path = periodic_report_path
        self._images_since_periodic_report = 0
        self._last_periodic_report_time = time.monotonic()
        if scene_concurrency < 1:
            raise ValueError(
                f"scene_concurrency must be >= 1, got {scene_concurrency}"
            )
        self._scene_concurrency = int(scene_concurrency)

    def _maybe_emit_periodic_timing_report(self):
        self._images_since_periodic_report += 1
        now = time.monotonic()
        elapsed_min = (now - self._last_periodic_report_time) / 60.0
        count_trip = (
            self._periodic_n > 0
            and self._images_since_periodic_report >= self._periodic_n
        )
        time_trip = (
            self._periodic_minutes > 0
            and elapsed_min >= self._periodic_minutes
        )
        if not (count_trip or time_trip):
            return
        if self._periodic_path is not None:
            timer.report(output_file_path=self._periodic_path, to_stdout=False)
        else:
            timer.report()
        self._images_since_periodic_report = 0
        self._last_periodic_report_time = now

    def generateDataset(self,
        dataloader: torch.utils.data.DataLoader
    ):
        """
        Iterate over the dataloader (which yields dict batches from
        make_collate_fn), unbatch into per-item imageInfo dicts, and run
        processImage on each.

        When ``scene_concurrency > 1``, scenes are dispatched through a single
        long-lived ThreadPoolExecutor with a bounded sliding window of K
        in-flight futures: as each one completes, a new scene is submitted,
        regardless of which batch it came from. This keeps workers
        continuously busy across batch boundaries (so the design works even
        when ``batch_size=1``, which is the project's default).

        Per-future tqdm updates: progress is tracked at scene-level rather
        than batch-level. The bar is determinate when the dataset exposes
        ``len()``; otherwise it shows count + rate without an ETA.

        ``_safely_process_image`` swallows per-scene exceptions so a single
        bad scene doesn't poison the rest of the run.
        """
        print(
            f"[generateDataset] scene_concurrency={self._scene_concurrency} "
            f"({'serial' if self._scene_concurrency <= 1 else 'sliding-window threaded'})"
        )

        # Determinate progress when the dataset exposes ``len()``; otherwise
        # tqdm shows count + rate without a total/ETA.
        total = None
        ds = getattr(dataloader, "dataset", None)
        try:
            if ds is not None:
                total = len(ds)
        except TypeError:
            pass

        pbar = tqdm(total=total, desc="Processing images", unit="image")
        try:
            if self._scene_concurrency <= 1:
                # Serial path — direct in-thread call; tracebacks come straight
                # from the calling thread for clean debugging.
                for batch in dataloader:
                    for imageInfo in self._unbatch(batch):
                        self._safely_process_image(imageInfo)
                        self._maybe_emit_periodic_timing_report()
                        pbar.update(1)
                return

            # Sliding-window threaded path. ``inflight`` is a set of currently
            # in-flight futures, capped at K by the ``while >= K`` gate before
            # each submit. ``wait(..., FIRST_COMPLETED)`` blocks until at
            # least one finishes, freeing slots; multiple futures may finish
            # at once, in which case they all flow through one drain_one call.
            K = self._scene_concurrency
            inflight = set()

            def drain_one():
                nonlocal inflight
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done:
                    # _safely_process_image swallows exceptions; .result()
                    # would still propagate any unexpected error from inside
                    # the executor itself, so wrap defensively.
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"Error draining scene future: {e}")
                    self._maybe_emit_periodic_timing_report()
                    pbar.update(1)

            with ThreadPoolExecutor(
                max_workers=K,
                thread_name_prefix="scene_worker",
            ) as ex:
                for batch in dataloader:
                    for imageInfo in self._unbatch(batch):
                        # Throttle submissions so we never exceed K in-flight.
                        # Without this, the executor would buffer every
                        # imageInfo (and therefore every LazyImage) for the
                        # whole dataset.
                        while len(inflight) >= K:
                            drain_one()
                        inflight.add(ex.submit(self._safely_process_image, imageInfo))

                # End-of-stream drain.
                while inflight:
                    drain_one()
        finally:
            pbar.close()

    def _unbatch(self, batch):
        """Yield per-item imageInfo dicts from a collated batch dict.

        Validates the batch has consistent sizes across fields and yields one
        dict per item, suitable for passing to ``processImage``.
        """
        if not isinstance(batch, dict):
            raise TypeError(
                f"Expected dataloader to yield dict batches, got: {type(batch)}"
            )

        # Determine batch size from a reliable field.
        if "index" in batch and torch.is_tensor(batch["index"]):
            bs = int(batch["index"].shape[0])
        elif "path" in batch and isinstance(batch["path"], list):
            bs = len(batch["path"])
        else:
            raise KeyError(
                "Batch is missing 'index' tensor or 'path' list to infer batch size."
            )

        # Sanity check: all fields should have length bs (for lists) or
        # shape[0]==bs (for tensors).
        for k, v in batch.items():
            if torch.is_tensor(v):
                if v.shape[0] != bs:
                    raise ValueError(
                        f"Tensor field batch['{k}'] has shape[0]={v.shape[0]} != bs={bs}"
                    )
            elif isinstance(v, list):
                if len(v) != bs:
                    raise ValueError(
                        f"List field batch['{k}'] has len={len(v)} != bs={bs}"
                    )
            else:
                # The collate_fn shouldn't produce scalars here; fail early.
                raise TypeError(
                    f"Unsupported batch field type for key '{k}': {type(v)}"
                )

        for i in range(bs):
            imageInfo = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    imageInfo[k] = v[i].item() if v.ndim == 1 else v[i]
                else:  # list
                    imageInfo[k] = v[i]
            yield imageInfo

    def _safely_process_image(self, imageInfo):
        """processImage wrapper that logs and swallows exceptions, matching
        the pre-refactor inline ``try/except`` behavior. Used by both serial
        and threaded paths so scene-level error handling stays identical."""
        try:
            return self.processImage(imageInfo)
        except Exception as e:
            print(
                f"Error processing image "
                f"{imageInfo.get('image_id', '')} at path "
                f"{imageInfo.get('path', imageInfo.get('relpath', 'unknown'))}: {e}"
            )
            return None


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
            periodic_report_every_n_images: int = 50,
            periodic_report_every_minutes: float = 20.0,
            periodic_report_path: Optional[str] = None,
            scene_concurrency: int = 4,
    ):
        super().__init__(
            periodic_report_every_n_images=periodic_report_every_n_images,
            periodic_report_every_minutes=periodic_report_every_minutes,
            periodic_report_path=periodic_report_path,
            scene_concurrency=scene_concurrency,
        )
        self.instanceSegmenter = instanceSegmenter
        self.sceneCaptioner = sceneCaptioner
        self.objectDescriber = objectDescriber
        self.relationshipDescriber = relationshipDescriber
        self.llmPersistenceQuerier = llmPersistenceQuerier
        self.intermediateLogger = intermediateLogger

        # run_phases = ["segmentation", "scene_caption", "object_descriptions"]
        # run_phases = ["segmentation"]
        # run_phases = ["scene_caption", "object_descriptions"]


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
        timeAndDate = None
        run = self.run_phases
        img_id = imageInfo.get("image_id", imageInfo.get("path", imageInfo.get("relpath", "unknown")))

        lazy_image = imageInfo["image"]
        # Build a per-image context object instead of mutating per-image state
        # on the logger. ctx is the only carrier of per-image identity through
        # this call; the logger itself holds no per-image fields.
        ctx = self.intermediateLogger.buildImageContext(lazy_image, timeAndDate, imageInfo)

        print("Image id: ", img_id)

        if self.intermediateLogger.checkIfAlreadyProcessed(ctx):
            return

        instances = None
        sceneCaption = None
        objectDescriptions = None
        relationshipDescriptions = None


        # --- Segmentation (run when needed by any later phase) ---
        if ("segmentation" in run):
            try:
                segmentationLambda = lambda: self.instanceSegmenter.segmentInstances(lazy_image.get(), image_id=ctx.image_id)
                instances = self.intermediateLogger.getOrGenerateSegmentationResults(ctx, segmentationLambda)
            except Exception as e:
                print(f"Skipping image {img_id}: segmentation failed or skipped — {e}")
                return

        if ("scene_caption" in run):
            try:
                captionLambda = lambda: self.sceneCaptioner.generateCaption(lazy_image.get(), image_id=ctx.image_id)
                sceneCaption = self.intermediateLogger.getOrGenerateSceneCaption(ctx, captionLambda)
            except Exception as e:
                print(f"Skipping image {img_id}: scene caption failed or skipped — {e}")
                return


        if ("object_descriptions" in run):
            try:
                objectDescriptionLambda = lambda: self.objectDescriber.describeObjects(lazy_image.get(), instances, image_id=ctx.image_id)
                objectDescriptions = self.intermediateLogger.getOrGenerateObjectDescriptions(ctx, objectDescriptionLambda)
            except Exception as e:
                print(f"Skipping image {img_id}: object descriptions failed or skipped — {e}")
                return

        if ("relationships" in run):
            try:
                relationshipDescriberLambda = lambda: self.relationshipDescriber.describeRelationships(lazy_image.get(), instances, objectDescriptions, image_id=ctx.image_id)
                relationshipDescriptions = self.intermediateLogger.getOrGenerateRelationshipDescriptions(ctx, relationshipDescriberLambda)
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
                    image=lazy_image.get(),
                    instances=instances,
                    image_id=ctx.image_id,
                )
                prompts, rawResponse, persistenceParameterEstimates = self.intermediateLogger.getOrGenerateLLMPromptsAndOutput(ctx, persistenceParamLambda)
                return instances, persistenceParameterEstimates
            except Exception as e:
                print(f"Skipping image {img_id}: persistence estimation failed or skipped — {e}")
                return

        return
