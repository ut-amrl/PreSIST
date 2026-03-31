from dataset_generation.dataset_generation_process import SceneInstanceAndRelationalDatasetGenerator
from instance_segmentation.sam2_auto_segmenter import SAM2InstanceSegmenter
from dataset_generation.semantic_class_classifier import SemanticClassClassifier
from dataset_generation.instance_descriptions.describe_anything_object_describer import DescribeAnythingObjectDescriber
from dataset_generation.scene_level_captioning.instructblip_scene_level_captioner import InstructBLIPSceneLevelCaptioner
from dataset_generation.llm_persistence_querying.prompt_handler.verbose_prompt_handler import VerbosePersistencePromptHandler
from dataset_generation.llm_persistence_querying.llm_querying.qwen_llm_querier import QwenLLMQuerier
from dataset_generation.llm_persistence_querying.chained_prompt_query_persistence_querier import ChainedPromptQueryPersistenceQuerier
from dataset_generation.relationship_descriptions.empty_relationships_describer import EmptyRelationshipDescriber

from dataset_generation.intermediate_results_logger.disk_intermediate_results_logger import DiskIntermediateResultsLogger

import torch


def createSamSegmenter(sam2Checkpoint, sam2Config):
        # Create the segmenter
    return SAM2InstanceSegmenter(
        checkpoint=sam2Checkpoint,
        model_cfg=sam2Config,
        device="cuda",
    )

def createDescribeAnythingDescriber(damPrompt):
    return DescribeAnythingObjectDescriber(
        model_path="nvidia/DAM-3B",
        device="cuda",
        prompt=damPrompt,
        conv_mode="v1",
        prompt_mode="focal_prompt",
        temperature=0.2,
        top_p=0.5,
        num_beams=1,
        max_new_tokens=512,
        streaming=False,  # simpler for this script
    )

def createInstructBlipCaptioner(blipPrompt):
    return InstructBLIPSceneLevelCaptioner(
        model_name="Salesforce/instructblip-flan-t5-xl",
        device="cuda",
        prompt=blipPrompt,
    )


def createLLMPromptHandler():
    return VerbosePersistencePromptHandler(prompt_template_file="prompt_files/llm_querier/v1_real.txt")

def createQwenLLMQuerier():
    return QwenLLMQuerier(torch_dtype=torch.bfloat16)

def createLLMPersistenceQuerier():
    return ChainedPromptQueryPersistenceQuerier(
        promptHandler=createLLMPromptHandler(),
        llmQuerier=createQwenLLMQuerier(),
    )

def createIntermediateLogger(outputDir):
    return DiskIntermediateResultsLogger(logDirectory=outputDir)

def createClipClassifier():
    return SemanticClassClassifier(
        model_name="ViT-L/14",
        device="cuda",
    )

class ConcreteDatasetGenerator(SceneInstanceAndRelationalDatasetGenerator):

    def __init__(
            self,
            sam2Checkpoint: str,
            sam2Config: str,
            damPrompt,
            blipPrompt,
            outputDir: str,
    ):
        super().__init__(
            instanceSegmenter=createSamSegmenter(
                sam2Checkpoint=sam2Checkpoint,
                sam2Config=sam2Config,
            ),
            sceneCaptioner=createInstructBlipCaptioner(
                blipPrompt=blipPrompt
            ),
            objectDescriber=createDescribeAnythingDescriber(
                damPrompt=damPrompt
            ),
            relationshipDescriber=EmptyRelationshipDescriber(),  # Placeholder, implement as needed
            llmPersistenceQuerier=createLLMPersistenceQuerier(),
            intermediateLogger=createIntermediateLogger(outputDir=outputDir),
            clipClassifier=createClipClassifier(),  
        )

