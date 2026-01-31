from dataset_generation.concrete_dataset_generation_process import ConcreteDatasetGenerator
import argparse
from PIL import Image
import numpy as np
import os
import torch


def main(image_path, sam2_checkpoint, sam2_config, dam_prompt, blip_prompt, output_dir):

    # ---- load image ----
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)  # (H, W, 3), uint8
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W), uint8

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    dataset_generator = ConcreteDatasetGenerator(
        sam2Checkpoint=sam2_checkpoint,
        sam2Config=sam2_config,
        damPrompt=dam_prompt,
        blipPrompt=blip_prompt,
        outputDir=output_dir,
    )

    dataset_generator.processImage(
        image=img_tensor,
        timeAndDate=None,
        inputBatchInfo={'filename':image_path}
    )

def arg_parse():
    parser = argparse.ArgumentParser(
        description="Run SAM2 + DAM on a single image and save per-instance overlays and descriptions."
    )
    parser.add_argument("--image_path", type=str, help="Path to the input image.")
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        required=True,
        help="Path to the SAM2 model checkpoint.",
    )
    parser.add_argument(
        "--sam_model_cfg",
        type=str,
        required=True,
        help="Path to the SAM2 model config file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug_outputs/full_pipeline_outputs",
        help="Directory to save overlay images and descriptions.json.",
    )
    parser.add_argument(
        "--dam_prompt",
        type=str,
        default="<image>\nDescribe the masked region in detail.",
        help="Prompt passed to DAM.",
    )
    parser.add_argument(
        "--blip_prompt",
        type=str,
        default="Describe this scene.",
        help="Prompt passed to InstructBLIP.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()

    main(
        image_path=args.image_path,
        sam2_checkpoint=args.sam_checkpoint,
        sam2_config=args.sam_model_cfg,
        dam_prompt=args.dam_prompt,
        blip_prompt=args.blip_prompt,
        output_dir=args.output_dir,
    )

