# tools/test_instructblip_on_single_image.py

import os
import argparse

import torch
from PIL import Image
import torchvision.transforms as T

from dataset_generation.scene_level_captioning.instructblip_scene_level_captioner import (
    InstructBLIPSceneLevelCaptioner,
)


def load_image_as_tensor(image_path: str) -> torch.Tensor:
    """
    Load an image from disk and convert to a (3, H, W) torch.FloatTensor in [0, 1].
    """
    img_pil = Image.open(image_path).convert("RGB")
    to_tensor = T.ToTensor()  # converts to float32 in [0,1], shape (C,H,W)
    img_tensor = to_tensor(img_pil)
    return img_tensor


def run_instructblip_on_image(
    image_path: str,
    device: str = "cuda",
    model_name: str = "Salesforce/instructblip-flan-t5-xl",
    prompt: str = "Describe this scene.",
    output_dir: str = "debug_outputs/instructblip_scene_captions",
    output_suffix: str = "",
) -> None:
    """
    - Loads an image from disk.
    - Runs InstructBLIPSceneLevelCaptioner to get a scene-level caption.
    - Prints the caption.
    - Saves the caption to a .txt file in `output_dir`.
    """
    # Make base output directory
    os.makedirs(output_dir, exist_ok=True)

    # Optional suffix subdirectory
    if output_suffix:
        output_dir = os.path.join(output_dir, output_suffix)
        os.makedirs(output_dir, exist_ok=True)

    # Subdirectory named after the image (without extension)
    image_basename = os.path.basename(image_path)
    image_name, _ = os.path.splitext(image_basename)
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # ---- load image as tensor ----
    img_tensor = load_image_as_tensor(image_path)  # (3, H, W), float32 in [0,1]

    # ---- set up captioner ----
    # device is a string, e.g. "cuda" or "cpu"
    captioner = InstructBLIPSceneLevelCaptioner(
        model_name=model_name,
        device=device,
        prompt=prompt,
    )

    # If you want the tensor on device (not strictly necessary since the processor
    # moves things internally, but it's fine to be explicit):
    img_tensor = img_tensor.to(device)

    # ---- generate caption ----
    caption, _debug = captioner.generateCaption(img_tensor, image_id="single_img")

    # Print to console
    print(f"Image: {image_path}")
    print("Generated caption:")
    print(caption)

    # ---- save caption to text file ----
    caption_txt_path = os.path.join(image_output_dir, "scene_caption.txt")
    with open(caption_txt_path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")

    print(f"Saved caption -> {caption_txt_path}")


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Run InstructBLIP scene-level captioning on a single image."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on (e.g. 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Salesforce/instructblip-flan-t5-xl",
        help="HF model name or path for InstructBLIP.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this scene.",
        help="Prompt passed to InstructBLIP.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug_outputs/instructblip_scene_captions",
        help="Directory to save caption text files.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Optional subdirectory suffix inside output_dir.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()

    # Example of overriding the default prompt with something more specific:
    # args.prompt = "Describe this scene in detail, including objects and their relationships."

    # args.prompt = "Describe the type of environment in the scene."
    # args.output_suffix = "environment_descriptions"

    # args.prompt = """
    # Describe this scene ONLY in terms of how stable objects are over time.

    # Answer with a single-line JSON object with keys:
    # - env_type
    # - indoor_outdoor
    # - time_of_day
    # - human_traffic_level
    # - small_obj_persistence
    # - large_obj_persistence
    # - fixed_struct_persistence

    # Use only these categories:
    # - time_of_day: ["night","morning","midday","afternoon","evening","unknown"]
    # - human_traffic_level: ["none","low","medium","high","very_high"]
    # - *_persistence: ["seconds","minutes","hours","days","weeks+","years+"]

    # No extra text, only JSON.
    # """


    args.prompt = """
    Describe this scene ONLY in terms of how stable objects are over time.

    Discuss the following aspects of the scene:
    - env_type
    - indoor_outdoor
    - time_of_day
    - human_traffic_level
    - small_obj_persistence
    - large_obj_persistence
    - fixed_struct_persistence

    Use these categories:
    - time_of_day: ["night","morning","midday","afternoon","evening","unknown"]
    - human_traffic_level: ["none","low","medium","high","very_high"]
    - *_persistence: ["seconds","minutes","hours","days","weeks+","years+"]
    """


    args.output_suffix = "persistence_focused_descriptions"


    run_instructblip_on_image(
        image_path=args.image_path,
        device=args.device,
        model_name=args.model_name,
        prompt=args.prompt,
        output_dir=args.output_dir,
        output_suffix=args.output_suffix,
    )
