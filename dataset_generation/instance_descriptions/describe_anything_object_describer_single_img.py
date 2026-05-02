# tools/test_dam_on_single_image.py

import os
import json
import argparse
from typing import Optional, Dict

import torch
import numpy as np
import cv2
from PIL import Image

from instance_segmentation.sam2_auto_segmenter import SAM2InstanceSegmenter
from instance_segmentation.abs_instance_segmenter import SegmentationResults
from dataset_generation.instance_descriptions.describe_anything_object_describer import DescribeAnythingObjectDescriber

from dam import DescribeAnythingModel, disable_torch_init


def overlay_single_mask(
    image_t: torch.Tensor,
    mask_t: torch.Tensor,
    alpha_mask: float = 0.4,
    alpha_img: float = 0.6,
    color_bgr=(0, 0, 255),
) -> np.ndarray:
    """
    Overlay a single mask on an image.

    image_t:
        (3, H, W) or (H, W, 3) torch.Tensor, values in [0,1] or [0,255]
    mask_t:
        (H, W) bool / 0-1 tensor

    Returns:
        (H, W, 3) uint8 image with the mask overlaid in the given color.
    """
    # ---- prep image ----
    if image_t.dim() == 3 and image_t.shape[0] in (1, 3):  # (C,H,W) -> (H,W,C)
        img_np = image_t.detach().cpu().permute(1, 2, 0).numpy()
    else:
        img_np = image_t.detach().cpu().numpy()

    # scale to 0–255 if needed
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)

    # ensure 3 channels
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif img_np.shape[2] == 1:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    H, W, _ = img_np.shape
    out = img_np.copy()

    # ---- prep mask ----
    mask_np = mask_t.detach().cpu().numpy()
    if mask_np.dtype != bool:
        mask_np = mask_np > 0.5
    assert mask_np.shape == (H, W), f"Mask has shape {mask_np.shape}, expected {(H, W)}"

    # colored mask
    colored_mask = np.zeros_like(out, dtype=np.uint8)
    colored_mask[mask_np] = np.array(color_bgr, dtype=np.uint8)

    # blend
    blended = cv2.addWeighted(colored_mask, alpha_mask, out, alpha_img, 0.0)
    out[mask_np] = blended[mask_np]

    return out


def run_dam_on_image(
    image_path: str,
    sam_checkpoint: str,
    sam_model_cfg: str,
    device: str,
    output_dir: str,
    dam_model_path: str = "nvidia/DAM-3B",
    dam_prompt: str = "<image>\nDescribe the masked region in detail.",
    dam_prompt_mode: str = "focal_prompt",
    dam_conv_mode: str = "v1",
    score_thresh: float = 0.0,
    output_suffix: str = "",
) -> None:
    """
    - Segments the image with SAM2InstanceSegmenter.
    - Uses DAM to describe each mask.
    - Saves one PNG per instance with mask overlay.
    - Saves descriptions.json mapping id -> description.
    """
    os.makedirs(output_dir, exist_ok=True)
    if (output_suffix is not None) and (output_suffix != ""):
        output_dir = os.path.join(output_dir, output_suffix)
        os.makedirs(output_dir, exist_ok=True)

    # Create a subdirectory that has the image name without extension
    image_basename = os.path.basename(image_path)
    image_name, _ = os.path.splitext(image_basename)
    output_dir = os.path.join(output_dir, image_name)
    os.makedirs(output_dir, exist_ok=True)

    # ---- load image ----
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)  # (H, W, 3), uint8
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W), uint8

    torch_device = torch.device(device)

    # ---- SAM2 segmenter ----
    segmenter = SAM2InstanceSegmenter(
        checkpoint=sam_checkpoint,
        model_cfg=sam_model_cfg,
        device=device,
    )
    seg, _seg_debug = segmenter.segmentInstances(img_tensor, image_id="single_img")

    # optional score thresholding
    if getattr(seg, "scores", None) is not None and score_thresh > 0:
        keep = (seg.scores >= score_thresh).nonzero(as_tuple=False).view(-1)
        if keep.numel() == 0:
            print("No instances above score threshold.")
            return
        seg.masks = seg.masks[keep]
        seg.boxes = seg.boxes[keep]
        seg.ids = seg.ids[keep] if getattr(seg, "ids", None) is not None else torch.arange(len(keep))
        seg.scores = seg.scores[keep]

    # ---- DAM setup ----
    disable_torch_init()

    # ---- DAM setup ----
    describer = DescribeAnythingObjectDescriber(
        model_path=dam_model_path,
        device=torch_device,
        prompt=dam_prompt,
        conv_mode=dam_conv_mode,
        prompt_mode=dam_prompt_mode,
        temperature=0.2,
        top_p=0.5,
        num_beams=1,
        max_new_tokens=512,
        streaming=False,  # simpler for this script
    )

    # ---- run DAM and save overlays ----
    descriptions, _debug = describer.describeObjects(img_tensor, seg, image_id="single_img")

    for i, mask_t in enumerate(seg.masks):
        obj_id = int(seg.ids[i].item()) if getattr(seg, "ids", None) is not None else i

        overlay_np = overlay_single_mask(
            img_tensor,
            mask_t,
            alpha_mask=0.4,
            alpha_img=0.6,
            color_bgr=(0, 0, 255),
        )
        overlay_pil = Image.fromarray(overlay_np)

        out_path = os.path.join(output_dir, f"mask_{obj_id}.png")
        overlay_pil.save(out_path)
        print(f"Saved overlay for id={obj_id} -> {out_path}")

    # ---- save JSON descriptions ----
    json_path = os.path.join(output_dir, "descriptions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)
    print(f"Saved descriptions JSON -> {json_path}")


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
        "--device",
        type=str,
        default="cuda",
        help="Device to run models on (e.g. 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="debug_outputs/dam_outputs",
        help="Directory to save overlay images and descriptions.json.",
    )
    # DAM-specific args
    parser.add_argument(
        "--dam_model_path",
        type=str,
        default="nvidia/DAM-3B",
        help="Path or HF id for the DAM checkpoint.",
    )
    parser.add_argument(
        "--dam_prompt",
        type=str,
        default="<image>\nDescribe the masked region in detail.",
        help="Prompt passed to DAM.",
    )
    parser.add_argument(
        "--dam_prompt_mode",
        type=str,
        default="focal_prompt",
        help="Describe Anything prompt_mode (e.g. 'focal_prompt').",
    )
    parser.add_argument(
        "--dam_conv_mode",
        type=str,
        default="v1",
        help="Describe Anything conv_mode.",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="Filter instances by score (if seg.scores is available).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()

    args.dam_prompt = "<image>\nDescribe the object in the highlighted " \
    "region and explicitly state: whether it is attached to any surface " \
    "or part of the environment; if free-standing, how easy it would be " \
    "for it to change position; whether external forces (wind, people, " \
    "vehicles) might move or deform it; any signs of imminent motion " \
    "(walking posture, wheels turned, being held by a person, etc.)."
    # args.dam_prompt = "<image>\nDescribe the object in the highlighted region. Comment on properties of the object that might affect its motion or persistence in the scene over time."
    suffix="detailed_persistence"
    run_dam_on_image(
        image_path=args.image_path,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_cfg=args.sam_model_cfg,
        device=args.device,
        output_dir=args.output_dir,
        dam_model_path=args.dam_model_path,
        dam_prompt=args.dam_prompt,
        dam_prompt_mode=args.dam_prompt_mode,
        dam_conv_mode=args.dam_conv_mode,
        score_thresh=args.score_thresh,
        output_suffix=suffix
    )
