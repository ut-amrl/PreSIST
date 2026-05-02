import cv2

# Create a main function that runs the sam2 auto segmenter on a single image
import sys
print(sys.path)

from instance_segmentation.sam2_auto_segmenter import SAM2InstanceSegmenter
from instance_segmentation.abs_instance_segmenter import SegmentationResults
import torch
from PIL import Image
import numpy as np

# Import optional
from typing import Optional


def overlay_segmentation(
    image_t: torch.Tensor,
    seg: SegmentationResults,
    alpha_mask: float = 0.4,
    alpha_img: float = 0.6,
    score_thresh: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Overlay all masks & boxes from a SegmentationResults onto an image.

    image_t:
        - (3, H, W) or (H, W, 3) torch.Tensor
        - values in [0,1] or [0,255]
    seg:
        SegmentationResults for this image.

    Returns:
        out: (H, W, 3) uint8 image with colored masks and boxes.
    """
    # ---- prep image ----
    if image_t.dim() == 3 and image_t.shape[0] in (1, 3):  # (C,H,W) -> (H,W,C)
        img_np = image_t.detach().cpu().permute(1, 2, 0).numpy()
    else:  # assume already (H,W,C)
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

    N = len(seg)
    if N == 0:
        return out

    # ---- filter by score if available ----
    if seg.scores is not None:
        keep = (seg.scores >= score_thresh).nonzero(as_tuple=False).view(-1)
        if keep.numel() == 0:
            return out
        masks = seg.masks[keep]
        boxes = seg.boxes[keep]
        scores = seg.scores[keep]
    else:
        masks = seg.masks
        boxes = seg.boxes
        scores = None

    N_keep = masks.shape[0]

    # ---- random colors ----
    if seed is not None:
        np.random.seed(seed)
    colors = [
        tuple(np.random.randint(0, 256, size=3).tolist())  # BGR
        for _ in range(N_keep)
    ]

    # ---- overlay each instance ----
    for i in range(N_keep):
        mask_t = masks[i]              # (H, W) bool
        box_t = boxes[i]               # (4,)
        color = colors[i]

        mask_np = mask_t.detach().cpu().numpy().astype(bool)
        assert mask_np.shape == (H, W), f"Mask {i} has shape {mask_np.shape}, expected {(H, W)}"

        # colored mask of this instance
        colored_mask = np.zeros_like(out, dtype=np.uint8)
        colored_mask[mask_np] = np.array(color, dtype=np.uint8)

        # blend with current out (so instances stack)
        blended = cv2.addWeighted(colored_mask, alpha_mask, out, alpha_img, 0.0)
        out[mask_np] = blended[mask_np]

        # draw bounding box
        x1, y1, x2, y2 = box_t.tolist()
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)

    return out


def main(image_path: str, checkpoint: str, model_cfg: str, device: str, output_overlay_file: str):
    # Load the iange as a pytorch tensor
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)  # (H, W, 3), uint8
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W), uint8

    # Create the segmenter
    segmenter = SAM2InstanceSegmenter(
        checkpoint=checkpoint,
        model_cfg=model_cfg,
        device=device,
    )

    # Segment instances
    results, _debug = segmenter.segmentInstances(img_tensor, image_id="single_img")

    # Overlay segmentation results
    overlay_img = overlay_segmentation(
        img_tensor,
        results,
        alpha_mask=0.4,
        alpha_img=0.6,
        score_thresh=0.0,
        seed=42,
    )

    # Save the overlay image
    overlay_pil = Image.fromarray(overlay_img)
    overlay_pil.save(output_overlay_file)

    # # Copy the image and overlay masks, highlighted by bounding boxes
    # overlay_img = img_tensor.clone().cpu().numpy().permute(1, 2, 0)  # (H, W, 3), uint8
    # num_masks = len(results)
    # for i in range(num_masks):
    #     mask = results.masks[i]  # (H, W), bool
    #     box = results.boxes[i]   # (4,), XYXY

    #     print("Overlay img shape: ", overlay_img.shape)

    #     color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)

    #     overlay_img[mask] = (0.4 * overlay_img[mask] + 0.6 * color).astype(np.uint8)

    #     # # Create a random color for the mask
    #     # color = torch.randint(0, 256, (3,), dtype=torch.uint8)

    #     # # Overlay the mask
    #     # overlay_img[:, mask] = (0.5 * overlay_img[:, mask].float() + 0.5 * color[:, None].float()).to(torch.uint8)

    #     # # Use cv2.rectangle to draw the bounding box

    #     # x1, y1, x2, y2 = map(int, box.tolist())
    #     # cv2.rectangle(overlay_img.permute(1, 2, 0).cpu().numpy(), (x1, y1), (x2, y2), color.cpu().numpy().tolist(), 2)

    # # Save the overlay image
    # overlay_pil = Image.fromarray(overlay_img.permute(1, 2, 0).cpu().numpy())
    # overlay_pil.save(output_overlay_file)

def argParse():
    import argparse
    parser = argparse.ArgumentParser(description="Run SAM2 Auto Segmenter on a single image.")
    parser.add_argument("--image_path", type=str, help="Path to the input image.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the SAM2 model checkpoint.")
    parser.add_argument("--model_cfg", type=str, required=True, help="Path to the SAM2 model config file.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--output_overlay_file", type=str, default="segmentation_overlay.png", help="Path to save the output overlay image.")
    return parser.parse_args()


if __name__ == "__main__":
    args = argParse()
    # Load image
    main(args.image_path, args.checkpoint, args.model_cfg, args.device, args.output_overlay_file)

