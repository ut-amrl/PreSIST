"""
Debug script: show the top-5 CLIP-matching classes for each mask in an image.

Input modes (at least one of --image or --seg_results_file is required):
  --seg_results_file only          → image path read from embedded imagePath, all masks
  --seg_results_file --mask_index  → image path read from embedded imagePath, one mask
  --image --seg_results_file       → explicit image, all masks from seg file
  --image --seg_results_file --mask_index → explicit image, one mask from seg file
  --image only                     → explicit image, whole-image all-true mask

Usage example:
  python scripts/debug_clip_alignment.py \
      --seg_results_file path/to/segmentation_results.npz \
      --cache_file cached_closed_set_info/closed_set_clip_embeddings/coco-stuff.pt \
      --class_list_file cached_closed_set_info/closed_set_lists/coco-stuff.txt
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.io as tvio

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from instance_segmentation.abs_instance_segmenter import SegmentationResults
from persistence_eval.methods.inferencers.closed_set_clip_align.closed_set_clip_embedder import (
    CachedCLIPAlignment,
    CLIPEmbedder,
    CLIPAligner,
)
from persistence_eval.methods.inferencers.closed_set_semantics.closed_set_semantics import ClosedSetSemanticClassList


def _load_image(image_path: str) -> torch.Tensor:
    img = tvio.read_image(image_path)  # CHW uint8
    if img.shape[0] == 4:
        img = img[:3]
    return img


def _top5(aligner: CLIPAligner, image: torch.Tensor, mask: torch.Tensor) -> tuple[list[tuple[str, float]], object]:
    crop_pil = aligner._extract_masked_crop(image, mask)
    image_embedding = aligner.clip_embedder.compute_clip_embedding_for_image(crop_pil)

    class_names = aligner.cache.class_names()
    text_embeddings = torch.stack([aligner.cache.get_embedding(c) for c in class_names])
    similarities = (text_embeddings @ image_embedding).tolist()

    ranked = sorted(zip(class_names, similarities), key=lambda x: x[1], reverse=True)
    return ranked[:5], crop_pil


def _show_mask(ax: plt.Axes, image_np: np.ndarray, mask_np: np.ndarray, title: str) -> None:
    ax.imshow(image_np)
    overlay = np.zeros((*mask_np.shape, 4), dtype=np.float32)
    overlay[mask_np, :] = [0.0, 1.0, 0.4, 0.45]
    ax.imshow(overlay)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _show_crop(ax: plt.Axes, crop_pil) -> None:
    ax.imshow(np.array(crop_pil))
    ax.set_title("CLIP crop", fontsize=9)
    ax.axis("off")


def _show_scores(ax: plt.Axes, top5: list[tuple[str, float]], class_list: ClosedSetSemanticClassList) -> None:
    names = [f"{name} (id {class_list.class_to_id.get(name, '?')})" for name, _ in top5]
    scores = [s for _, s in top5]
    colors = ["#2196F3" if i == 0 else "#90CAF9" for i in range(len(scores))]
    bars = ax.barh(range(len(names)), scores, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Cosine similarity", fontsize=8)
    ax.set_title("Top-5 matches", fontsize=9)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=8)
    ax.set_xlim(0, max(scores) * 1.18)
    ax.tick_params(axis="x", labelsize=8)


def main():
    parser = argparse.ArgumentParser(description="Show top-5 CLIP class matches for masks.")
    parser.add_argument("--image", default=None, help="Path to the image file.")
    parser.add_argument("--seg_results_file", default=None, help="Path to a SegmentationResults .npz file.")
    parser.add_argument("--mask_index", type=int, default=None, help="Run only on this mask index (0-based).")
    parser.add_argument("--cache_file", required=True, help="Path to the CLIP text embeddings cache (.pt).")
    parser.add_argument("--class_list_file", required=True, help="Path to the class list text file.")
    parser.add_argument("--device", default="cuda", help="Device for CLIP model (default: cuda).")
    parser.add_argument("--zero_out_background", action="store_true", help="Zero out non-mask pixels in the CLIP crop.")
    parser.add_argument("--debug_dir", default="debug_clip_alignment", help="Directory to save output figures.")
    args = parser.parse_args()

    if args.image is None and args.seg_results_file is None:
        parser.error("At least one of --image or --seg_results_file must be provided.")

    seg_results = None
    embedded_image_path = None
    if args.seg_results_file is not None:
        seg_results, embedded_image_path = SegmentationResults.loadFromFile(args.seg_results_file)

    image_path = args.image or embedded_image_path
    if image_path is None:
        parser.error("No image path available: provide --image or use a seg_results_file with an embedded imagePath.")

    image = _load_image(image_path)
    H, W = image.shape[1], image.shape[2]
    image_np = image.permute(1, 2, 0).numpy()

    if seg_results is not None:
        masks = seg_results.masks  # (N, H, W) bool
        boxes = seg_results.boxes  # (N, 4) float32 XYXY
        if args.mask_index is not None:
            indices = [args.mask_index]
        else:
            indices = list(range(len(seg_results)))
    else:
        masks = torch.ones((1, H, W), dtype=torch.bool)
        boxes = torch.tensor([[0.0, 0.0, float(W), float(H)]])
        indices = [0]

    cache = CachedCLIPAlignment(cache_file=args.cache_file)
    embedder = CLIPEmbedder(device=args.device)
    class_list = ClosedSetSemanticClassList(class_text_file=args.class_list_file)
    aligner = CLIPAligner(clip_embedder=embedder, cache=cache, class_list=class_list,
                          zero_out_background=args.zero_out_background)

    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    total = len(indices)
    print(f"Image : {image_path}  ({W}x{H})")
    print(f"Masks : {total} of {masks.shape[0]}")
    print(f"Output: {debug_dir.resolve()}")
    print()

    for step, idx in enumerate(indices, 1):
        top5, crop_pil = _top5(aligner, image, masks[idx])
        best_name = top5[0][0]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(
            f"Mask {idx} of {masks.shape[0] - 1}  ({step}/{total})  —  best match: {best_name}",
            fontsize=11,
        )

        _show_mask(axes[0], image_np, masks[idx].numpy(), "Image + mask")
        _show_crop(axes[1], crop_pil)
        _show_scores(axes[2], top5, class_list)

        plt.tight_layout()
        out_path = debug_dir / f"mask_{idx:04d}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{step}/{total}] mask {idx:>4}  →  {best_name}  ({out_path})")

    print(f"\nDone. Image: {image_path}")


if __name__ == "__main__":
    main()
