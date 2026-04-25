"""
One-time script to compute and cache CLIP text embeddings for a closed set of semantic classes.

Usage:
    python -m persistence_eval.methods.inferencers.closed_set_clip_align.class_clipalignment_cacher \
        --class_list_file path/to/classes.txt \
        --output_file path/to/clip_cache.pt \
        [--model_name openai/clip-vit-large-patch14] \
        [--device cuda]
"""

import argparse

from persistence_eval.methods.inferencers.closed_set_semantics.closed_set_semantics import ClosedSetSemanticClassList
from persistence_eval.methods.inferencers.closed_set_clip_align.closed_set_clip_embedder import (
    CachedCLIPAlignment,
    CLIPEmbedder,
)


def build_and_save_cache(
    class_list_file: str,
    output_file: str,
    model_name: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
) -> None:
    class_list = ClosedSetSemanticClassList(class_text_file=class_list_file)
    classes = class_list.semantic_classes
    print(f"Computing CLIP embeddings for {len(classes)} classes...")

    embedder = CLIPEmbedder(model_name=model_name, device=device)
    embeddings = embedder.compute_clip_embeddings(classes)  # (N, D)

    cache = CachedCLIPAlignment()
    for class_name, embedding in zip(classes, embeddings):
        cache.mapping[class_name] = embedding

    cache.save_cache(output_file)
    print(f"Saved CLIP alignment cache to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute CLIP text embeddings for a class list.")
    parser.add_argument("--class_list_file", required=True, help="Text file with one class name per line.")
    parser.add_argument("--output_file", required=True, help="Output .pt file for the cached alignment.")
    parser.add_argument("--model_name", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    build_and_save_cache(
        class_list_file=args.class_list_file,
        output_file=args.output_file,
        model_name=args.model_name,
        device=args.device,
    )
