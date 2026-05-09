"""
Inspect the source image + masked crop for inference artifacts whose
assigned closed-set class matches a filter (e.g., everything that wasn't
classified as "car"). Useful for diagnosing CLIP misclassifications
that produce frozen-positive predictions at long horizons.

Usage:
  python scripts/inspect_anchor_class_images.py \
      --artifacts /path/to/inference_artifacts.json \
      --manifest /path/to/annotation_manifest.json \
      --exclude-class car \
      --out-dir /tmp/anchor_inspect

  # Or include only specific classes:
  python scripts/inspect_anchor_class_images.py \
      --artifacts ... --manifest ... \
      --include-class banner chair "parking meter" \
      --out-dir /tmp/anchor_inspect
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

_US_TO_S = 1_000_000.0
_POSITIVE = "Positive"
_PADDING_FRAC = 0.15  # mirrors CLIPAligner._extract_masked_crop


def _remap(path_str: str, host_root: str | None, container_root: str | None) -> str:
    if host_root and container_root and path_str.startswith(host_root):
        return container_root + path_str[len(host_root):]
    return path_str


def load_manifest_index(
    manifest_path: Path,
    host_root: str | None,
    container_root: str | None,
) -> tuple[Path, dict[tuple[str, float], dict]]:
    """Return (location_root, {(track_id, anchor_time_s): observation_dict})."""
    manifest = json.loads(manifest_path.read_text())
    location_root = Path(_remap(manifest["location_root"], host_root, container_root))
    manifest_dir = manifest_path.parent

    index: dict[tuple[str, float], dict] = {}
    for rel in manifest["track_files"]:
        track = json.loads((manifest_dir / rel).read_text())
        tid = track["track_id"]
        for obs in track["observations"]:
            if obs.get("state") != _POSITIVE or not obs.get("mask_path"):
                continue
            anchor_time = obs["timestamp"] / _US_TO_S
            obs_remapped = {
                **obs,
                "mask_path": _remap(obs["mask_path"], host_root, container_root),
            }
            index[(tid, anchor_time)] = obs_remapped
    return location_root, index


def extract_padded_crop(image_np: np.ndarray, mask_np: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Mirror CLIPAligner._extract_masked_crop's padded bbox crop (without
    zeroing background — visualization is more informative with full pixels).
    Returns (crop_np_HWC, (r0, r1, c0, c1))."""
    H, W = mask_np.shape
    rows = np.flatnonzero(mask_np.any(axis=1))
    cols = np.flatnonzero(mask_np.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return image_np.copy(), (0, H - 1, 0, W - 1)
    r_min, r_max = int(rows[0]), int(rows[-1])
    c_min, c_max = int(cols[0]), int(cols[-1])
    pad_r = max(1, int(_PADDING_FRAC * (r_max - r_min + 1)))
    pad_c = max(1, int(_PADDING_FRAC * (c_max - c_min + 1)))
    r0, r1 = max(0, r_min - pad_r), min(H - 1, r_max + pad_r)
    c0, c1 = max(0, c_min - pad_c), min(W - 1, c_max + pad_c)
    return image_np[r0:r1 + 1, c0:c1 + 1].copy(), (r0, r1, c0, c1)


def render(
    artifact: dict,
    obs: dict,
    location_root: Path,
    out_path: Path,
) -> None:
    image_path = location_root / obs["image_path"]
    mask_path = Path(obs["mask_path"])
    if not mask_path.is_absolute():
        mask_path = location_root / mask_path

    image_np = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask_np = np.array(Image.open(mask_path).convert("L")) > 0

    crop_np, (r0, r1, c0, c1) = extract_padded_crop(image_np, mask_np)

    md = artifact["estimate"]["metadata"]
    cls = md.get("anchor_class_name", "?")
    cls_id = md.get("anchor_class_id", "?")
    params = artifact["estimate"]["params"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"track={artifact['track_id']}  anchor_time={artifact['anchor_time']}\n"
        f"assigned class: {cls} (id {cls_id})  params={params}",
        fontsize=10,
    )

    axes[0].imshow(image_np)
    overlay = np.zeros((*mask_np.shape, 4), dtype=np.float32)
    overlay[mask_np] = [0.0, 1.0, 0.4, 0.45]
    axes[0].imshow(overlay)
    rect = plt.Rectangle((c0, r0), c1 - c0 + 1, r1 - r0 + 1, fill=False, edgecolor="red", linewidth=1.5)
    axes[0].add_patch(rect)
    axes[0].set_title(f"Full image + mask + crop bbox\n{image_path.name}", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(crop_np)
    axes[1].set_title("Padded crop (CLIP input region)", fontsize=9)
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts", type=Path, required=True, help="inference_artifacts.json path.")
    p.add_argument("--manifest", type=Path, required=True, help="Annotation-tool manifest JSON path.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--exclude-class", nargs="+", default=None,
                   help="Render every artifact whose anchor_class_name is NOT in this list.")
    g.add_argument("--include-class", nargs="+", default=None,
                   help="Render only artifacts whose anchor_class_name IS in this list.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="Cap number of rendered artifacts.")
    p.add_argument("--host-root", default=None, help="If manifest paths used a different mount, source prefix.")
    p.add_argument("--container-root", default=None, help="Replacement prefix for --host-root.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    artifacts = json.loads(args.artifacts.read_text())
    print(f"loaded {len(artifacts)} artifacts from {args.artifacts}")

    if args.include_class is not None:
        keep = set(args.include_class)
        filtered = [a for a in artifacts if a["estimate"]["metadata"].get("anchor_class_name") in keep]
    elif args.exclude_class is not None:
        drop = set(args.exclude_class)
        filtered = [a for a in artifacts if a["estimate"]["metadata"].get("anchor_class_name") not in drop]
    else:
        filtered = artifacts
    print(f"after class filter: {len(filtered)} artifacts")

    if args.limit is not None:
        filtered = filtered[: args.limit]
        print(f"limited to first {len(filtered)}")

    location_root, index = load_manifest_index(args.manifest, args.host_root, args.container_root)
    print(f"manifest index has {len(index)} positive observations under {location_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_rendered = 0
    n_missing = 0
    for a in filtered:
        key = (a["track_id"], float(a["anchor_time"]))
        obs = index.get(key)
        if obs is None:
            print(f"  WARN: no manifest match for {key}")
            n_missing += 1
            continue
        cls = a["estimate"]["metadata"].get("anchor_class_name", "unknown")
        safe_cls = cls.replace(" ", "_").replace("/", "_")
        out = args.out_dir / f"{safe_cls}__{a['track_id']}__{a['anchor_time']}.png"
        render(a, obs, location_root, out)
        print(f"  wrote {out}")
        n_rendered += 1

    print(f"\nrendered {n_rendered}, missing {n_missing}, into {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
