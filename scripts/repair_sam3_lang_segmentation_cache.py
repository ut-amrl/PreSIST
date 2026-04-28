"""
In-place repair of cached SAM3-language segmentation .npz files.

Background
----------
For square images (H == W), the SAM3 batch language-prompted segmenter
([instance_segmentation/language_prompted_segmentation/sam3_batch_language_prompted_segmenter.py:402-403])
contains a defensive transpose `if m.shape == (W, H): m = m.T` that
unintentionally fires whenever H == W (since (W, H) == (H, W) for squares),
producing a transposed mask. SAM3's boxes are returned in image coords and
are NOT transposed. So the only thing wrong in the cache for square images
is the masks; everything else (boxes, scores, ids, imagePath) is correct.

What this script does
---------------------
For each <image_id>/ directory under SEG_ROOT:
  - segmentation_results.npz is loaded.
  - If the masks are square (H == W), it transposes them, renames the
    original to pre_rot_segmentation_results.npz, and writes the
    corrected version as segmentation_results.npz.
  - If the masks are non-square, it leaves the directory untouched.

Idempotent: if pre_rot_segmentation_results.npz already exists, the script
refuses to clobber it and skips that directory (logs ALREADY_BACKED_UP_SKIP).

Usage
-----
Repair a small subset (5 square + 5 non-square) and write verification viz:
  python -m scripts.repair_sam3_lang_segmentation_cache --subset 5

Repair the entire SEG_ROOT (no viz):
  python -m scripts.repair_sam3_lang_segmentation_cache --all

Dry-run a subset (no writes; viz uses the unrepaired current cache):
  python -m scripts.repair_sam3_lang_segmentation_cache --subset 5 --dry-run

Pass --seg-root to point at a different cache. Default is the production
langGround_p_qwenvl37bT_disc... cache mounted in the dev container.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Allow running as a plain script in addition to `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from instance_segmentation.abs_instance_segmenter import SegmentationResults  # noqa: E402
from utils.visualization_utils import overlay_single_mask  # noqa: E402


DEFAULT_SEG_ROOT = "/workspace/data/output/segmentation_results/langGround_p_qwenvl37bT_disc_maxObj30_u3_s_sam3b32__s0p35_mA0p001_dedupe0p9"
DEFAULT_VIZ_ROOT = "/workspace/data/output/debug_viz/inplace_repair_verify"

CURRENT_NAME = "segmentation_results.npz"
BACKUP_NAME = "pre_rot_segmentation_results.npz"

COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 128, 255), (128, 0, 255), (255, 128, 0), (128, 255, 128),
]


def overlay_topk_with_boxes(image_rgb_np, masks, boxes, k=10, alpha_mask=0.5, alpha_img=0.5):
    if len(masks) == 0:
        return image_rgb_np.copy()
    areas = masks.reshape(masks.shape[0], -1).sum(dim=1)
    order = torch.argsort(areas, descending=True)[:k].tolist()
    img_t = torch.from_numpy(image_rgb_np).permute(2, 0, 1)
    out = image_rgb_np.copy()
    for i, mi in enumerate(order):
        c = COLORS[i % len(COLORS)]
        ov = overlay_single_mask(
            img_t, masks[mi], alpha_mask=alpha_mask, alpha_img=alpha_img, color_bgr=c,
        )
        m = masks[mi].numpy().astype(bool)
        out[m] = ov[m]
        x0, y0, x1, y1 = [int(round(float(v))) for v in boxes[mi].tolist()]
        cv2.rectangle(out, (x0, y0), (x1, y1), color=c, thickness=2)
    return out


def repair_one(img_dir: str, dry_run: bool) -> dict:
    cur_path = os.path.join(img_dir, CURRENT_NAME)
    bak_path = os.path.join(img_dir, BACKUP_NAME)
    info: dict = {"img_dir": img_dir, "status": "", "masks_shape": None, "image_path": None}
    if not os.path.exists(cur_path):
        info["status"] = "NO_CACHE"
        return info

    seg, image_path = SegmentationResults.loadFromFile(cur_path)
    info["image_path"] = image_path
    masks = seg.masks
    if masks.ndim != 3 or masks.shape[0] == 0:
        info["status"] = "EMPTY"
        info["masks_shape"] = tuple(masks.shape)
        return info

    H, W = int(masks.shape[1]), int(masks.shape[2])
    info["masks_shape"] = (H, W)

    if H != W:
        info["status"] = "NO_REPAIR_NEEDED"
        return info

    if os.path.exists(bak_path):
        info["status"] = "ALREADY_BACKED_UP_SKIP"
        return info

    repaired = SegmentationResults(
        ids=seg.ids,
        masks=masks.permute(0, 2, 1).contiguous(),
        boxes=seg.boxes,  # SAM3-native, already in image coords
        scores=seg.scores,
    )

    if dry_run:
        info["status"] = "WOULD_REPAIR"
        return info

    shutil.move(cur_path, bak_path)
    try:
        repaired.saveToFile(cur_path, imagePath=image_path)
    except Exception as e:
        # Roll back the rename if the write fails so we never lose the cache.
        shutil.move(bak_path, cur_path)
        info["status"] = f"FAILED_WRITE:{e!r}"
        return info
    info["status"] = "REPAIRED"
    return info


def render_viz(img_dir: str, image_path: str, viz_label: str, viz_subdir: str, viz_root: str) -> str:
    cur_path = os.path.join(img_dir, CURRENT_NAME)
    bak_path = os.path.join(img_dir, BACKUP_NAME)
    if not os.path.exists(image_path):
        return f"  [viz] image missing: {image_path}"
    with Image.open(image_path) as im:
        img_np = np.array(im.convert("RGB"))
    out_dir = os.path.join(viz_root, viz_subdir)
    os.makedirs(out_dir, exist_ok=True)
    slug = Path(image_path).stem

    seg_cur, _ = SegmentationResults.loadFromFile(cur_path)
    if seg_cur.masks.ndim == 3 and seg_cur.masks.shape[0] > 0:
        ov = overlay_topk_with_boxes(img_np, seg_cur.masks.bool(), seg_cur.boxes, k=10)
        Image.fromarray(ov).save(os.path.join(out_dir, f"{viz_label}__{slug}__current.png"))
    if os.path.exists(bak_path):
        seg_bak, _ = SegmentationResults.loadFromFile(bak_path)
        if seg_bak.masks.ndim == 3 and seg_bak.masks.shape[0] > 0:
            ov_b = overlay_topk_with_boxes(img_np, seg_bak.masks.bool(), seg_bak.boxes, k=10)
            Image.fromarray(ov_b).save(os.path.join(out_dir, f"{viz_label}__{slug}__pre_rot.png"))
    Image.fromarray(img_np).save(os.path.join(out_dir, f"{viz_label}__{slug}__orig.png"))
    suffix = "current+pre_rot" if os.path.exists(bak_path) else "current only"
    return f"  [viz] wrote {viz_subdir}/{viz_label}__{slug} ({suffix})"


def pick_subset(seg_root: str, n_per_kind: int, seed: int = 0) -> tuple[list[str], list[str]]:
    """Pick `n_per_kind` square + `n_per_kind` non-square cache directories."""
    random.seed(seed)
    candidates = sorted(glob.glob(os.path.join(seg_root, "*", CURRENT_NAME)))
    random.shuffle(candidates)
    sq, ns = [], []
    for p in candidates:
        if len(sq) >= n_per_kind and len(ns) >= n_per_kind:
            break
        try:
            with np.load(p, allow_pickle=True) as d:
                m = d["masks"]
                if m.ndim != 3 or m.shape[0] == 0:
                    continue
                shape = (int(m.shape[1]), int(m.shape[2]))
        except Exception:
            continue
        d_dir = os.path.dirname(p)
        if shape[0] == shape[1] and len(sq) < n_per_kind:
            sq.append(d_dir)
        elif shape[0] != shape[1] and len(ns) < n_per_kind:
            ns.append(d_dir)
    return sq, ns


def list_all_dirs(seg_root: str) -> list[str]:
    return sorted(
        os.path.dirname(p) for p in glob.glob(os.path.join(seg_root, "*", CURRENT_NAME))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-root", default=DEFAULT_SEG_ROOT)
    ap.add_argument("--viz-root", default=DEFAULT_VIZ_ROOT)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--subset", type=int, help="Sample N square + N non-square dirs and render viz.")
    g.add_argument("--all", action="store_true", help="Repair every dir under --seg-root (no viz).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"SEG_ROOT = {args.seg_root}")
    print(f"VIZ_ROOT = {args.viz_root}")
    print(f"DRY_RUN  = {args.dry_run}\n")

    if args.subset is not None:
        sq, ns = pick_subset(args.seg_root, args.subset)
        print(f"Picked {len(sq)} square + {len(ns)} non-square directories.\n")
        os.makedirs(args.viz_root, exist_ok=True)
        print("--- SQUARE (will be repaired) ---")
        for d in sq:
            info = repair_one(d, args.dry_run)
            print(f"  {os.path.basename(d)[:12]}.. masks_shape={info['masks_shape']} status={info['status']}")
            if info["image_path"]:
                print(render_viz(d, info["image_path"], "SQ", "square", args.viz_root))
        print("\n--- NON-SQUARE (no repair) ---")
        for d in ns:
            info = repair_one(d, args.dry_run)
            print(f"  {os.path.basename(d)[:12]}.. masks_shape={info['masks_shape']} status={info['status']}")
            if info["image_path"]:
                print(render_viz(d, info["image_path"], "NS", "nonsquare", args.viz_root))
    else:
        all_dirs = list_all_dirs(args.seg_root)
        print(f"Walking {len(all_dirs)} cache directories ...")
        counts: dict[str, int] = {}
        for d in all_dirs:
            info = repair_one(d, args.dry_run)
            counts[info["status"]] = counts.get(info["status"], 0) + 1
        print("\nSummary:")
        for k, v in sorted(counts.items()):
            print(f"  {k:32s} {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
