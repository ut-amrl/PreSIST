"""Build the inverse-frequency weight table consumed by
``LabelBinnedFrequencyNormalizer``.

Iterates the training-split labels of a given dense Hydra config (post
``label_transform`` and, if the dataset config has y50_mean/std/yr_mean/std
set, post z-score normalization — matching the label space the rasterizer
sees inside ``_get_dense``), bins the chosen channel into B equal-width
bins, and writes a small npz with::

    bin_edges (B+1,) float — sorted bin edges spanning [min, max] of the data
    weights   (B,)   float — per-bin weight = (mean_count / count_b), so the
                             average weight across training samples is 1.0
                             (empty bins get weight 0).

Usage (from repo root, inside the container)::

    python scripts/compute_label_inverse_freq.py \
        --config-name=dpt_dense_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50 \
        --out-npz /workspace/data/output/dense_supervision_cache/label_invfreq_y50_post_norm.npz

The output path is what you point the dataset config's
``dense_rasterizer.normalizer.freq_table_path`` at.

Channel index defaults to 0 (y50). Pass ``--label-channel 1`` for yr.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        required=True,
        help="Hydra config name under configs/train_confs/ (without .yaml). "
             "Use the dense training config you intend to train with — the "
             "bin space matches whatever z-score state that config implies.",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Override the Hydra config directory.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to scan (default: train).",
    )
    parser.add_argument(
        "--label-channel",
        type=int,
        default=0,
        help="Which label channel to bin (0=y50, 1=yr, 2=yc). Default 0.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=50,
        help="Number of equal-width bins (default 50).",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=0.5,
        help="Clip the histogram domain to [pct, 100-pct] percentile of the "
             "data before binning, so a handful of outliers don't compress "
             "the useful range. Default 0.5 (i.e. bins span p0.5..p99.5).",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=20.0,
        help="Cap per-bin weight at this multiple of the mean. Prevents "
             "very-low-count bins from dominating the loss. Default 20.",
    )
    parser.add_argument(
        "--out-npz",
        required=True,
        help="Where to write the freq table .npz file.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_dir = Path(args.config_dir) if args.config_dir else (repo_root / "configs" / "train_confs")
    if not config_dir.exists():
        print(f"Config dir does not exist: {config_dir}", file=sys.stderr)
        return 2

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name)

    from dataset_generation.images_dataset.relpath_id_fn import RelpathToImageId  # type: ignore
    from dataset_generation.images_dataset.image_id_fn import ImageIdToRelpath  # type: ignore
    print("Instantiating train dataset config + registry helpers...")
    ds_cfg = hydra.utils.instantiate(cfg.train_dataset)
    registry_cfg = hydra.utils.instantiate(cfg.registry_info)
    relpath_to_image_id = RelpathToImageId(
        image_root=ds_cfg.input_root_dir,
        registry_cfg=registry_cfg,
    )
    image_id_to_relpath = ImageIdToRelpath(
        image_root=ds_cfg.input_root_dir,
        registry_cfg=registry_cfg,
        warn=print,
    )
    img_size = int(cfg.trainer.loop.img_size) if "img_size" in cfg.trainer.loop else 518

    # Build in per_instance mode — same label population the dense path
    # iterates, but without paying for mask resizing or image decoding.
    print(f"Building {args.split} dataset (per_instance, may take a minute)...")
    ds = ds_cfg.make_dataset(
        split=args.split,
        mode="per_instance",
        relpath_to_image_id=relpath_to_image_id,
        image_id_to_relpath=image_id_to_relpath,
        img_size=img_size,
    )

    transform = ds._label_transform  # type: ignore[attr-defined]
    if transform is None:
        print("Dataset has no _label_transform set; aborting.", file=sys.stderr)
        return 2

    # If the dataset has z-score stats set, apply them so the freq table
    # is in the same space the normalizer will see at training time
    # (the dense path z-scores instance_labels before they reach the
    # rasterizer / normalizer).
    z_active = (
        ds.cfg.y50_mean is not None  # type: ignore[attr-defined]
        and ds.cfg.y50_std is not None
        and ds.cfg.yr_mean is not None
        and ds.cfg.yr_std is not None
    )
    if z_active:
        ymean = [float(ds.cfg.y50_mean), float(ds.cfg.yr_mean), 0.0]  # type: ignore[attr-defined]
        ystd = [float(ds.cfg.y50_std), float(ds.cfg.yr_std), 1.0]
        print(f"Z-score normalization active: y50 N({ymean[0]:.4f}, {ystd[0]:.4f}), "
              f"yr  N({ymean[1]:.4f}, {ystd[1]:.4f})")
        print("Freq-table bins will be in z-score space.")
    else:
        ymean = [0.0, 0.0, 0.0]
        ystd = [1.0, 1.0, 1.0]
        print("No z-score normalization; freq-table bins will be in raw "
              "(post-label_transform) space.")

    instance_index = ds._instance_index  # type: ignore[attr-defined]
    records = ds._records  # type: ignore[attr-defined]
    n_total = len(instance_index)
    print(f"Total instances in {args.split}: {n_total:,}")

    values: List[float] = []
    n_missing = 0
    last_rec_idx = -1
    cached_obj = None
    for i, (rec_idx, inst_id) in enumerate(instance_index):
        if i % 50000 == 0 and i > 0:
            print(f"  processed {i:,} / {n_total:,}")
        rec = records[rec_idx]
        labels_path = rec["labels_path"]
        if rec_idx != last_rec_idx:
            try:
                cached_obj = ds._load_label_obj(labels_path)  # type: ignore[attr-defined]
            except Exception:
                n_missing += 1
                cached_obj = None
                last_rec_idx = rec_idx
                continue
            last_rec_idx = rec_idx

        raw = cached_obj.get(str(inst_id)) if cached_obj is not None else None
        if raw is None and cached_obj is not None:
            raw = cached_obj.get(int(inst_id))
        if raw is None:
            n_missing += 1
            continue

        try:
            t = transform(raw)
        except Exception:
            n_missing += 1
            continue
        if not torch.is_tensor(t) or t.numel() < 3:
            n_missing += 1
            continue
        v = float(t[args.label_channel])
        if not math.isfinite(v):
            n_missing += 1
            continue
        # Apply z-score if active so the binning matches training-time space.
        v = (v - ymean[args.label_channel]) / ystd[args.label_channel]
        values.append(v)

    if not values:
        print("No labels collected; aborting.", file=sys.stderr)
        return 1

    arr = np.asarray(values, dtype=np.float64)
    print(f"\nCollected {len(arr):,} values. min={arr.min():.4f} max={arr.max():.4f} "
          f"mean={arr.mean():.4f} std={arr.std():.4f}")

    lo = float(np.percentile(arr, args.clip_percentile))
    hi = float(np.percentile(arr, 100.0 - args.clip_percentile))
    if hi <= lo:
        print(f"Degenerate range [{lo}, {hi}]; aborting.", file=sys.stderr)
        return 1
    print(f"Binning over [{lo:.4f}, {hi:.4f}] "
          f"(p{args.clip_percentile}..p{100.0 - args.clip_percentile}) "
          f"into {args.num_bins} bins.")

    bin_edges = np.linspace(lo, hi, args.num_bins + 1)
    counts, _ = np.histogram(arr, bins=bin_edges)
    # Per-sample weights: 1 / count_of_my_bin. Empty bins -> weight 0.
    # Use the same bucketize convention as the normalizer: right=False.
    idx = np.searchsorted(bin_edges, arr, side="right") - 1
    idx = np.clip(idx, 0, args.num_bins - 1)
    sample_counts = counts[idx].astype(np.float64)
    inv = np.where(sample_counts > 0, 1.0 / sample_counts, 0.0)
    # Normalize so mean weight across samples == 1.
    mean_inv = float(inv.mean())
    if mean_inv <= 0:
        print("All bins empty; aborting.", file=sys.stderr)
        return 1
    inv_normed = inv / mean_inv
    # Per-bin weight (so the normalizer can do a single lookup):
    weights = np.zeros(args.num_bins, dtype=np.float64)
    for b in range(args.num_bins):
        if counts[b] > 0:
            weights[b] = (1.0 / float(counts[b])) / mean_inv
    # Cap.
    cap = float(args.max_weight)
    capped = np.minimum(weights, cap)
    n_capped = int((weights > cap).sum())
    if n_capped > 0:
        print(f"Capped {n_capped} bin weights at {cap:.2f}× mean.")

    out_path = Path(args.out_npz).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        bin_edges=bin_edges.astype(np.float32),
        weights=capped.astype(np.float32),
    )
    print(f"\nWrote freq table to {out_path}")
    print(f"  num bins              = {args.num_bins}")
    print(f"  domain                = [{lo:.4f}, {hi:.4f}]")
    print(f"  min/max per-bin weight = {capped.min():.4f} / {capped.max():.4f}")
    if n_missing:
        print(f"\nMissing/unparseable labels skipped: {n_missing:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
