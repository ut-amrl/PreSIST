"""One-off: compute (mean, std) of label channels 0 and 1 over the train
dataset, with normalization disabled, for the configured label_transform.

Used to fill in the y50_*/yr_* z-score placeholders in
__p25p75_linear.yaml / __p25p75_loggap.yaml. Run once per new transform.

Usage (from container, working dir = semantic-persistence-filter):
  python scripts/compute_target_stats.py \
      --config-name cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__p25p75_linear

Pass any normal Hydra overrides after `--`. The script:
  - composes the named config,
  - clears y50_mean/std and yr_mean/std (so the dataset returns raw labels),
  - iterates the train dataset (no augmentation, no model),
  - prints (mean, std, min, max) for channels 0 and 1.

Not committed-to / not part of the trained pipeline.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
import hydra.utils


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True,
                        help="Hydra config name under configs/train_confs/")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap on number of samples (default: full epoch)")
    parser.add_argument("--num-workers", type=int, default=4)
    args, hydra_overrides = parser.parse_known_args()

    with initialize(version_base=None, config_path="../configs/train_confs"):
        cfg = compose(config_name=args.config_name, overrides=hydra_overrides)

    # Disable z-score normalization so we observe raw transform outputs.
    for k in ("y50_mean", "y50_std", "yr_mean", "yr_std"):
        OmegaConf.update(cfg, f"train_dataset.{k}", None, merge=True)

    dataset = hydra.utils.instantiate(cfg.train_dataset)
    print(f"Train dataset size: {len(dataset)}", flush=True)

    n = 0
    s0 = 0.0; s1 = 0.0
    sq0 = 0.0; sq1 = 0.0
    min0 = math.inf; min1 = math.inf
    max0 = -math.inf; max1 = -math.inf

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, num_workers=args.num_workers,
        shuffle=False, collate_fn=lambda b: b,
    )
    cap = args.max_samples if args.max_samples is not None else len(dataset)
    for batch in loader:
        for item in batch:
            # The dataset yields a dict; the label tensor lives under one
            # of a few standard keys depending on the dataset variant.
            label = None
            for key in ("label", "y", "target", "labels"):
                if isinstance(item, dict) and key in item:
                    label = item[key]; break
            if label is None and isinstance(item, dict):
                # Fall back to (y50, yr, yc) tuple of scalars.
                if all(k in item for k in ("y50", "yr", "yc")):
                    label = torch.tensor([float(item["y50"]), float(item["yr"]),
                                          float(item["yc"])], dtype=torch.float32)
            if label is None:
                raise RuntimeError(f"Could not locate label in item keys: "
                                   f"{list(item.keys()) if isinstance(item, dict) else type(item)}")
            c0 = float(label[0]); c1 = float(label[1])
            s0 += c0; s1 += c1
            sq0 += c0 * c0; sq1 += c1 * c1
            if c0 < min0: min0 = c0
            if c0 > max0: max0 = c0
            if c1 < min1: min1 = c1
            if c1 > max1: max1 = c1
            n += 1
            if n >= cap:
                break
        if n >= cap:
            break
        if n % 5000 == 0:
            print(f"  n={n} (c0 running mean={s0/n:.4f})", flush=True)

    mean0 = s0 / n; mean1 = s1 / n
    var0 = max(sq0 / n - mean0 * mean0, 0.0)
    var1 = max(sq1 / n - mean1 * mean1, 0.0)
    std0 = math.sqrt(var0); std1 = math.sqrt(var1)

    print()
    print(f"n samples seen: {n}")
    print(f"channel 0:  mean={mean0:.6f}  std={std0:.6f}  min={min0:.6f}  max={max0:.6f}")
    print(f"channel 1:  mean={mean1:.6f}  std={std1:.6f}  min={min1:.6f}  max={max1:.6f}")
    print()
    print("Paste into the ablation config:")
    print(f"  y50_mean: {mean0:.4f}")
    print(f"  y50_std:  {std0:.4f}")
    print(f"  yr_mean:  {mean1:.4f}")
    print(f"  yr_std:   {std1:.4f}")


if __name__ == "__main__":
    main()
