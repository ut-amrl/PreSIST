"""One-off measurement: distribution of instances-per-image in train/val splits.

Determines whether the dense-prediction mask representation should be padded
(B, N_max, H, W) — fine for low variance — or flat/concatenated
(Σ N_i, H, W) + batch_index — better for long-tailed counts.

Run with the same --config-name that you'd pass to train.py, e.g.:
    python3 scripts/inspect_instance_counts.py \
        --config-name cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal
"""
from collections import Counter

import hydra
import numpy as np
from omegaconf import DictConfig

from dataset_generation.images_dataset.image_id_fn import ImageIdToRelpath
from dataset_generation.images_dataset.registry import RegistryConfig
from dataset_generation.images_dataset.relpath_id_fn import RelpathToImageId
from training.dataset.distillation_dataset import DistillationDatasetConfig


def _counts_per_image(dataset) -> np.ndarray:
    per_image = Counter(rec_idx for rec_idx, _ in dataset._instance_index)
    return np.asarray(sorted(per_image.values()), dtype=np.int64)


def _report(name: str, counts: np.ndarray) -> None:
    if counts.size == 0:
        print(f"\n=== {name}: empty split ===")
        return
    pcts = [50, 75, 90, 95, 99, 99.9]
    percentiles = np.percentile(counts, pcts)
    print(f"\n=== {name} ===")
    print(f"  images:           {counts.size}")
    print(f"  total instances:  {int(counts.sum())}")
    print(f"  min / max:        {int(counts.min())} / {int(counts.max())}")
    print(f"  mean / median:    {counts.mean():.2f} / {int(np.median(counts))}")
    print(f"  std:              {counts.std():.2f}")
    for p, v in zip(pcts, percentiles):
        print(f"  p{p:>5}:           {v:.1f}")
    ratio = percentiles[4] / max(np.median(counts), 1)  # p99 / median
    print(f"  p99 / median:     {ratio:.2f}x  (>5x => switch to flat/concat)")
    bins = [1, 2, 3, 5, 10, 20, 50, 100, 500, 10_000]
    hist, edges = np.histogram(counts, bins=bins)
    print("  histogram (counts per bin):")
    for lo, hi, h in zip(edges[:-1], edges[1:], hist):
        bar = "#" * min(60, int(60 * h / max(hist.max(), 1)))
        print(f"    [{lo:>4}, {hi:>5}): {h:>7d}  {bar}")


@hydra.main(version_base=None, config_path="../configs/train_confs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print(f"[inspect_instance_counts] train_dataset.name = {cfg.train_dataset.name}")

    registry_cfg: RegistryConfig = hydra.utils.instantiate(cfg.registry_info)
    dataset_cfg: DistillationDatasetConfig = hydra.utils.instantiate(cfg.train_dataset)

    relpath_to_id = RelpathToImageId(
        image_root=dataset_cfg.input_root_dir,
        registry_cfg=registry_cfg,
    )
    id_to_relpath = ImageIdToRelpath(
        image_root=dataset_cfg.input_root_dir,
        registry_cfg=registry_cfg,
        warn=print,
    )

    img_size = cfg.trainer.loop.img_size if "img_size" in cfg.trainer.loop else 518

    for split in ("train", "val"):
        print(f"\n[inspect_instance_counts] building {split} dataset (per_instance mode)...")
        ds = dataset_cfg.make_dataset(
            split=split,
            mode="per_instance",
            relpath_to_image_id=relpath_to_id,
            image_id_to_relpath=id_to_relpath,
            img_size=img_size,
        )
        counts = _counts_per_image(ds)
        _report(f"{split} split", counts)


if __name__ == "__main__":
    main()
