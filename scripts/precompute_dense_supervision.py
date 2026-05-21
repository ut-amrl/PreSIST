"""Warm the dense-supervision cache before training.

Iterates the dataset in ``dense`` mode once and writes the rasterized
``(target_map, loss_mask, pixel_weights)`` for every image to disk.
After this script completes, training can re-read cached tensors from
disk instead of recomputing the rasterization on every epoch.

Usage:

    PYTHONPATH=/workspace/code/semantic-persistence-filter \\
    CUDA_VISIBLE_DEVICES= \\
    python3 scripts/precompute_dense_supervision.py \\
        --config-name dense_seg_langSam3Q8T_behav4_mO20_imgbal \\
        +precompute_splits='[train,val]'

Notes:
  - Set ``CUDA_VISIBLE_DEVICES=`` to keep this script off the GPU; it's
    purely CPU work.
  - Re-runnable: the cache no-ops for entries that already exist on disk.
  - Cache root must be set in the train_dataset config
    (``dense_cache_dir``); the script will fail loudly otherwise.
"""
from __future__ import annotations

import os
from typing import List

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_generation.images_dataset.image_id_fn import ImageIdToRelpath
from dataset_generation.images_dataset.registry import RegistryConfig
from dataset_generation.images_dataset.relpath_id_fn import RelpathToImageId
from training.dataset.distillation_dataset import DistillationDatasetConfig


def _warm_split(
    dataset_cfg: DistillationDatasetConfig,
    registry_cfg: RegistryConfig,
    split: str,
    img_size: int,
    num_workers: int,
) -> None:
    relpath_to_id = RelpathToImageId(
        image_root=dataset_cfg.input_root_dir,
        registry_cfg=registry_cfg,
    )
    id_to_relpath = ImageIdToRelpath(
        image_root=dataset_cfg.input_root_dir,
        registry_cfg=registry_cfg,
        warn=print,
    )

    print(f"[precompute] building {split} dataset in dense mode...")
    ds = dataset_cfg.make_dataset(
        split=split,
        mode="dense",
        relpath_to_image_id=relpath_to_id,
        image_id_to_relpath=id_to_relpath,
        img_size=img_size,
    )

    if getattr(ds, "_dense_cache", None) is None or not ds._dense_cache.enable:  # type: ignore[attr-defined]
        raise RuntimeError(
            "Dense cache is disabled (or dense_cache_dir is unset). "
            "Set train_dataset.dense_cache_dir and dense_cache_enabled=true "
            "in the config to use the cache."
        )

    # Drive the dataset through a DataLoader so multiple workers can
    # rasterize in parallel. We don't actually consume the batches — we
    # just want __getitem__ to fire on every index.
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: None,  # discard
        pin_memory=False,
    )
    print(f"[precompute] {split}: warming cache for {len(ds)} images using "
          f"{num_workers} workers...")
    for _ in tqdm(loader, desc=f"[precompute:{split}]", unit="img"):
        pass
    print(f"[precompute] {split}: done.")


@hydra.main(
    version_base=None,
    config_path="../configs/train_confs",
    config_name="config.yaml",
)
def main(cfg: DictConfig) -> None:
    splits: List[str] = list(cfg.get("precompute_splits", ["train", "val"]))
    workers = int(cfg.get("precompute_num_workers", 4))

    registry_cfg = hydra.utils.instantiate(cfg.registry_info)
    dataset_cfg: DistillationDatasetConfig = hydra.utils.instantiate(
        cfg.train_dataset
    )
    img_size = int(cfg.trainer.loop.img_size) if "img_size" in cfg.trainer.loop else 518

    for split in splits:
        _warm_split(dataset_cfg, registry_cfg, split, img_size, workers)


if __name__ == "__main__":
    main()
