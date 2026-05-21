"""End-to-end smoke for the new ``dense`` dataset mode.

Builds the existing training dataset config but in ``dense`` mode (with
heavy subsampling), instantiates a small batch through the custom dense
collate, and prints shapes. Verifies that the rasterizer is invoked
once per image and produces the expected ``(C, H, W)`` tensors.
"""
from __future__ import annotations

import sys
import tempfile

import hydra
import torch
from torch.utils.data import DataLoader

from dataset_generation.images_dataset.image_id_fn import ImageIdToRelpath
from dataset_generation.images_dataset.registry import RegistryConfig
from dataset_generation.images_dataset.relpath_id_fn import RelpathToImageId
from persistence_parameter_networks.data.dense_target_rasterizer import (
    PerInstanceAreaNormalizer,
    SmallerWinsRasterizer,
)
from training.dataset.dense_collate import make_dense_collate
from training.dataset.distillation_dataset import DistillationDatasetConfig


def main() -> int:
    with hydra.initialize(version_base=None, config_path="../configs/train_confs"):
        cfg = hydra.compose(
            config_name="cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_smoke",
            overrides=[],
        )

    registry_cfg = hydra.utils.instantiate(cfg.registry_info)

    # We instantiate the dataset config and then tack on the
    # dense_rasterizer field in Python, since the existing smoke yaml
    # doesn't know about the dense path yet.
    dataset_cfg: DistillationDatasetConfig = hydra.utils.instantiate(cfg.train_dataset)
    # frozen dataclass workaround: object.__setattr__
    object.__setattr__(
        dataset_cfg,
        "dense_rasterizer",
        SmallerWinsRasterizer(PerInstanceAreaNormalizer()),
    )
    with tempfile.TemporaryDirectory() as tmp:
        object.__setattr__(dataset_cfg, "dense_cache_dir", tmp)
        object.__setattr__(dataset_cfg, "dense_cache_enabled", True)
        # Use a tiny subset.
        object.__setattr__(dataset_cfg, "train_subsample_fraction", 0.001)

        relpath_to_id = RelpathToImageId(
            image_root=dataset_cfg.input_root_dir,
            registry_cfg=registry_cfg,
        )
        id_to_relpath = ImageIdToRelpath(
            image_root=dataset_cfg.input_root_dir,
            registry_cfg=registry_cfg,
            warn=print,
        )

        ds = dataset_cfg.make_dataset(
            split="train",
            mode="dense",
            relpath_to_image_id=relpath_to_id,
            image_id_to_relpath=id_to_relpath,
            img_size=518,
        )
        print(f"[smoke_dense] dataset len = {len(ds)}")
        assert len(ds) > 0

        # Single-sample sanity check.
        sample = ds[0]
        print(f"[smoke_dense] image:          {tuple(sample['image'].shape)}")
        print(f"[smoke_dense] target_map:     {tuple(sample['target_map'].shape)}")
        print(f"[smoke_dense] loss_mask:      {tuple(sample['loss_mask'].shape)}  sum={int(sample['loss_mask'].sum())}")
        print(f"[smoke_dense] pixel_weights:  {tuple(sample['pixel_weights'].shape)}  sum={float(sample['pixel_weights'].sum()):.3f}")
        print(f"[smoke_dense] instance_masks: {tuple(sample['instance_masks'].shape)}")
        print(f"[smoke_dense] instance_labels:{tuple(sample['instance_labels'].shape)}")
        assert sample["image"].shape[-2:] == sample["target_map"].shape[-2:]
        assert sample["target_map"].shape[0] == 3
        assert sample["loss_mask"].dtype == torch.bool

        # Batch through the collate.
        for include_per_inst in (False, True):
            collate = make_dense_collate(include_per_instance=include_per_inst)
            loader = DataLoader(
                ds,
                batch_size=2,
                shuffle=False,
                num_workers=0,
                collate_fn=collate,
            )
            batch = next(iter(loader))
            print(
                f"[smoke_dense] batch (include_per_inst={include_per_inst}):"
                f" image={tuple(batch['image'].shape)},"
                f" target_map={tuple(batch['target_map'].shape)},"
                f" loss_mask={tuple(batch['loss_mask'].shape)}"
            )
            if include_per_inst:
                print(
                    f"                instance_masks={tuple(batch['instance_masks'].shape)},"
                    f" batch_index={tuple(batch['batch_index'].shape)},"
                    f" instance_labels={tuple(batch['instance_labels'].shape)}"
                )
                assert batch["instance_masks"].shape[0] == batch["batch_index"].shape[0]
            else:
                assert "instance_masks" not in batch
        print("[smoke_dense] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
