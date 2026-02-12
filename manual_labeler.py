# scripts/label_gui.py
from __future__ import annotations
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from utils.image_utils import loadImageForDatasetGeneration
from dataset_generation.images_dataset.preprocess import preprocess_and_build_dataloader

from manual_labeling.image_source import DatasetImageSource
from manual_labeling.gui_qt import run_gui, LabelingPaths


@hydra.main(version_base=None, config_path="configs/dataset_gen_confs", config_name="config")
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    # _config_name = hydra_cfg.job.config_name
    semantic_segmentation_config_name = cfg.dataset_generator.instanceSegmenter.name
    outputDirRoot = cfg.dataset_generator.intermediateLogger.baseLogDirectory

    dataset_config = hydra.utils.instantiate(cfg.image_dataset)
    ds_kwargs = dataset_config.to_preprocess_kwargs()
    registryConfig = hydra.utils.instantiate(cfg.registry_info)

    dataset, loader, db_path, meta = preprocess_and_build_dataloader(
        image_loader=loadImageForDatasetGeneration,
        registry_root=registryConfig.registry_root,
        strict=registryConfig.strict,
        export_human_readable=registryConfig.export_human_readable,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        **ds_kwargs,
    )

    image_source = DatasetImageSource(dataset)

    paths = LabelingPaths(
        output_root=Path(outputDirRoot),
        segmentation_config_name=semantic_segmentation_config_name,
        labeling_tag=str(cfg.labeling_tag),
    )

    print("Semantic segmentation config name:", semantic_segmentation_config_name)

    key_map = dict(cfg.get("json_key_map", {}))
    run_gui(image_source=image_source, paths=paths, key_map=key_map)


if __name__ == "__main__":
    main()
