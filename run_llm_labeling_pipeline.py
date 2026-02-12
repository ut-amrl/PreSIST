from hydra.core.hydra_config import HydraConfig
import hydra
from omegaconf import DictConfig
import os

from PIL import Image
import numpy as np
import torch

from utils.image_utils import loadImageForDatasetGeneration
from dataset_generation.images_dataset.preprocess import preprocess_and_build_dataloader


def runProcess(dataset, loader, db_path, meta, datasetGenerator):
    return datasetGenerator.generateDataset(loader)


@hydra.main(version_base=None, config_path="configs/dataset_gen_confs", config_name="config")
def main(cfg: DictConfig):

    # Access Hydra's own config
    hydra_cfg = HydraConfig.get()

    # This is the name of the top-level config (e.g. "config")
    config_name = hydra_cfg.job.config_name      # or hydra_cfg.job.name

    # Check if there is a image_dataset entry and retrieve it
    if "image_dataset" not in cfg:
        raise ValueError("Config must have a 'image_dataset' entry specifying the dataset generator to use.")

    dataset_config = hydra.utils.instantiate(cfg.image_dataset)

    ds_kwargs = dataset_config.to_preprocess_kwargs()

    registryConfig = hydra.utils.instantiate(cfg.registry_info)

    dataset, loader, db_path, meta = preprocess_and_build_dataloader(
        image_loader=loadImageForDatasetGeneration,
        registry_root=registryConfig.registry_root,
        strict=registryConfig.strict,
        export_human_readable=registryConfig.export_human_readable,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle=cfg.dataloader.shuffle,
        pin_memory=cfg.dataloader.pin_memory,
        **ds_kwargs,
    )

    datasetGenerator = hydra.utils.instantiate(cfg.dataset_generator)

    # datasetGenerator.intermediateLogger.logDirectory = output_dir
    datasetGenerator.intermediateLogger.configFileName = config_name

    # TODO hook up to caching dataset generation process loader

    runProcess(dataset, loader, db_path, meta, datasetGenerator)



if __name__ == "__main__":
    main()

# dataset:
#   image_root: /work/datasets
#   spec_path: /work/images.txt

# registry:
#   registry_root: /work/registry
#   strict: true
#   export_human_readable: true

# dataloader:
#   batch_size: 8
#   num_workers: 4