from hydra.core.hydra_config import HydraConfig
import hydra
from omegaconf import DictConfig

from utils.image_utils import loadImageForDatasetGeneration
from dataset_generation.images_dataset.preprocess import preprocess_and_build_dataloader


@hydra.main(version_base=None, config_path="configs/dataset_gen_confs", config_name="config")
def main(cfg: DictConfig):
    if "image_dataset" not in cfg:
        raise ValueError("Config must have a 'image_dataset' entry specifying the dataset to index.")

    dataset_config = hydra.utils.instantiate(cfg.image_dataset)
    ds_kwargs = dataset_config.to_preprocess_kwargs()
    registryConfig = hydra.utils.instantiate(cfg.registry_info)

    _, _, db_path, _ = preprocess_and_build_dataloader(
        image_loader=loadImageForDatasetGeneration,
        registry_root=registryConfig.registry_root,
        strict=registryConfig.strict,
        export_human_readable=registryConfig.export_human_readable,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle=False,
        pin_memory=False,
        **ds_kwargs,
    )

    print(f"Registry pre-cached: {db_path}")


if __name__ == "__main__":
    main()
