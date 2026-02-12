from __future__ import annotations

import sys
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from dataset_generation.images_dataset.image_id_fn import ImageIdToRelpath
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from training.dataset.split_management import SplitGenConfig, generate_random_splits, write_split_json



# Replace this with your real mapping
def image_id_to_relpath(image_id: str) -> str:
    raise NotImplementedError("Provide your image_id_to_relpath() implementation")


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: SplitGenConfig) -> None:
    # # cfg contains only SplitGenConfig fields; other hydra config groups can still be loaded if you use defaults.yaml etc.
    # print(OmegaConf.to_yaml(OmegaConf.structured(cfg)))

    obj = generate_random_splits(cfg, image_id_to_relpath=image_id_to_relpath)
    write_split_json(obj, cfg.split_json_out)
    print(f"Wrote split json: {cfg.split_json_out}")


@hydra.main(version_base=None, config_path="../configs/dataset_gen_confs", config_name="config")
def main(cfg: DictConfig):
    splitGenConfig = hydra.utils.instantiate(cfg.split_gen_config)
    outputDirRoot = cfg.dataset_generator.intermediateLogger.baseLogDirectory
    # Override the output_root_dir in splitGenConfig with the one from the dataset generator config
    splitGenConfig.output_root_dir = outputDirRoot
    print("SplitGenConfig:", splitGenConfig)

    registryConfig = hydra.utils.instantiate(cfg.registry_info)

    image_id_to_relpath = ImageIdToRelpath(
        image_root=cfg.image_dataset.image_root,
        registry_cfg=registryConfig,
        warn=print,  # or your logger.warning
    )

    obj = generate_random_splits(splitGenConfig, image_id_to_relpath=image_id_to_relpath)
    write_split_json(obj, splitGenConfig.split_json_out)
    print(f"Wrote split json: {splitGenConfig.split_json_out}")


if __name__ == "__main__":
    main()
