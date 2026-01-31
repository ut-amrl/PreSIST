from hydra.core.hydra_config import HydraConfig
import hydra
from omegaconf import DictConfig
import os

from PIL import Image
import numpy as np
import torch

def loadImageForDatasetGeneration(image_path: str):
    # ---- load image ----
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)  # (H, W, 3), uint8
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W), uint8

    return img_tensor

def processSingleImage(
    datasetGenerator,
    image_path: str,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # Read the image
    img_tensor = loadImageForDatasetGeneration(image_path)

    datasetGenerator.processImage(
        image=img_tensor,
        timeAndDate=None,
        inputBatchInfo={'filename':image_path},
    )


@hydra.main(version_base=None, config_path="configs/dataset_gen_confs", config_name="config")
def main(cfg: DictConfig):

    # Access Hydra's own config
    hydra_cfg = HydraConfig.get()

    # This is the name of the top-level config (e.g. "config")
    config_name = hydra_cfg.job.config_name      # or hydra_cfg.job.name

    # Output directory from config
    # Combine output_root_dir and config_name to get the final output directory
    output_dir = os.path.join(cfg.output_root_dir, config_name)
    os.makedirs(output_dir, exist_ok=True)

    img_path = cfg.image_path

    datasetGenerator = hydra.utils.instantiate(cfg.dataset_generator)

    datasetGenerator.intermediateLogger.logDirectory = output_dir

    processSingleImage(
        datasetGenerator=datasetGenerator,
        image_path=img_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()

