from hydra.core.hydra_config import HydraConfig
import hydra
from omegaconf import DictConfig
import os

from PIL import Image
import numpy as np
import torch

from utils.image_utils import loadImageForDatasetGeneration

def processSingleImage(
    datasetGenerator,
    image_path: str,
):
    # os.makedirs(output_dir, exist_ok=True)

    print("Reading image", image_path)

    # Read the image
    img_tensor = loadImageForDatasetGeneration(image_path)

    datasetGenerator.processImage(
        image=img_tensor,
        timeAndDate=None,
        # inputBatchInfo={'filename':image_path},
        inputBatchInfo={'path': image_path},
    )

def processImageDir(
    datasetGenerator,
    image_dir_path: str,
    seed: int = 53,
    recursive: bool = False,
):
    # os.makedirs(output_dir, exist_ok=True)

    # Get all of the image files in the directory
    # If recursive is True, walk the directory tree
    if (recursive):
        image_files = []
        for root, dirs, files in os.walk(image_dir_path):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_files.append(os.path.join(root, f))
    else:
        image_files = os.listdir(image_dir_path)
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]



    # print("Image files: ", image_files)

    # Shuffle the image files for randomness
    rng = np.random.default_rng(seed)
    rng.shuffle(image_files)

    print("Here 2!")

    # Iterate over all images in the directory
    for filename in image_files:
        image_path = os.path.join(image_dir_path, filename)
        # Add a try/except block to catch errors and continue processing other images
        try:
            processSingleImage(
                datasetGenerator=datasetGenerator,
                image_path=image_path
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")



@hydra.main(version_base=None, config_path="configs/dataset_gen_confs", config_name="config")
def main(cfg: DictConfig):

    # Access Hydra's own config
    hydra_cfg = HydraConfig.get()

    # This is the name of the top-level config (e.g. "config")
    config_name = hydra_cfg.job.config_name      # or hydra_cfg.job.name

    # Output directory from config
    # Combine output_root_dir and config_name to get the final output directory
    # output_dir = os.path.join(cfg.output_root_dir, config_name)
    # os.makedirs(output_dir, exist_ok=True)

    img_path = cfg.image_dir_path

    # Check if the recursive flag is set
    flagName = "image_dir_recursive"
    recursive = False
    if flagName in cfg and cfg[flagName]:
        recursive = cfg[flagName]
        print("Setting recursive image directory processing to ", recursive)

    print("Here!")

    datasetGenerator = hydra.utils.instantiate(cfg.dataset_generator)

    # datasetGenerator.intermediateLogger.logDirectory = output_dir
    datasetGenerator.intermediateLogger.configFileName = config_name

    processImageDir(
        datasetGenerator=datasetGenerator,
        image_dir_path=img_path,
        recursive=recursive,
    )


if __name__ == "__main__":
    main()

