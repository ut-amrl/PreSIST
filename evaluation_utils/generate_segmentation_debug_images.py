import os
import numpy as np

import sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

from utils.visualization_utils import saveMasksOnImages
from utils.image_utils import loadImageAsTensor, loadImageForDatasetGeneration
import argparse
import os
from instance_segmentation.abs_instance_segmenter import SegmentationResults
import torch



# Print the path
print("Sys path:", sys.path)


# Arguments
# - root level directory: no default
# - segmentation results numpy file name: default segmentation_results.npz
# - overwrite existing images: bool, default False

# For any subdirectory of the root level directory containing segmentation results numpy file:
# - Load segmentation results from numpy file
# - Call saveMasksOnImages to save debug images in that directory


def argument_parser():
    parser = argparse.ArgumentParser(
        description="Generate segmentation debug images from segmentation results numpy files."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing segmentation results numpy files.",
    )
    parser.add_argument(
        "--segmentation_results_filename",
        type=str,
        default="segmentation_results.npz",
        help="Filename of the segmentation results numpy file in each subdirectory.",
    )
    parser.add_argument(
        "--overwriteExisting",
        action="store_true",
        help="Whether to overwrite existing debug images.",
    )
    # Add an argument to optionally exclude files that contain an underscore in their base name
    parser.add_argument(
        "--excludeUnderscoreFiles",
        action="store_true",
        help="Whether to exclude files that contain an underscore in their base name.",
    )

    # Add an option to optionally save the debug images to a different directory
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default="",
        help="Root directory to save debug images. If not provided, the debug images will be saved in the same directory as the segmentation results.",
    )
    return parser

if __name__ == "__main__":
    args = argument_parser().parse_args()



    root_dir = args.root_dir
    segmentation_results_filename = args.segmentation_results_filename
    overwriteExisting = args.overwriteExisting
    excludeUnderscoreFiles = args.excludeUnderscoreFiles
    print("Exclude files with underscore in base name: ", excludeUnderscoreFiles)

    # Walk the directory tree starting from root_dir
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if segmentation_results_filename in filenames:
            # Get the name of the directory that contains the segmentation results file
            dirName = os.path.basename(dirpath)
            if (excludeUnderscoreFiles and "_" in dirName):
                continue
            segmentation_results_path = os.path.join(dirpath, segmentation_results_filename)
            print(f"Processing segmentation results in: {segmentation_results_path}")

            segmentationResults, imgFileName = SegmentationResults.loadFromFile(segmentation_results_path)
            print("Loaded segmentation results with {} instances.".format(len(segmentationResults)))
            print("Image file name from segmentation results: ", imgFileName)


            if (imgFileName is None) or (not os.path.isabs(imgFileName)):
                print(f"  Warning: imagePath not found or not absolute in segmentation results file. Skipping debug image generation for this file.")
                continue
            instance_masks = segmentationResults.masks

            # Load the original image
            print("Image file name from segmentation results: ", imgFileName)
            image = loadImageForDatasetGeneration(imgFileName)


            # Save debug images
            saveMasksOnImages(
                masks=instance_masks,
                image=image,
                outputDir=dirpath,
                overwriteExisting=overwriteExisting,
                scores=segmentationResults.scores,
            )


"""
Example usage:
python generate_segmentation_debug_images.py  --excludeUnderscoreFiles --root_dir /workspace/data/output/
"""
