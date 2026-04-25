#!/bin/bash
set -euo pipefail

SAVE_FROM_SPF=0
TARGET_CONTAINER="pf_plus_datasets"
SOURCE_CONTAINER="spf_container"
IMAGE_REPO="spf_with_datasets"
SAVED_TAG="${IMAGE_REPO}:saved"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --save_from_spf) SAVE_FROM_SPF=1 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TIMESTAMPED_SAVED_TAG="${IMAGE_REPO}:saved_${TIMESTAMP}"

echo "Source container: $SOURCE_CONTAINER"
echo "Target container: $TARGET_CONTAINER"

if [ "$SAVE_FROM_SPF" -eq 1 ]; then
    echo "Refreshing saved image from live container ${SOURCE_CONTAINER}..."

    if [ -z "$(podman ps -q -f name="^${SOURCE_CONTAINER}$")" ]; then
        echo "Error: ${SOURCE_CONTAINER} is not running, so --save_from_spf cannot be used."
        exit 1
    fi

    echo "Committing ${SOURCE_CONTAINER} -> ${TIMESTAMPED_SAVED_TAG}"
    podman commit "$SOURCE_CONTAINER" "$TIMESTAMPED_SAVED_TAG"

    echo "Updating rolling saved tag -> ${SAVED_TAG}"
    podman tag "$TIMESTAMPED_SAVED_TAG" "$SAVED_TAG"
else
    echo "Using previously saved image: ${SAVED_TAG}"
    if [[ "$(podman images -q "$SAVED_TAG" 2> /dev/null)" == "" ]]; then
        echo "Error: No previously saved image found at ${SAVED_TAG}"
        echo "Run this script once with --save_from_spf first."
        exit 1
    fi
fi

# Always use the rolling saved tag for the dataset container
RUN_TAG="$SAVED_TAG"

# If target container doesn't exist, create it
if [ ! "$(podman ps -a -q -f name="^${TARGET_CONTAINER}$")" ]; then
    echo "Creating new container ${TARGET_CONTAINER}..."

    if ! podman run -d --gpus all \
        --shm-size=12g \
        -v "$(pwd)":/workspace/code/semantic-persistence-filter \
        -v /scratch/aaadkins/data/semantic-persistence-filter:/workspace/data \
        -v /robodata/aaadkins/Datasets/semPersFilter/trainCandidates/kitti/dataset/:/workspace/data/input/kitti/ \
        -v /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/pklot/pklot:/workspace/data/input/pklot/ \
        -v /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/new_pklot:/workspace/data/input/new_pklot/ \
        -v /robodata/aaadkins/Datasets/semPersFilter/trainCandidates/epic-kitchens/downloaded/EPIC-KITCHENS/:/workspace/data/input/EPIC-KITCHENS/ \
        -v /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/exmaps/exmaps_dataset/:/workspace/data/input/exmaps/ \
        -v /robodata/aaadkins/Datasets/semPersFilter/trainCandidates/sku110k/SKU110K_fixed/images:/workspace/data/input/sku110k/ \
        -v /robodata/aaadkins/Datasets/CODa/:/workspace/data/input/CODa/ \
        -v /robodata/aaadkins/Datasets/semPersFilter/trainCandidates/sun397/data/:/workspace/data/input/sun397/ \
        -v /robodata/aaadkins/Datasets/semPersFilter/copyToTACC:/workspace/data/input/copyToTACC/ \
        -e DISPLAY="${DISPLAY:-}" \
        -v "${XAUTHORITY:-$HOME/.Xauthority}":/tmp/.Xauthority:ro \
        -e XAUTHORITY=/tmp/.Xauthority \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /scratch/aaadkins/hf_cache:/opt/hf \
        --net=host \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        --device /dev/dri \
        --name "$TARGET_CONTAINER" \
        "$RUN_TAG" \
        tail -f /dev/null; then
        echo "Error: Failed to create container ${TARGET_CONTAINER}."
        exit 1
    fi

else
    if [ ! "$(podman ps -q -f name="^${TARGET_CONTAINER}$")" ]; then
        echo "Starting existing container ${TARGET_CONTAINER}..."
        if ! podman start "$TARGET_CONTAINER"; then
            echo "Error: Failed to start ${TARGET_CONTAINER}."
            exit 1
        fi
    else
        echo "Container ${TARGET_CONTAINER} is already running."
    fi
fi

echo "Connecting to ${TARGET_CONTAINER}..."
if ! podman exec -it "$TARGET_CONTAINER" bash -c "cd /workspace/code/semantic-persistence-filter && bash"; then
    echo "Error: Failed to connect to container."
    exit 1
fi

echo "Disconnected from ${TARGET_CONTAINER}. The container is still running in the background."