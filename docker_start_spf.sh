#!/bin/bash

FORCE_FROM_SCRATCH=0
SHOULD_TIMESTAMP_TAG=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--force) FORCE_FROM_SCRATCH=1 ;;
        --timestamp-tag) SHOULD_TIMESTAMP_TAG=1 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Generate timestamp tag if requested
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ "$SHOULD_TIMESTAMP_TAG" -eq 1 ]; then
    TAG="spf:$TIMESTAMP"
else
    TAG="spf:latest"
fi

if [[ "$(podman images -q $TAG 2> /dev/null)" == "" || "$FORCE_FROM_SCRATCH" -eq 1 || "$SHOULD_TIMESTAMP_TAG" -eq 1 ]]; then
    echo "Building image $TAG for semantic-persistence-filter..."
    BUILD_ARGS=""
    if [ "$FORCE_FROM_SCRATCH" -eq 1 ]; then
        echo "Force rebuilding image from scratch (don't use cache)..."
        BUILD_ARGS="--no-cache"
    fi

    if ! podman build $BUILD_ARGS -t $TAG .; then
        echo "ERROR: Docker build failed"
        exit 1
    fi

    # Update the latest tag if we're using a specific timestamp tag
    if [ "$SHOULD_TIMESTAMP_TAG" -eq 1 ]; then
        echo "Tagging $TAG as spf:latest"
        podman tag $TAG spf:latest
    fi
fi

# Check if the container already exists
if [ ! "$(podman ps -a -q -f name=^spf_container$)" ]; then
    # The container does not exist --> create and start it
    XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"

    if ! podman run -d --gpus all \
        --shm-size=12g \
        -v $(pwd):/workspace/code/semantic-persistence-filter \
        -v /scratch/tarunrav/data/semantic-persistence-filter:/workspace/data \
        -e DISPLAY=$DISPLAY \
        -v $XAUTHORITY:/tmp/.Xauthority:ro \
        -e XAUTHORITY=/tmp/.Xauthority \
        -v /tmp/.X11-unix:/tmp/.X11-unix  \
        -v /scratch/tarunrav/hf_cache:/opt/hf \
        --net=host  \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        --device /dev/dri \
        --name spf_container \
        $TAG \
        tail -f /dev/null; then
        echo "Error: Failed to create container. Please check the docker run command and try again."
        exit 1
    fi

    # This is where any additional setup commands would go, if needed in the future

else
    # Check if the container is stopped
    if [ ! "$(podman ps -q -f name=^spf_container$)" ]; then
        echo "Starting existing semantic-persistence-filter container..."
        if ! podman start spf_container; then
            echo "Error: Failed to start container."
            exit 1
        fi
    else
        echo "semantic-persistence-filter container is already running."
    fi
fi

# Connect to the container
echo "Connecting to semantic-persistence-filter container..."
if ! podman exec -it spf_container bash -c "cd /workspace/code/semantic-persistence-filter && bash"; then
    echo "Error: Failed to connect to container."
    exit 1
fi

echo "Disconnected from semantic-persistence-filter container. The container is still running in the background."
echo "Use './docker_enter_spf.sh' to reconnect or './docker_stop_spf.sh' to stop it."