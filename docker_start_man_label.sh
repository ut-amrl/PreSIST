#!/bin/bash

FORCE_FROM_SCRATCH=0
SHOULD_TIMESTAMP_TAG=0
DOCKERFILE="ManualLabelerDockerfile"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--force) FORCE_FROM_SCRATCH=1 ;;
        --timestamp-tag) SHOULD_TIMESTAMP_TAG=1 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ "$SHOULD_TIMESTAMP_TAG" -eq 1 ]; then
    TAG="man_label:$TIMESTAMP"
else
    TAG="man_label:latest"
fi

if [[ "$(podman images -q $TAG 2> /dev/null)" == "" || "$FORCE_FROM_SCRATCH" -eq 1 || "$SHOULD_TIMESTAMP_TAG" -eq 1 ]]; then
    echo "Building image $TAG for man_label using Dockerfile $DOCKERFILE..."
    BUILD_ARGS=""
    if [ "$FORCE_FROM_SCRATCH" -eq 1 ]; then
        echo "Force rebuilding image from scratch..."
        BUILD_ARGS="--no-cache"
    fi

    if ! podman build $BUILD_ARGS -t $TAG -f $DOCKERFILE .; then
        echo "ERROR: Docker build failed"
        exit 1
    fi

    if [ "$SHOULD_TIMESTAMP_TAG" -eq 1 ]; then
        echo "Tagging $TAG as man_label:latest"
        podman tag $TAG man_label:latest
    fi
fi

# Create container if missing
if [ ! "$(podman ps -a -q -f name=^man_label_container$)" ]; then
    if ! podman run -d --gpus all \
        --shm-size=12g \
        -v $(pwd):/workspace/code/semantic-persistence-filter \
        -v /scratch/aaadkins/data/semantic-persistence-filter:/workspace/data \
        -e DISPLAY=$DISPLAY \
        -v $XAUTHORITY:/tmp/.Xauthority:ro \
        -e XAUTHORITY=/tmp/.Xauthority \
        -v /tmp/.X11-unix:/tmp/.X11-unix  \
        -v /scratch/aaadkins/hf_cache:/opt/hf \
        --net=host \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        --device /dev/dri \
        --name man_label_container \
        $TAG \
        tail -f /dev/null; then
        echo "Error: Failed to create container."
        exit 1
    fi
else
    if [ ! "$(podman ps -q -f name=^man_label_container$)" ]; then
        echo "Starting existing man_label container..."
        if ! podman start man_label_container; then
            echo "Error: Failed to start container."
            exit 1
        fi
    else
        echo "man_label container is already running."
    fi
fi

# Connect interactively
echo "Connecting to man_label container..."
if ! podman exec -it man_label_container bash -c "cd /workspace/code/semantic-persistence-filter && bash"; then
    echo "Error: Failed to connect to container."
    exit 1
fi

echo "Disconnected. The container is still running background."
echo "Use './docker_enter_man_label.sh' to reconnect or './docker_stop_man_label.sh' to stop it."
