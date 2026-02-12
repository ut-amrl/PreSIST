#!/bin/bash

# Parse arguments
FORCE_FROM_SCRATCH=0
USE_TIMESTAMP_TAG=1
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--force) FORCE_FROM_SCRATCH=1 ;;
        -n|--no-timestamp) USE_TIMESTAMP_TAG=0 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Stop the container if running
./docker_stop_man_label.sh

# Remove the container
if [ "$(podman ps -a -q -f name=man_label_container)" ]; then
    echo "Removing existing man_label container..."
    if ! podman rm man_label_container; then
        echo "ERROR: Failed to remove existing man_label container."
        exit 1
    fi
fi

# Build the image and start container
echo "Building fresh image and starting new man_label container..."

START_ARGS=""
if [ "$USE_TIMESTAMP_TAG" -eq 1 ]; then
    START_ARGS="--timestamp-tag"
fi
if [ "$FORCE_FROM_SCRATCH" -eq 1 ]; then
    START_ARGS="$START_ARGS -f"
fi

./docker_start_man_label.sh $START_ARGS
