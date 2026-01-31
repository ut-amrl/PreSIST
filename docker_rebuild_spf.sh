#!/bin/bash

# Parse arguments
# Add another option that skip the timestamp tagging and just uses the latest image
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

# Stop the container if it's running
./docker_stop_spf.sh

# Remove the container
if [ "$(podman ps -a -q -f name=spf_container)" ]; then
    echo "Removing existing semantic-persistence-filter container..."
    if ! podman rm spf_container; then
        echo "ERROR: Failed to remove existing semantic-persistence-filter container."
        exit 1
    fi
fi

# Build the image and start a new container (both with the start script)
echo "Building fresh image and starting new semantic-persistence-filter container..."

START_ARGS=""
if [ "$USE_TIMESTAMP_TAG" -eq 1 ]; then
    START_ARGS="--timestamp-tag"
fi
if [ "$FORCE_FROM_SCRATCH" -eq 1 ]; then
    START_ARGS="$START_ARGS -f"
fi

./docker_start_spf.sh $START_ARGS
