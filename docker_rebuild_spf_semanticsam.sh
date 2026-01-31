#!/bin/bash

CONTAINER_NAME="spf_container_semanticsam"
START_SCRIPT="./docker_start_spf_semanticsam.sh"

FORCE_FROM_SCRATCH=0
USE_TIMESTAMP_TAG=1

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--force) FORCE_FROM_SCRATCH=1 ;;
        -n|--no-timestamp) USE_TIMESTAMP_TAG=0 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Stop old container
./docker_stop_spf_semanticsam.sh

# Remove old container
if [ "$(podman ps -a -q -f name=$CONTAINER_NAME)" ]; then
    echo "Removing existing Semantic-SAM container..."
    if ! podman rm $CONTAINER_NAME; then
        echo "ERROR: Failed to remove container."
        exit 1
    fi
fi

# Build and start new container
echo "Rebuilding Semantic-SAM image and container..."

START_ARGS=""
if [ "$USE_TIMESTAMP_TAG" -eq 1 ]; then
    START_ARGS="--timestamp-tag"
fi
if [ "$FORCE_FROM_SCRATCH" -eq 1 ]; then
    START_ARGS="$START_ARGS -f"
fi

$START_SCRIPT $START_ARGS
