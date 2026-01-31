#!/bin/bash

CONTAINER_NAME="spf_container_semanticsam"

# Check if container exists
if [ ! "$(podman ps -a -q -f name=$CONTAINER_NAME)" ]; then
    echo "ERROR: Semantic-SAM container doesn't exist yet!"
    echo "Run ./docker_start_spf_semanticsam.sh to start the container."
    exit 1
fi

# Check if container is running
if [ ! "$(podman ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Starting Semantic-SAM container..."
    if ! podman start $CONTAINER_NAME; then
        echo "ERROR: Failed to start Semantic-SAM container."
        exit 1
    fi
fi

# Connect to the container
echo "Connecting to Semantic-SAM container..."
if ! podman exec -it $CONTAINER_NAME bash -c "cd /workspace/code/semantic-persistence-filter && bash"; then
    echo "ERROR: Failed to connect to Semantic-SAM container."
    exit 1
fi

echo "Disconnected from Semantic-SAM container (still running)."
echo "Use this script again to reconnect or './docker_stop_spf_semanticsam.sh' to stop it."
