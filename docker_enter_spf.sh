#!/bin/bash

# Check if container exists
if [ ! "$(podman ps -a -q -f name=spf_container)" ]; then
    echo "ERROR: semantic-persistence-filter container doesn't exist yet!"
    echo "Run ./docker_start_spf.sh to start the container."
    exit 1
fi

# Check if container is running
if [ ! "$(podman ps -q -f name=spf_container)" ]; then
    echo "Starting semantic-persistence-filter container..."
    if ! podman start spf_container; then
        echo "ERROR: Failed to start semantic-persistence-filter container."
        exit 1
    fi
fi

# Connect to the container
echo "Connecting to semantic-persistence-filter container..."
if ! podman exec -it spf_container bash -c "cd /workspace/code/semantic-persistence-filter && bash"; then
    echo "ERROR: Failed to connect to semantic-persistence-filter container."
    exit 1
fi

echo "Disconnected from semantic-persistence-filter container. The container is still running in the background."
echo "Use this script again to reconnect or './docker_stop_spf.sh' to stop it."