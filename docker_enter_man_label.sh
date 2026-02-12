#!/bin/bash

# Check if container exists
if [ ! "$(podman ps -a -q -f name=^man_label_container$)" ]; then
    echo "ERROR: man_label container doesn't exist yet!"
    echo "Run ./docker_start_man_label.sh to start the container."
    exit 1
fi

# Check if container is running
if [ ! "$(podman ps -q -f name=^man_label_container$)" ]; then
    echo "Starting man_label container..."
    if ! podman start man_label_container; then
        echo "ERROR: Failed to start man_label container."
        exit 1
    fi
fi

# Connect to the container
echo "Connecting to man_label container..."
if ! podman exec -it man_label_container bash -c "cd /workspace/code/semantic-persistence-filter && bash"; then
    echo "ERROR: Failed to connect to man_label container."
    exit 1
fi

echo "Disconnected from man_label container. The container is still running in the background."
echo "Use this script again to reconnect or './docker_stop_man_label.sh' to stop it."
