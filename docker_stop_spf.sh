#!/bin/bash

# Check if container exists and is running -- if so, stop it
if [ "$(podman ps -q -f name=spf_container)" ]; then
    echo "Stopping semantic-persistence-filter container..."
    if ! podman stop spf_container; then
        echo "Error: Failed to stop container."
        exit 1
    fi
    echo "Container stopped."
else
    echo "No running semantic-persistence-filter container found; nothing to stop."
fi