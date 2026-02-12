#!/bin/bash

# Check if running
if [ "$(podman ps -q -f name=man_label_container)" ]; then
    echo "Stopping man_label container..."
    if ! podman stop man_label_container; then
        echo "Error: Failed to stop container."
        exit 1
    fi
    echo "Container stopped."
else
    echo "No running man_label container found; nothing to stop."
fi
