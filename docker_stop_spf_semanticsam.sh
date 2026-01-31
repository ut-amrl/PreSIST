#!/bin/bash

CONTAINER_NAME="spf_container_semanticsam"

if [ "$(podman ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping Semantic-SAM container..."
    if ! podman stop $CONTAINER_NAME; then
        echo "ERROR: Failed to stop container."
        exit 1
    fi
    echo "Semantic-SAM container stopped."
else
    echo "No running Semantic-SAM container to stop."
fi
