#!/bin/bash
# Reconnect to a running spf_perpetua_container.

set -euo pipefail

if [ ! "$(podman ps -a -q -f name=^spf_perpetua_container$)" ]; then
    echo "ERROR: spf_perpetua_container does not exist. Run ./docker_start_spf_perpetua.sh first."
    exit 1
fi

if [ ! "$(podman ps -q -f name=^spf_perpetua_container$)" ]; then
    echo "Starting spf_perpetua_container..."
    podman start spf_perpetua_container
fi

podman exec -it spf_perpetua_container \
    bash -c "cd /workspace/code/semantic-persistence-filter && bash"
