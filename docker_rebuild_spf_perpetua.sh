#!/bin/bash
# Build (or rebuild) the Perpetua-layered SPF image and recreate its container.
# The image FROMs spf:latest unless overridden with BASE_IMAGE_TAG.

set -euo pipefail

BASE_IMAGE_TAG="${BASE_IMAGE_TAG:-spf_with_datasets:saved}"
TAG="${PERPETUA_TAG:-spf_perpetua:latest}"

# Sanity check: the saved-from-live tag must exist. If it doesn't, the live
# spf_container hasn't been committed yet — run docker_start_datasets_plus_spf.sh
# with --save_from_spf first.
if [ "$(podman images -q "$BASE_IMAGE_TAG" 2>/dev/null)" = "" ]; then
    echo "ERROR: base image $BASE_IMAGE_TAG not found."
    echo "Commit the live spf_container first:"
    echo "    ./docker_start_datasets_plus_spf.sh --save_from_spf"
    echo "Or override BASE_IMAGE_TAG=<other_tag> when invoking this script."
    exit 1
fi

# Stop / remove the container if present.
if [ "$(podman ps -a -q -f name=^spf_perpetua_container$)" ]; then
    echo "Removing existing spf_perpetua_container..."
    podman stop spf_perpetua_container >/dev/null 2>&1 || true
    podman rm spf_perpetua_container
fi

echo "Building $TAG from BASE_IMAGE=$BASE_IMAGE_TAG ..."
podman build \
    --build-arg "BASE_IMAGE=$BASE_IMAGE_TAG" \
    -f PerpetuaDockerfile \
    -t "$TAG" \
    .

echo "Built $TAG."
echo "Run ./docker_start_spf_perpetua.sh to start a container."
