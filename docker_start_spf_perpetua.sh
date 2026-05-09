#!/bin/bash
# Start (or attach to) the Perpetua-layered SPF container.
# Mirrors docker_start_spf.sh but uses the spf_perpetua image and container name.

set -euo pipefail

TAG="${PERPETUA_TAG:-spf_perpetua:latest}"

if [ "$(podman images -q "$TAG" 2>/dev/null)" = "" ]; then
    echo "ERROR: image $TAG not found. Run ./docker_rebuild_spf_perpetua.sh first."
    exit 1
fi

if [ ! "$(podman ps -a -q -f name=^spf_perpetua_container$)" ]; then
    echo "Creating spf_perpetua_container from $TAG..."
    # Mirror the dataset bind mounts from docker_start_datasets_plus_spf.sh so
    # PKLot / KITTI / etc. paths resolve identically inside the perpetua
    # container.
    podman run -d --gpus all \
        --shm-size=12g \
        -v "$(pwd)":/workspace/code/semantic-persistence-filter \
        -v /scratch/aaadkins/data/semantic-persistence-filter:/workspace/data \
        -v /robodata/aaadkins/Datasets/semPersFilter/trainCandidates/kitti/dataset/:/workspace/data/input/kitti/ \
        -v /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/pklot/pklot:/workspace/data/input/pklot/ \
        -v /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/new_pklot:/workspace/data/input/new_pklot/ \
        -v /robodata/aaadkins/Datasets/semPersFilter/trainCandidates/epic-kitchens/downloaded/EPIC-KITCHENS/:/workspace/data/input/EPIC-KITCHENS/ \
        -v /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/exmaps/exmaps_dataset/:/workspace/data/input/exmaps/ \
        -v /robodata/aaadkins/Datasets/semPersFilter/trainCandidates/sku110k/SKU110K_fixed/images:/workspace/data/input/sku110k/ \
        -v /robodata/aaadkins/Datasets/CODa/:/workspace/data/input/CODa/ \
        -v /robodata/aaadkins/Datasets/semPersFilter/trainCandidates/sun397/data/:/workspace/data/input/sun397/ \
        -v /robodata/aaadkins/Datasets/semPersFilter/copyToTACC:/workspace/data/input/copyToTACC/ \
        -e DISPLAY="${DISPLAY:-}" \
        -v "${XAUTHORITY:-$HOME/.Xauthority}":/tmp/.Xauthority:ro \
        -e XAUTHORITY=/tmp/.Xauthority \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /scratch/aaadkins/hf_cache:/opt/hf \
        --net=host \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e JAX_PLATFORMS=cpu \
        --device /dev/dri \
        --name spf_perpetua_container \
        "$TAG" \
        tail -f /dev/null
else
    if [ ! "$(podman ps -q -f name=^spf_perpetua_container$)" ]; then
        echo "Starting existing spf_perpetua_container..."
        podman start spf_perpetua_container
    fi
fi

# Reinstall perpetua-code editable from the bind-mounted source so edits to
# src/perpetua-code/ in the host repo are picked up (overrides the build-time
# install at /opt/perpetua-code).
echo "Installing perpetua-code editable from bind mount..."
podman exec spf_perpetua_container \
    pip install --no-deps --quiet -e /workspace/code/semantic-persistence-filter/src/perpetua-code

echo "Connecting to spf_perpetua_container..."
podman exec -it spf_perpetua_container \
    bash -c "cd /workspace/code/semantic-persistence-filter && bash"
