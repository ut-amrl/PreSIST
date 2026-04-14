#!/bin/bash
# Start (or restart) a Describe Anything Model (DAM) server container.
#
# The container uses the pre-built image from the ut-amrl fork of describe-anything,
# which adds a /batch/chat/completions endpoint for efficient multi-mask inference.
#
# Usage examples:
#   ./docker_start_dam_server.sh                                               # all GPUs, default port 9014
#   ./docker_start_dam_server.sh -f                                            # force re-pull + recreate container
#   DAM_HOST_PORT=9002 ./docker_start_dam_server.sh                           # custom port, all GPUs
#   DAM_CUDA_VISIBLE_DEVICES=0 ./docker_start_dam_server.sh                   # GPU 0 only, port 9014
#   DAM_CUDA_VISIBLE_DEVICES=0,1 DAM_HOST_PORT=9002 ./docker_start_dam_server.sh  # GPU 0+1, port 9002
#   HF_CACHE_HOST=/my/hf/cache ./docker_start_dam_server.sh                   # custom HF cache path

FORCE=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--force) FORCE=1 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

IMAGE="docker.io/mandi1267/describe-anything:latest"
CONTAINER_NAME="dam_server_container"
HOST_PORT="${DAM_HOST_PORT:-9014}"
CONTAINER_PORT="9014"
HF_CACHE_HOST="${HF_CACHE_HOST:-/scratch/aaadkins/hf_cache}"
HF_CACHE_CONTAINER="/workspace/hf_home"

# Build GPU env args: set CUDA_VISIBLE_DEVICES directly inside the container.
# dam_server.py is run via "python dam_server.py" (not uvicorn) so CUDA_VISIBLE_DEVICES
# must be set before the process starts rather than relying on run_describe_anything_api.sh.
GPU_ENV_ARGS=()
if [ -n "${DAM_CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_ENV_ARGS+=(-e "CUDA_VISIBLE_DEVICES=${DAM_CUDA_VISIBLE_DEVICES}")
    echo "GPU selection: CUDA_VISIBLE_DEVICES=${DAM_CUDA_VISIBLE_DEVICES}"
else
    echo "GPU selection: all GPUs visible (set DAM_CUDA_VISIBLE_DEVICES to restrict)"
fi

echo "DAM server image:     $IMAGE"
echo "Container name:       $CONTAINER_NAME"
echo "Host port:            $HOST_PORT"
echo "HF cache (host):      $HF_CACHE_HOST"
echo "==========================================="

# Force mode: stop + remove the existing container and re-pull the image
if [ "$FORCE" -eq 1 ]; then
    echo "Force mode: removing existing container and re-pulling image..."
    podman stop "$CONTAINER_NAME" 2>/dev/null || true
    podman rm "$CONTAINER_NAME" 2>/dev/null || true
    if ! podman pull "$IMAGE"; then
        echo "ERROR: Failed to pull image $IMAGE"
        exit 1
    fi
fi

# Pull image if not present locally
if [[ "$(podman images -q "$IMAGE" 2>/dev/null)" == "" ]]; then
    echo "Image not found locally. Pulling $IMAGE ..."
    if ! podman pull "$IMAGE"; then
        echo "ERROR: Failed to pull image $IMAGE"
        exit 1
    fi
fi

# Check if container already exists
if [ ! "$(podman ps -a -q -f name=^${CONTAINER_NAME}$)" ]; then
    echo "Creating and starting DAM server container..."
    if ! podman run -d \
        --gpus all \
        --shm-size=12g \
        --net=host \
        -e "DAM_PORT=${CONTAINER_PORT}" \
        -e "HF_HOME=${HF_CACHE_CONTAINER}" \
        -e PYTHONUNBUFFERED=1 \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        "${GPU_ENV_ARGS[@]}" \
        -v "${HF_CACHE_HOST}:${HF_CACHE_CONTAINER}" \
        --name "$CONTAINER_NAME" \
        "$IMAGE" \
        python /workspace/project/dam_server.py; then
        echo "ERROR: Failed to create DAM server container."
        exit 1
    fi
    echo "DAM server container started. Listening on port ${HOST_PORT}."
    echo "Configure your client with: server_base_url=http://localhost  server_port=${HOST_PORT}"
else
    # Container exists — start it if stopped
    if [ ! "$(podman ps -q -f name=^${CONTAINER_NAME}$)" ]; then
        echo "Starting existing DAM server container..."
        if ! podman start "$CONTAINER_NAME"; then
            echo "ERROR: Failed to start DAM server container."
            exit 1
        fi
        echo "DAM server container restarted. Listening on port ${HOST_PORT}."
    else
        echo "DAM server container is already running on port ${HOST_PORT}."
    fi
fi

echo "Use 'podman logs -f $CONTAINER_NAME' to follow server logs."
echo "Use 'podman stop $CONTAINER_NAME' to stop the server."
