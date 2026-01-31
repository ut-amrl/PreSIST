#!/bin/bash

IMAGE_PREFIX="spf_semanticsam"
CONTAINER_NAME="spf_container_semanticsam"
DOCKERFILE="SemanticSAMDockerfile"

FORCE_FROM_SCRATCH=0
SHOULD_TIMESTAMP_TAG=0

# Parse args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--force) FORCE_FROM_SCRATCH=1 ;;
        --timestamp-tag) SHOULD_TIMESTAMP_TAG=1 ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Tag
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ "$SHOULD_TIMESTAMP_TAG" -eq 1 ]; then
    TAG="$IMAGE_PREFIX:$TIMESTAMP"
else
    TAG="$IMAGE_PREFIX:latest"
fi

# Build if needed
if [[ "$(podman images -q $TAG)" == "" || "$FORCE_FROM_SCRATCH" -eq 1 || "$SHOULD_TIMESTAMP_TAG" -eq 1 ]]; then

    echo "Building image $TAG using Dockerfile: $DOCKERFILE"

    BUILD_ARGS=""
    if [ "$FORCE_FROM_SCRATCH" -eq 1 ]; then
        BUILD_ARGS="--no-cache"
    fi

    if ! podman build $BUILD_ARGS -t $TAG -f $DOCKERFILE .; then
        echo "ERROR: Build failed"
        exit 1
    fi

    # Update latest tag if timestamp was used
    if [ "$SHOULD_TIMESTAMP_TAG" -eq 1 ]; then
        podman tag $TAG $IMAGE_PREFIX:latest
    fi
fi

# Start container
if [ ! "$(podman ps -a -q -f name=$CONTAINER_NAME)" ]; then

    echo "Creating Semantic-SAM container: $CONTAINER_NAME"

    if ! podman run -d --gpus all \
        --shm-size=12g \
        -v "$(pwd)":/workspace/code/semantic-persistence-filter \
        -v /scratch/aaadkins/data/semantic-persistence-filter:/workspace/data \
        -e DISPLAY="${DISPLAY:-}" \
        -v "${XAUTHORITY:-/dev/null}":/tmp/.Xauthority:ro \
        -e XAUTHORITY=/tmp/.Xauthority \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v /scratch/aaadkins/hf_cache:/opt/hf \
        --net=host \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        --device /dev/dri \
        --name "$CONTAINER_NAME" \
        "$TAG" \
        tail -f /dev/null; then

        echo "ERROR: Failed running Semantic-SAM container."
        exit 1
    fi

else
    if [ ! "$(podman ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "Starting existing Semantic-SAM container..."
        podman start "$CONTAINER_NAME" || {
            echo "ERROR: Failed to start container."
            exit 1
        }
    else
        echo "Semantic-SAM container is already running."
    fi
fi

# ---------- One-time bootstrap: compile Semantic-SAM CUDA op ----------
echo "Bootstrapping Semantic-SAM CUDA extension (if needed)..."

podman exec -it "$CONTAINER_NAME" bash -lc "
set -euo pipefail

python - <<PY
import torch, numpy as np
print('torch:', torch.__version__, 'torch_cuda:', torch.version.cuda, 'cuda_available:', torch.cuda.is_available())
print('numpy:', np.__version__)
PY

# Try import first
python -c \"import MultiScaleDeformableAttention as MSDA; print('MultiScaleDeformableAttention already available')\" 2>/dev/null && exit 0

OPS_DIR=/workspace/semantic-sam/semantic_sam/body/encoder/ops
CUDA_SRC=\$OPS_DIR/src/cuda/ms_deform_attn_cuda.cu
HDR_SRC=\$OPS_DIR/src/ms_deform_attn.h

echo \"Patching MSDeformAttn sources to avoid Tensor.type() deprecations...\"
sed -i \"s/AT_DISPATCH_FLOATING_TYPES(value.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g\" \"\$CUDA_SRC\"
sed -i \"s/\\\\.type()\\\\.is_cuda()/\\\\.is_cuda()/g\" \"\$CUDA_SRC\" \"\$HDR_SRC\"
sed -i \"s/\\\\.data</\\\\.data_ptr</g\" \"\$CUDA_SRC\" || true

echo \"Remaining .type() references (should be none or only non-critical):\"
grep -n \"\\\\.type()\" \"\$CUDA_SRC\" \"\$HDR_SRC\" || true

echo \"Compiling MultiScaleDeformableAttention...\"
cd \"\$OPS_DIR\"

export FORCE_CUDA=1
export CUDA_HOME=\${CUDA_HOME:-/usr/local/cuda}
export TORCH_CUDA_ARCH_LIST=\${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0}
export MAX_JOBS=\${MAX_JOBS:-8}

rm -rf build *.so dist *.egg-info /root/.local/lib/python3.10/site-packages/MultiScaleDeformableAttention* || true
sh make.sh

# Ensure torch shared libs are discoverable
TORCH_LIB=\$(python - <<PY
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)
echo \"torch lib dir: \$TORCH_LIB\"
export LD_LIBRARY_PATH=\"\$TORCH_LIB:\${LD_LIBRARY_PATH:-}\"

# Copy built extension somewhere importable and add to PYTHONPATH
BUILT_SO=\$(ls -1 \"\$OPS_DIR\"/build/lib.linux-x86_64-3.10/MultiScaleDeformableAttention*.so 2>/dev/null | head -n 1 || true)
if [ -z \"\$BUILT_SO\" ]; then
  echo \"ERROR: Built .so not found in \$OPS_DIR/build/lib.linux-x86_64-3.10\"
  ls -la \"\$OPS_DIR/build\" || true
  exit 1
fi

cp -f \"\$BUILT_SO\" \"\$OPS_DIR/\"
echo \"Copied extension to: \$OPS_DIR/\$(basename \"\$BUILT_SO\")\"

export PYTHONPATH=\"\$OPS_DIR:\${PYTHONPATH:-}\"

python -c \"import MultiScaleDeformableAttention as MSDA; print('built OK:', MSDA)\"
"


echo "Disconnected. Container continues running."
echo "Use \"./docker_enter_spf_semanticsam.sh\" to reconnect."
