#!/usr/bin/env bash
set -euo pipefail
# Start a TRT-LLM server for Gemma 4 (OpenAI-compatible) via Docker.
# Gemma 4 is natively multimodal, so this script always uses --backend pytorch
# and disables KV cache block reuse (required for multimodal models).
#
# NOTE: TRT-LLM 26.03 may not fully support Gemma 4 multimodal. If you hit an
# "unknown architecture" or build error, try a newer NGC image tag (e.g. 26.06+).
#
# NOTE: Gemma 4 is a gated model on HuggingFace. Set HUGGING_FACE_HUB_TOKEN
# before running (or export it in your shell environment).

############################################
# ========== EDIT THESE DEFAULTS ===========
############################################

# ---- Networking ----
PORT=9008
HOST="0.0.0.0"

# ---- Model selection (HF repo or local path) ----
# MODEL="google/gemma-4-27b-it"
MODEL="google/gemma-4-31B-it"
# MODEL="google/gemma-4-9b-it"

# ---- Engine/runtime params ----
# NOTE: trtllm-serve has no --dtype or --quantization CLI flags.
# Quantization is baked in at engine-build time (or via HF checkpoint format).
TENSOR_PARALLEL_SIZE=""           # leave empty to auto = number of GPUs passed
KV_CACHE_FREE_GPU_MEM_FRACTION="0.90"
MAX_SEQ_LEN="32768"
MAX_BATCH_SIZE=""                 # leave empty to use model default

# ---- Container image ----
# NOTE: As of 2026-04-23, no publicly available TRT-LLM image supports Gemma 4.
# 26.03 is the latest on NGC and TRT-LLM 1.2.0 does not have gemma4 in its model registry.
# Check for newer tags at:
#   https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags
# When a new tag appears, bump this and try again.
TRTLLM_IMAGE="nvcr.io/nvidia/tritonserver:26.03-trtllm-python-py3"

# HF cache (host)
HF_CACHE="${HOME}/.cache/huggingface"

# Optional: HF token (required for gated Gemma 4 weights)
HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

# pip packages to upgrade at container startup.
# Only openai needed — 26.06+ bundles transformers/mistral-common versions
# that already support gemma4 and are compatible with TRT-LLM.
PIP_UPGRADE_PACKAGES="openai"

############################################
# ========== ARGUMENT PARSING =============
############################################

usage() {
  cat <<EOF
Usage: $0 <GPU_IDS> [--port N] [--model HF_ID] [--max-len N] [--max-batch N] [--name INSTANCE]

Note: trtllm-serve has no --dtype or --quantization flags. Quantization is
      baked into the HF checkpoint or TensorRT engine, not set at serve time.
      Backend is always pytorch (Gemma 4 is multimodal).

Examples:
  $0 0
  $0 0 --port 9014 --model google/gemma-4-9b-it
  $0 0,1 --port 9013 --max-len 16384
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 1
fi

GPU_IDS="$1"
shift

if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Error: GPU_IDS must be comma-separated integers (e.g., 0 or 0,1,2)"
  exit 1
fi

INSTANCE_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)      PORT="$2"; shift 2 ;;
    --model)     MODEL="$2"; shift 2 ;;
    --max-len)   MAX_SEQ_LEN="$2"; shift 2 ;;
    --max-batch) MAX_BATCH_SIZE="$2"; shift 2 ;;
    --name)      INSTANCE_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

GPU_COUNT=$(echo "$GPU_IDS" | awk -F',' '{print NF}')
if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
  TENSOR_PARALLEL_SIZE="$GPU_COUNT"
fi

############################################
# ========== HELPERS =======================
############################################

slugify() {
  local s="$1"
  s="$(echo "$s" | tr '[:upper:]' '[:lower:]')"
  s="$(echo "$s" | sed -E 's/[^a-z0-9_.-]+/-/g; s/^-+//; s/-+$//')"
  if [ "${#s}" -gt 63 ]; then
    s="${s:0:63}"
    s="$(echo "$s" | sed -E 's/-+$//')"
  fi
  echo "$s"
}

############################################
# ========== SERVED MODEL NAME ============
############################################

MODEL_BASENAME="$(basename "$MODEL")"
MODEL_NORMALIZED="$(echo "$MODEL_BASENAME" | tr '[:upper:]' '[:lower:]')"

# TRT-LLM has no dtype flag, so served name is just the model basename.
SERVED_MODEL_NAME="${MODEL_NORMALIZED}"

SERVED_SLUG="$(slugify "$SERVED_MODEL_NAME")"
if [ -n "$INSTANCE_NAME" ]; then
  INSTANCE_SLUG="$(slugify "$INSTANCE_NAME")"
  INSTANCE_ID="${INSTANCE_SLUG}-${PORT}"
else
  INSTANCE_ID="${SERVED_SLUG}-${PORT}"
fi

CONTAINER_NAME="trtllm-${INSTANCE_ID}"

############################################
# ========== PRE-FLIGHT ====================
############################################

mkdir -p "$HF_CACHE"

if command -v ss >/dev/null 2>&1; then
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "(:|\\])${PORT}\$"; then
    echo "WARNING: Something appears to be listening on TCP port ${PORT} already."
    echo "         If this instance fails to start, choose a different --port."
  fi
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found. Install Docker and ensure the daemon is running."
  exit 1
fi

############################################
# ========== START SERVER =================
############################################

echo "Starting TRT-LLM server for Gemma 4"
echo "  Instance ID:          $INSTANCE_ID"
echo "  Container:            $CONTAINER_NAME"
echo "  Model:                $MODEL"
echo "  Served model name:    $SERVED_MODEL_NAME"
echo "  Port:                 $PORT"
echo "  GPUs (restricted to): $GPU_IDS"
echo "  Tensor parallel:      $TENSOR_PARALLEL_SIZE"
echo "  Max seq len:          $MAX_SEQ_LEN"
echo "  Max batch size:       ${MAX_BATCH_SIZE:-<model default>}"
echo "  KV cache GPU mem:     $KV_CACHE_FREE_GPU_MEM_FRACTION"
echo "  Backend:              pytorch (Gemma 4 multimodal — always pytorch)"
echo "  Reasoning parser:     <none>"
echo "  Image:                $TRTLLM_IMAGE"
echo ""

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Removing existing container: $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME"
fi

# Always use pytorch backend: Gemma 4 is multimodal.
# Always disable KV cache block reuse: required for multimodal serving.
EXTRA_CONFIG_FILE="$(mktemp /tmp/trtllm-config.XXXXXX.yaml)"
CONTAINER_CONFIG_FILE="/tmp/$(basename "$EXTRA_CONFIG_FILE")"
cat > "$EXTRA_CONFIG_FILE" <<EOF
kv_cache_config:
  enable_block_reuse: false
EOF

TRTLLM_ARGS=(
  "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --backend pytorch
  --tp_size "$TENSOR_PARALLEL_SIZE"
  --kv_cache_free_gpu_memory_fraction "$KV_CACHE_FREE_GPU_MEM_FRACTION"
  --max_seq_len "$MAX_SEQ_LEN"
  --max_num_tokens "$MAX_SEQ_LEN"
  --trust_remote_code
  --config "$CONTAINER_CONFIG_FILE"
)

if [ -n "$MAX_BATCH_SIZE" ]; then
  TRTLLM_ARGS+=( --max_batch_size "$MAX_BATCH_SIZE" )
fi

DOCKER_FLAGS=(
  -d
  --name "$CONTAINER_NAME"
  -p "${PORT}:${PORT}"
  --ipc=host
  --shm-size=8g
  --gpus all
  -e CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  -e HF_HOME="/root/.cache/huggingface"
  -v "${HF_CACHE}:/root/.cache/huggingface"
  -v "${EXTRA_CONFIG_FILE}:${CONTAINER_CONFIG_FILE}:ro"
  --label "app=trtllm"
  --label "trtllm.instance=${INSTANCE_ID}"
  --label "trtllm.model=${MODEL}"
  --label "trtllm.served_model_name=${SERVED_MODEL_NAME}"
)
if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
  DOCKER_FLAGS+=( -e HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" )
fi

if [ -n "$PIP_UPGRADE_PACKAGES" ]; then
  # $PIP_UPGRADE_PACKAGES is intentionally unquoted so bash word-splits it into
  # separate package names for pip (quoting it would pass the whole string as one
  # invalid package name).
  TRTLLM_CMD="pip install -q --upgrade $PIP_UPGRADE_PACKAGES && exec trtllm-serve$(printf ' %q' "${TRTLLM_ARGS[@]}")"
  docker run "${DOCKER_FLAGS[@]}" \
    --entrypoint bash \
    "$TRTLLM_IMAGE" \
    -c "$TRTLLM_CMD"
else
  docker run "${DOCKER_FLAGS[@]}" \
    --entrypoint trtllm-serve \
    "$TRTLLM_IMAGE" \
    "${TRTLLM_ARGS[@]}"
fi

echo ""
echo "  TRT-LLM server started (Gemma 4)"
echo "  Endpoint: http://localhost:${PORT}/v1"
echo "  Model ID (use this in client): ${SERVED_MODEL_NAME}"
echo ""
echo "Logs:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo ""
echo "List running TRT-LLM instances:"
echo "  docker ps --filter label=app=trtllm --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
