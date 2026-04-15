#!/usr/bin/env bash
set -euo pipefail
# Start a TRT-LLM server (OpenAI-compatible) via trtllm-serve.
# Supports both text-only LLMs and vision-language models (VLMs).
# Uses Docker with --gpus for GPU access.
#
# Mirrors start_vllm_server.sh but targets NVIDIA TensorRT-LLM instead of vLLM.

############################################
# ========== EDIT THESE DEFAULTS ===========
############################################

# ---- Networking ----
# PORT=9009
# PORT=9010
PORT=9015
# PORT=9008
HOST="0.0.0.0"

# ---- Model selection (HF repo or local path) ----
# MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"
# MODEL="Qwen/Qwen3-VL-8B-Instruct"
# MODEL="Qwen/Qwen3-VL-8B-Thinking"
# MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"
# MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
# MODEL="Qwen/Qwen3-VL-30B-A3B-Thinking"
# MODEL="Qwen/Qwen3.5-122B-A10B"

# ---- Engine/runtime params (fixed at server start) ----
# NOTE: trtllm-serve has no --dtype or --quantization CLI flags.
# Quantization is baked in at engine-build time (or via HF checkpoint format).
TENSOR_PARALLEL_SIZE=""           # leave empty to auto = number of GPUs passed
KV_CACHE_FREE_GPU_MEM_FRACTION="0.90"   # analogous to vLLM's --gpu-memory-utilization
MAX_SEQ_LEN="32768"
MAX_BATCH_SIZE=""                 # leave empty to use model default

# ---- Container image ----
# Pin to a specific NGC tag in production; 'latest' is fine for experimentation.
# Check: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags
#   format: YY.MM-trtllm-python-py3
# 25.04 does not support Qwen3MoeForCausalLM (model predates that release).
# 25.07+ adds Qwen3-MoE support. Bump this tag if you hit unknown-architecture errors.
TRTLLM_IMAGE="nvcr.io/nvidia/tritonserver:25.07-trtllm-python-py3"

# HF cache (host)
HF_CACHE="${HOME}/.cache/huggingface"

# Optional: HF token if pulling gated models
HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

# pip packages to upgrade at container startup (space-separated, version specs OK).
# Leave empty to skip (recommended for 25.07+ which ships transformers 4.51.3
# with qwen3_moe support already). Only needed for 25.04 (transformers 4.48.3),
# but upgrading transformers in 25.04 breaks modelopt's Conv1D import, so
# the real fix for qwen3_moe on 25.04 is simply to use 25.07+ instead.
PIP_UPGRADE_PACKAGES=""

############################################
# ========== ARGUMENT PARSING =============
############################################

usage() {
  cat <<EOF
Usage: $0 <GPU_IDS> [--port N] [--model HF_ID] [--max-len N] [--max-batch N] [--name INSTANCE]

Note: trtllm-serve has no --dtype or --quantization flags. Quantization is
      baked into the HF checkpoint or TensorRT engine, not set at serve time.
      VLMs are auto-detected and served with --backend pytorch.

Examples:
  $0 0 --port 9001 --model Qwen/Qwen3-30B-A3B-Instruct-2507
  $0 1 --port 9002 --model meta-llama/Llama-3.1-8B-Instruct
  $0 0,1 --port 9003 --model Qwen/Qwen3-30B-A3B-Thinking-2507 --name thinking
  $0 0 --port 9010 --model Qwen/Qwen2-VL-7B-Instruct
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 1
fi

GPU_IDS="$1"
shift

# Validate GPU_IDS format
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Error: GPU_IDS must be comma-separated integers (e.g., 0 or 0,1,2)"
  exit 1
fi

INSTANCE_NAME=""   # optional user-supplied name to disambiguate multiple runs

# Optional overrides
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

# Compute TP size from GPU list if not provided
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

# trtllm-serve has no --dtype or --quantization args, so the served name is
# just the model basename (quantization lives in the checkpoint, not the CLI).
SERVED_MODEL_NAME="${MODEL_NORMALIZED}"

# Default instance name: served model name (sanitized) + port to avoid collisions
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

# Quick port sanity check: only warns (doesn't fail) if something is already listening
if command -v ss >/dev/null 2>&1; then
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "(:|\\])${PORT}\$"; then
    echo "WARNING: Something appears to be listening on TCP port ${PORT} already."
    echo "         If this instance fails to start, choose a different --port."
  fi
fi

# Check docker is available
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found. Install Docker and ensure the daemon is running."
  exit 1
fi

############################################
# ========== START SERVER =================
############################################

# Auto-select backend:
#   - VLMs require pytorch (C++ backend has no multimodal support)
#   - MoE models (a3b, a10b, a22b, moe in name) require pytorch; the C++
#     backend uses AutoConfig.from_pretrained which fails on architectures
#     that post-date the container's bundled transformers (e.g. qwen3_moe)
#   - Dense text-only models default to the faster C++ backend
# Keep a separate display string since BACKEND="" means "don't pass the flag".
MODEL_LC="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')"
if [[ "$MODEL_LC" == *"-vl"* || "$MODEL_LC" == *"vision"* || "$MODEL_LC" == *"vlm"* ]]; then
  BACKEND="pytorch"
  BACKEND_DISPLAY="pytorch (VLM auto-detected)"
elif [[ "$MODEL_LC" =~ -a[0-9]+b || "$MODEL_LC" == *"moe"* ]]; then
  BACKEND="pytorch"
  BACKEND_DISPLAY="pytorch (MoE auto-detected)"
else
  BACKEND=""
  BACKEND_DISPLAY="<default (cpp)>"
fi

echo "Starting TRT-LLM server container"
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
echo "  Backend:              $BACKEND_DISPLAY"
echo "  Image:                $TRTLLM_IMAGE"
echo ""

# Remove old container of same name (only this instance)
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Removing existing container: $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME"
fi

# Build trtllm-serve args.
# trtllm-serve takes the model as a positional argument, then flags.
# Valid flags (25.04): --host, --port, --backend, --tp_size, --pp_size,
#   --max_seq_len, --max_batch_size, --max_num_tokens, --max_beam_width,
#   --kv_cache_free_gpu_memory_fraction, --trust_remote_code
# NOTE: No --dtype, --quantization, or --served-model-name in this version.
TRTLLM_ARGS=(
  "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --tp_size "$TENSOR_PARALLEL_SIZE"
  --kv_cache_free_gpu_memory_fraction "$KV_CACHE_FREE_GPU_MEM_FRACTION"
  --max_seq_len "$MAX_SEQ_LEN"
  --trust_remote_code
)

if [ -n "$BACKEND" ]; then
  TRTLLM_ARGS+=( --backend "$BACKEND" )
fi

if [ -n "$MAX_BATCH_SIZE" ]; then
  TRTLLM_ARGS+=( --max_batch_size "$MAX_BATCH_SIZE" )
fi

# Docker GPU flag: restrict container to the requested GPU IDs
# Formats accepted by --gpus: "device=0", "device=0,1", etc.
GPU_DEVICE_FLAG="device=${GPU_IDS}"

# Common docker flags (same regardless of entrypoint strategy)
# We use -p (bridge mode) rather than --network host because this environment
# runs inside a container (devcontainer/cluster), where --network host binds
# to an outer network namespace unreachable from the shell. -p publishes the
# port through Docker's proxy so it's always reachable at localhost:PORT.
DOCKER_FLAGS=(
  -d
  --name "$CONTAINER_NAME"
  -p "${PORT}:${PORT}"
  --ipc=host
  --shm-size=8g
  --gpus "$GPU_DEVICE_FLAG"
  -e HF_HOME="/root/.cache/huggingface"
  -v "${HF_CACHE}:/root/.cache/huggingface"
  --label "app=trtllm"
  --label "trtllm.instance=${INSTANCE_ID}"
  --label "trtllm.model=${MODEL}"
  --label "trtllm.served_model_name=${SERVED_MODEL_NAME}"
)
if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
  DOCKER_FLAGS+=( -e HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN" )
fi

# Start container
if [ -n "$PIP_UPGRADE_PACKAGES" ]; then
  # Wrap in bash -c so we can pip-upgrade before launching trtllm-serve.
  # printf '%q' shell-escapes each arg so they survive the bash -c boundary.
  # printf '%q' shell-escapes the package spec (handles <, > in version ranges)
  # and each trtllm-serve arg so they survive the bash -c boundary.
  TRTLLM_CMD="pip install -q --upgrade $(printf '%q' "$PIP_UPGRADE_PACKAGES") && exec trtllm-serve$(printf ' %q' "${TRTLLM_ARGS[@]}")"
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
echo "  TRT-LLM server started"
echo "  Endpoint: http://localhost:${PORT}/v1"
echo "  Model ID (use this in client): ${SERVED_MODEL_NAME}"
echo ""
echo "Logs:"
echo "  docker logs -f ${CONTAINER_NAME}"
echo ""
echo "List running TRT-LLM instances:"
echo "  docker ps --filter label=app=trtllm --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
