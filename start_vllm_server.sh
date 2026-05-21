#!/usr/bin/env bash
set -euo pipefail
# Start a vLLM server (OpenAI-compatible). Supports both text-only LLMs and
# vision-language models (VLMs); use --model with a VLM (e.g. Qwen/Qwen2-VL-7B-Instruct)
# to serve image+text requests.

############################################
# ========== EDIT THESE DEFAULTS ===========
############################################

# ---- Networking ----
# PORT=9009
# PORT=9010
# PORT=9016
PORT=9017
# PORT=9008
HOST="0.0.0.0"

# ---- Model selection (HF repo or local path) ----
# MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
# MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507-FP8"
# MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"
# MODEL="Qwen/Qwen3-VL-8B-Instruct"
MODEL="Qwen/Qwen3-VL-8B-Thinking"
# MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"
# MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
# MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
# MODEL="Qwen/Qwen3-VL-30B-A3B-Thinking"
# MODEL="Qwen/Qwen3.5-122B-A10B"

# MODEL="Qwen/Qwen3-Next-80B-A3B-Thinking"
# MODEL="Qwen/Qwen3-Next-80B-A3B-Thinking-FP8"


# ---- Engine/runtime params (fixed at server start) ----
DTYPE="bfloat16"                  # float16 | bfloat16 | float32
# Use FP8
# DTYPE="torch.float8"
# DTYPE="auto"
QUANTIZATION=""                   # e.g. "awq" or "gptq" or "" for none
TENSOR_PARALLEL_SIZE=""           # leave empty to auto = number of GPUs passed
GPU_MEMORY_UTILIZATION="0.90"
MAX_MODEL_LEN="32768"
# MAX_MODEL_LEN="16384"
# MAX_MODEL_LEN="4096"

# ---- Container image ----
VLLM_IMAGE="docker.io/vllm/vllm-openai:latest"   # strongly recommended: pin to a tag you trust

# HF cache (host)
HF_CACHE="${HOME}/.cache/huggingface"

# Optional: HF token if pulling gated models
HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

############################################
# ========== ARGUMENT PARSING =============
############################################

usage() {
  cat <<EOF
Usage: $0 <GPU_IDS> [--port N] [--model HF_ID] [--dtype DTYPE] [--quant Q] [--max-len N] [--name INSTANCE]

Examples:
  $0 0 --port 9001 --model Qwen/Qwen3-30B-A3B-Instruct-2507
  $0 1 --port 9002 --model meta-llama/Llama-3.1-8B-Instruct --dtype float16
  $0 0,1 --port 9003 --model Qwen/Qwen3-30B-A3B-Thinking-2507 --quant awq --name thinking
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
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --quant) QUANTIZATION="$2"; shift 2 ;;
    --max-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --name) INSTANCE_NAME="$2"; shift 2 ;;
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

# slugify to container-safe name:
# - lowercase
# - replace non [a-z0-9_.-] with -
# - trim leading/trailing separators
# - limit length (Podman names can be longer, but labels/interop often assume <=63)
slugify() {
  local s="$1"
  s="$(echo "$s" | tr '[:upper:]' '[:lower:]')"
  s="$(echo "$s" | sed -E 's/[^a-z0-9_.-]+/-/g; s/^-+//; s/-+$//')"
  # hard cap to 63 chars
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

SERVED_MODEL_NAME="${MODEL_NORMALIZED}__${DTYPE}"
if [ -n "$QUANTIZATION" ]; then
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME}__${QUANTIZATION}"
fi

# Default instance name: served model name (sanitized) + port to avoid collisions
SERVED_SLUG="$(slugify "$SERVED_MODEL_NAME")"
if [ -n "$INSTANCE_NAME" ]; then
  INSTANCE_SLUG="$(slugify "$INSTANCE_NAME")"
  INSTANCE_ID="${INSTANCE_SLUG}-${PORT}"
else
  INSTANCE_ID="${SERVED_SLUG}-${PORT}"
fi

# Unique pod/container names per instance
POD_NAME="llm-pod-${INSTANCE_ID}"
CONTAINER_NAME="vllm-${INSTANCE_ID}"

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

############################################
# ========== START SERVER =================
############################################

echo "🚀 Starting vLLM server container"
echo "  Instance ID:         $INSTANCE_ID"
echo "  Pod:                 $POD_NAME"
echo "  Container:           $CONTAINER_NAME"
echo "  Model:               $MODEL"
echo "  Served model name:   $SERVED_MODEL_NAME"
echo "  Port:                $PORT"
echo "  GPUs (restricted to): $GPU_IDS"
echo "  Tensor parallel:     $TENSOR_PARALLEL_SIZE"
echo "  Dtype:               $DTYPE"
echo "  Quantization:        ${QUANTIZATION:-<none>}"
echo "  Max model len:       $MAX_MODEL_LEN"
echo "  GPU mem util:        $GPU_MEMORY_UTILIZATION"
echo "  Image:               $VLLM_IMAGE"
echo ""

# # Create pod if needed (port mapping host->pod)
# if ! podman pod exists "$POD_NAME"; then
#   echo "Creating pod: $POD_NAME (host port ${PORT} -> pod port ${PORT})"
#   podman pod create --name "$POD_NAME" -p "${PORT}:${PORT}"
# else
#   echo "Pod $POD_NAME already exists."
# fi

# Remove old container of same name (only this instance)
if podman container exists "$CONTAINER_NAME"; then
  echo "Removing existing container: $CONTAINER_NAME"
  podman rm -f "$CONTAINER_NAME"
fi

MODEL_LC="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')"
REASONING_PARSER=""

if [[ "$MODEL_LC" == *"qwen3"* || "$MODEL_LC" == *"qwen3.5"* ]]; then
  if [[ "$MODEL_LC" == *"thinking"* ]]; then
    # REASONING_PARSER="qwen3"
    REASONING_PARSER="deepseek_r1"
  fi
fi

# Build vLLM args
VLLM_ARGS=(
  serve
  --model "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --trust-remote-code
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype "$DTYPE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
)

if [ -n "$REASONING_PARSER" ]; then
  VLLM_ARGS+=( --reasoning-parser "$REASONING_PARSER" )
fi

if [ -n "$QUANTIZATION" ]; then
  VLLM_ARGS+=( --quantization "$QUANTIZATION" )
fi

# GPU restriction via NVIDIA OCI hook (runtime-level restriction)
GPU_FLAGS=()
if [ -d /usr/share/containers/oci/hooks.d ] && ls /usr/share/containers/oci/hooks.d/*nvidia* >/dev/null 2>&1; then
  GPU_FLAGS+=( --hooks-dir=/usr/share/containers/oci/hooks.d )
  GPU_FLAGS+=( -e NVIDIA_VISIBLE_DEVICES="${GPU_IDS}" )
  GPU_FLAGS+=( -e NVIDIA_DRIVER_CAPABILITIES=compute,utility )
else
  echo "ERROR: NVIDIA Podman hook not found at /usr/share/containers/oci/hooks.d."
  echo "Install/configure nvidia-container-toolkit for Podman, or configure CDI, then retry."
  exit 1
fi

# Start container
podman run -d \
  --name "$CONTAINER_NAME" \
  --network host \
  --ipc=host \
  --security-opt=label=disable \
  "${GPU_FLAGS[@]}" \
  -e HF_HOME="/root/.cache/huggingface" \
  -v "${HF_CACHE}:/root/.cache/huggingface:Z" \
  ${HUGGING_FACE_HUB_TOKEN:+-e HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN"} \
  --label "app=vllm" \
  --label "vllm.instance=${INSTANCE_ID}" \
  --label "vllm.model=${MODEL}" \
  --label "vllm.served_model_name=${SERVED_MODEL_NAME}" \
  --entrypoint vllm \
  "$VLLM_IMAGE" \
  "${VLLM_ARGS[@]}"

  # --pod "$POD_NAME" \

echo ""
echo "  vLLM server started"
echo "  Endpoint: http://localhost:${PORT}/v1"
echo "  Model ID (use this in client): ${SERVED_MODEL_NAME}"
echo ""
echo "Logs:"
echo "  podman logs -f ${CONTAINER_NAME}"
echo ""
echo "List running vLLM instances:"
echo "  podman ps --filter label=app=vllm --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Labels}}'"



# #!/usr/bin/env bash
# set -euo pipefail

# ############################################
# # ========== EDIT THESE DEFAULTS ===========
# ############################################

# # ---- Networking ----
# PORT=9001
# HOST="0.0.0.0"

# # ---- Model selection (HF repo or local path) ----
# MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

# # ---- Engine/runtime params (fixed at server start) ----
# DTYPE="bfloat16"                  # float16 | bfloat16 | float32
# QUANTIZATION=""                   # e.g. "awq" or "gptq" or "" for none
# TENSOR_PARALLEL_SIZE=""           # leave empty to auto = number of GPUs passed
# GPU_MEMORY_UTILIZATION="0.90"
# MAX_MODEL_LEN="32768"

# # ---- Container/pod settings ----
# POD_NAME="llm-pod"
# CONTAINER_NAME="vllm-server"
# VLLM_IMAGE="docker.io/vllm/vllm-openai:latest"   # strongly recommended: pin to a tag you trust

# # HF cache (host)
# HF_CACHE="${HOME}/.cache/huggingface"

# # Optional: HF token if pulling gated models
# HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

# ############################################
# # ========== ARGUMENT PARSING =============
# ############################################

# usage() {
#   echo "Usage: $0 <GPU_IDS> [--port N] [--model HF_ID] [--dtype DTYPE] [--quant Q] [--max-len N]"
#   echo "Examples:"
#   echo "  $0 0"
#   echo "  $0 0,1 --port 9002"
#   echo "  $0 2,3 --model Qwen/Qwen3-30B-A3B-Thinking-2507 --dtype float16"
# }

# if [ "$#" -lt 1 ]; then
#   usage
#   exit 1
# fi

# GPU_IDS="$1"
# shift

# # Validate GPU_IDS format
# if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
#   echo "Error: GPU_IDS must be comma-separated integers (e.g., 0 or 0,1,2)"
#   exit 1
# fi

# # Optional overrides
# while [[ $# -gt 0 ]]; do
#   case "$1" in
#     --port) PORT="$2"; shift 2 ;;
#     --model) MODEL="$2"; shift 2 ;;
#     --dtype) DTYPE="$2"; shift 2 ;;
#     --quant) QUANTIZATION="$2"; shift 2 ;;
#     --max-len) MAX_MODEL_LEN="$2"; shift 2 ;;
#     -h|--help) usage; exit 0 ;;
#     *) echo "Unknown argument: $1"; usage; exit 1 ;;
#   esac
# done

# # Compute TP size from GPU list if not provided
# GPU_COUNT=$(echo "$GPU_IDS" | awk -F',' '{print NF}')
# if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
#   TENSOR_PARALLEL_SIZE="$GPU_COUNT"
# fi

# ############################################
# # ========== SERVED MODEL NAME ============
# ############################################

# MODEL_BASENAME="$(basename "$MODEL")"
# MODEL_NORMALIZED="$(echo "$MODEL_BASENAME" | tr '[:upper:]' '[:lower:]')"

# SERVED_MODEL_NAME="${MODEL_NORMALIZED}__${DTYPE}"
# if [ -n "$QUANTIZATION" ]; then
#   SERVED_MODEL_NAME="${SERVED_MODEL_NAME}__${QUANTIZATION}"
# fi

# ############################################
# # ========== PRE-FLIGHT ====================
# ############################################

# # Ensure HF cache dir exists on host for bind mount
# mkdir -p "$HF_CACHE"

# ############################################
# # ========== START SERVER =================
# ############################################

# echo "🚀 Starting vLLM server container"
# echo "  Model:               $MODEL"
# echo "  Served model name:   $SERVED_MODEL_NAME"
# echo "  Port:                $PORT"
# echo "  GPUs (restricted to): $GPU_IDS"
# echo "  Tensor parallel:     $TENSOR_PARALLEL_SIZE"
# echo "  Dtype:               $DTYPE"
# echo "  Quantization:        ${QUANTIZATION:-<none>}"
# echo "  Max model len:       $MAX_MODEL_LEN"
# echo "  GPU mem util:        $GPU_MEMORY_UTILIZATION"
# echo "  Image:               $VLLM_IMAGE"
# echo ""

# # Create pod if needed (port mapping host->pod)
# if ! podman pod exists "$POD_NAME"; then
#   echo "Creating pod: $POD_NAME (host port ${PORT} -> pod port ${PORT})"
#   podman pod create --name "$POD_NAME" -p "${PORT}:${PORT}"
# else
#   echo "Pod $POD_NAME already exists."
# fi

# # Remove old container if it exists
# if podman container exists "$CONTAINER_NAME"; then
#   echo "Removing existing container: $CONTAINER_NAME"
#   podman rm -f "$CONTAINER_NAME"
# fi

# # Build vLLM args (newer CLI syntax: --model)
# VLLM_ARGS=(
#   serve
#   --model "$MODEL"
#   --host "$HOST"
#   --port "$PORT"
#   --trust-remote-code
#   --served-model-name "$SERVED_MODEL_NAME"
#   --dtype "$DTYPE"
#   --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
#   --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
#   --max-model-len "$MAX_MODEL_LEN"
# )

# if [ -n "$QUANTIZATION" ]; then
#   VLLM_ARGS+=( --quantization "$QUANTIZATION" )
# fi

# # GPU restriction via NVIDIA OCI hook (runtime-level restriction)
# GPU_FLAGS=()
# if [ -d /usr/share/containers/oci/hooks.d ] && ls /usr/share/containers/oci/hooks.d/*nvidia* >/dev/null 2>&1; then
#   GPU_FLAGS+=( --hooks-dir=/usr/share/containers/oci/hooks.d )
#   # IMPORTANT: restrict devices at runtime so container cannot see others
#   GPU_FLAGS+=( -e NVIDIA_VISIBLE_DEVICES="${GPU_IDS}" )
#   GPU_FLAGS+=( -e NVIDIA_DRIVER_CAPABILITIES=compute,utility )
# else
#   echo "ERROR: NVIDIA Podman hook not found at /usr/share/containers/oci/hooks.d."
#   echo "Install/configure nvidia-container-toolkit for Podman, or configure CDI, then retry."
#   exit 1
# fi

# # Start container
# podman run -d \
#   --name "$CONTAINER_NAME" \
#   --pod "$POD_NAME" \
#   --ipc=host \
#   --security-opt=label=disable \
#   "${GPU_FLAGS[@]}" \
#   -e CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPU_COUNT-1)))" \
#   -e HF_HOME="/root/.cache/huggingface" \
#   -v "${HF_CACHE}:/root/.cache/huggingface:Z" \
#   ${HUGGING_FACE_HUB_TOKEN:+-e HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN"} \
#   --entrypoint vllm \
#   "$VLLM_IMAGE" \
#   "${VLLM_ARGS[@]}"

# echo ""
# echo "vLLM server started"
# echo "  Endpoint: http://localhost:${PORT}/v1"
# echo "  Model ID (use this in client): ${SERVED_MODEL_NAME}"
# echo ""
# echo "Logs:"
# echo "  podman logs -f ${CONTAINER_NAME}"


# # #!/usr/bin/env bash
# # set -euo pipefail

# # ############################################
# # # ========== EDIT THESE DEFAULTS ===========
# # ############################################

# # # ---- Networking ----
# # PORT=9001
# # HOST="0.0.0.0"

# # # ---- Model selection (HF repo or local path) ----
# # MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

# # # ---- Engine/runtime params (fixed at server start) ----
# # DTYPE="bfloat16"                  # float16 | bfloat16 | float32
# # QUANTIZATION=""                   # e.g. "awq" or "gptq" or "" for none
# # TENSOR_PARALLEL_SIZE=""           # leave empty to auto = number of GPUs passed
# # GPU_MEMORY_UTILIZATION="0.90"
# # MAX_MODEL_LEN="32768"

# # # ---- Container/pod settings ----
# # POD_NAME="llm-pod"
# # CONTAINER_NAME="vllm-server"
# # # VLLM_IMAGE="vllm/vllm-openai:latest"   # strongly recommended: pin to a tag you trust
# # VLLM_IMAGE="docker.io/vllm/vllm-openai:latest"


# # # HF cache (host)
# # HF_CACHE="${HOME}/.cache/huggingface"

# # # Optional: HF token if pulling gated models
# # HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

# # ############################################
# # # ========== ARGUMENT PARSING =============
# # ############################################

# # usage() {
# #   echo "Usage: $0 <GPU_IDS> [--port N] [--model HF_ID] [--dtype DTYPE] [--quant Q] [--max-len N]"
# #   echo "Examples:"
# #   echo "  $0 0"
# #   echo "  $0 0,1 --port 9002"
# #   echo "  $0 2,3 --model Qwen/Qwen3-30B-A3B-Thinking-2507 --dtype float16"
# # }

# # if [ "$#" -lt 1 ]; then
# #   usage
# #   exit 1
# # fi

# # GPU_IDS="$1"
# # shift

# # # Validate GPU_IDS format
# # if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
# #   echo "Error: GPU_IDS must be comma-separated integers (e.g., 0 or 0,1,2)"
# #   exit 1
# # fi

# # # Optional overrides
# # while [[ $# -gt 0 ]]; do
# #   case "$1" in
# #     --port) PORT="$2"; shift 2 ;;
# #     --model) MODEL="$2"; shift 2 ;;
# #     --dtype) DTYPE="$2"; shift 2 ;;
# #     --quant) QUANTIZATION="$2"; shift 2 ;;
# #     --max-len) MAX_MODEL_LEN="$2"; shift 2 ;;
# #     -h|--help) usage; exit 0 ;;
# #     *) echo "Unknown argument: $1"; usage; exit 1 ;;
# #   esac
# # done

# # # Compute TP size from GPU list if not provided
# # GPU_COUNT=$(echo "$GPU_IDS" | awk -F',' '{print NF}')
# # if [ -z "$TENSOR_PARALLEL_SIZE" ]; then
# #   TENSOR_PARALLEL_SIZE="$GPU_COUNT"
# # fi

# # ############################################
# # # ========== SERVED MODEL NAME ============
# # ############################################
# # # Served model name should identify *fixed* behavior-altering runtime config:
# # # - model
# # # - dtype
# # # - (optional) quantization
# # #
# # # Everything else (temperature/top_p/max_tokens/etc.) is per-request.

# # MODEL_BASENAME="$(basename "$MODEL")"
# # MODEL_NORMALIZED="$(echo "$MODEL_BASENAME" | tr '[:upper:]' '[:lower:]')"

# # SERVED_MODEL_NAME="${MODEL_NORMALIZED}__${DTYPE}"
# # if [ -n "$QUANTIZATION" ]; then
# #   SERVED_MODEL_NAME="${SERVED_MODEL_NAME}__${QUANTIZATION}"
# # fi

# # ############################################
# # # ========== START SERVER =================
# # ############################################

# # echo "🚀 Starting vLLM server container"
# # echo "  Model:               $MODEL"
# # echo "  Served model name:   $SERVED_MODEL_NAME"
# # echo "  Port:                $PORT"
# # echo "  GPUs (visible):      $GPU_IDS"
# # echo "  Tensor parallel:     $TENSOR_PARALLEL_SIZE"
# # echo "  Dtype:               $DTYPE"
# # echo "  Quantization:        ${QUANTIZATION:-<none>}"
# # echo "  Max model len:       $MAX_MODEL_LEN"
# # echo "  GPU mem util:        $GPU_MEMORY_UTILIZATION"
# # echo "  Image:               $VLLM_IMAGE"
# # echo ""

# # # Create pod if needed (port mapping host->pod)
# # if ! podman pod exists "$POD_NAME"; then
# #   echo "Creating pod: $POD_NAME (host port ${PORT} -> pod port ${PORT})"
# #   podman pod create --name "$POD_NAME" -p "${PORT}:${PORT}"
# # else
# #   # If the pod already exists with different port mapping, you'll need a new pod name
# #   # or recreate the pod. We won't auto-recreate to avoid surprising you.
# #   echo "Pod $POD_NAME already exists."
# # fi

# # # Remove old container if it exists
# # if podman container exists "$CONTAINER_NAME"; then
# #   echo "Removing existing container: $CONTAINER_NAME"
# #   podman rm -f "$CONTAINER_NAME"
# # fi

# # # Build vLLM args
# # # VLLM_ARGS=(
# # #   vllm serve "$MODEL"
# # #   --host "$HOST"
# # #   --port "$PORT"
# # #   --trust-remote-code
# # #   --served-model-name "$SERVED_MODEL_NAME"
# # #   --dtype "$DTYPE"
# # #   --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
# # #   --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
# # #   --max-model-len "$MAX_MODEL_LEN"
# # # )

# # VLLM_ARGS=(
# #   vllm serve
# #   --model "$MODEL"
# #   --host "$HOST"
# #   --port "$PORT"
# #   --trust-remote-code
# #   --served-model-name "$SERVED_MODEL_NAME"
# #   --dtype "$DTYPE"
# #   --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
# #   --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
# #   --max-model-len "$MAX_MODEL_LEN"
# # )


# # if [ -n "$QUANTIZATION" ]; then
# #   VLLM_ARGS+=( --quantization "$QUANTIZATION" )
# # fi

# # # Start container
# # podman run -d \
# #   --name "$CONTAINER_NAME" \
# #   --pod "$POD_NAME" \
# #   --ipc=host \
# #   --security-opt=label=disable \
# #   --hooks-dir=/usr/share/containers/oci/hooks.d \
# #   --env NVIDIA_VISIBLE_DEVICES=all \
# #   --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
# #   -e CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
# #   -e HF_HOME="/root/.cache/huggingface" \
# #   -v "${HF_CACHE}:/root/.cache/huggingface:Z" \
# #   ${HUGGING_FACE_HUB_TOKEN:+-e HUGGING_FACE_HUB_TOKEN="$HUGGING_FACE_HUB_TOKEN"} \
# #   "$VLLM_IMAGE" \
# #   "${VLLM_ARGS[@]}"

# # echo ""
# # echo "  vLLM server started"
# # echo "  Endpoint: http://localhost:${PORT}/v1"
# # echo "  Model ID (use this in client): ${SERVED_MODEL_NAME}"
# # echo ""
# # echo "Logs:"
# # echo "  podman logs -f ${CONTAINER_NAME}"
