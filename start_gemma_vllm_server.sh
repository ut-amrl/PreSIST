#!/usr/bin/env bash
set -euo pipefail
# Start a vLLM server for Gemma 4 (OpenAI-compatible) via Podman or Docker.
# Gemma 4 is natively multimodal; use google/gemma-4-27b-it for the 27B model
# or google/gemma-4-9b-it for the smaller variant.
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

# ---- Engine/runtime params (fixed at server start) ----
DTYPE="bfloat16"                  # float16 | bfloat16 | float32
QUANTIZATION=""                   # e.g. "awq" or "gptq" or "" for none
TENSOR_PARALLEL_SIZE=""           # leave empty to auto = number of GPUs passed
GPU_MEMORY_UTILIZATION="0.90"
MAX_MODEL_LEN="32768"

# ---- Container runtime ----
# "podman" or "docker". Note: the NVIDIA hook flags below are podman-specific;
# for docker you'd typically swap them for `--gpus '"device=$GPU_IDS"'`.
CONTAINER_RUNTIME="docker"

# ---- Container image ----
VLLM_IMAGE="docker.io/vllm/vllm-openai:nightly"

# HF cache (host)
HF_CACHE="${HOME}/.cache/huggingface"

# Optional: HF token (required for gated Gemma 4 weights)
HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

# pip packages to upgrade at container startup (space-separated, NOT quoted as one arg).
# transformers: bundled version doesn't know gemma4 architecture.
# mistral_common: newer transformers requires ReasoningEffort from mistral_common.
PIP_UPGRADE_PACKAGES="transformers mistral_common"

############################################
# ========== ARGUMENT PARSING =============
############################################

usage() {
  cat <<EOF
Usage: $0 <GPU_IDS> [--port N] [--model HF_ID] [--dtype DTYPE] [--quant Q] [--max-len N] [--name INSTANCE] [--no-thinking]

Flags:
  --no-thinking   Skip --reasoning-parser gemma4. By default the server is
                  started with the gemma4 reasoning parser so that clients
                  passing chat_template_kwargs={"enable_thinking": true} get
                  the thought trace split into message.reasoning. Verified
                  empirically that enable_thinking=false requests are
                  unaffected by the parser, so default-on is safe.

Examples:
  $0 0
  $0 0 --port 9014 --model google/gemma-4-9b-it
  $0 0,1 --port 9013 --dtype bfloat16
  $0 0 --no-thinking
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
ENABLE_THINKING=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)  PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --quant) QUANTIZATION="$2"; shift 2 ;;
    --max-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --name)  INSTANCE_NAME="$2"; shift 2 ;;
    --no-thinking) ENABLE_THINKING=0; shift 1 ;;
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

SERVED_MODEL_NAME="${MODEL_NORMALIZED}__${DTYPE}"
if [ -n "$QUANTIZATION" ]; then
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME}__${QUANTIZATION}"
fi

SERVED_SLUG="$(slugify "$SERVED_MODEL_NAME")"
if [ -n "$INSTANCE_NAME" ]; then
  INSTANCE_SLUG="$(slugify "$INSTANCE_NAME")"
  INSTANCE_ID="${INSTANCE_SLUG}-${PORT}"
else
  INSTANCE_ID="${SERVED_SLUG}-${PORT}"
fi

CONTAINER_NAME="vllm-${INSTANCE_ID}"

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

############################################
# ========== START SERVER =================
############################################

echo "Starting vLLM server for Gemma 4"
echo "  Instance ID:         $INSTANCE_ID"
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
if [ "$ENABLE_THINKING" -eq 1 ]; then
  echo "  Thinking:            enabled (--reasoning-parser gemma4)"
else
  echo "  Thinking:            disabled"
fi
echo ""

if "$CONTAINER_RUNTIME" container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Removing existing container: $CONTAINER_NAME"
  "$CONTAINER_RUNTIME" rm -f "$CONTAINER_NAME"
fi

# Gemma 4 is instruct-only — no reasoning parser needed.
# Model is passed as a positional arg (not --model) per vLLM v0.13+ convention.
VLLM_ARGS=(
  serve
  "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --trust-remote-code
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype "$DTYPE"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
)

if [ -n "$QUANTIZATION" ]; then
  VLLM_ARGS+=( --quantization "$QUANTIZATION" )
fi

if [ "$ENABLE_THINKING" -eq 1 ]; then
  VLLM_ARGS+=( --reasoning-parser gemma4 )
fi

GPU_FLAGS=()
case "$CONTAINER_RUNTIME" in
  podman)
    if [ -d /usr/share/containers/oci/hooks.d ] && ls /usr/share/containers/oci/hooks.d/*nvidia* >/dev/null 2>&1; then
      GPU_FLAGS+=( --hooks-dir=/usr/share/containers/oci/hooks.d )
      GPU_FLAGS+=( -e NVIDIA_VISIBLE_DEVICES="${GPU_IDS}" )
      GPU_FLAGS+=( -e NVIDIA_DRIVER_CAPABILITIES=compute,utility )
    else
      echo "ERROR: NVIDIA Podman hook not found at /usr/share/containers/oci/hooks.d."
      echo "Install/configure nvidia-container-toolkit for Podman, or configure CDI, then retry."
      exit 1
    fi
    ;;
  docker)
    # Embedded quotes are required: docker's --gpus parser treats commas as
    # field separators, so `device=0,1` would be misparsed without them.
    GPU_FLAGS+=( --gpus "\"device=${GPU_IDS}\"" )
    ;;
  *)
    echo "ERROR: Unknown CONTAINER_RUNTIME '$CONTAINER_RUNTIME' (expected 'podman' or 'docker')."
    exit 1
    ;;
esac

# $PIP_UPGRADE_PACKAGES is intentionally unquoted so bash word-splits it into
# separate package names for pip (quoting it would pass the whole string as one
# invalid package name).
VLLM_CMD="pip install -q --upgrade $PIP_UPGRADE_PACKAGES && exec vllm$(printf ' %q' "${VLLM_ARGS[@]}")"

"$CONTAINER_RUNTIME" run -d \
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
  --entrypoint bash \
  "$VLLM_IMAGE" \
  -c "$VLLM_CMD"

echo ""
echo "  vLLM server started (Gemma 4)"
echo "  Endpoint: http://localhost:${PORT}/v1"
echo "  Model ID (use this in client): ${SERVED_MODEL_NAME}"
echo ""
echo "Logs:"
echo "  ${CONTAINER_RUNTIME} logs -f ${CONTAINER_NAME}"
echo ""
echo "List running vLLM instances:"
echo "  ${CONTAINER_RUNTIME} ps --filter label=app=vllm --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Labels}}'"
