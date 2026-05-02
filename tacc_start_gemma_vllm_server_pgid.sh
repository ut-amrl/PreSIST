#!/usr/bin/env bash
set -eo pipefail
# Start a vLLM server for Gemma 4 (OpenAI-compatible) on TACC.
#
# Mirrors tacc_start_vllm_server_pgid.sh but adapts the gemma-specific bits
# from start_gemma_vllm_server.sh:
#   - default model is a gated google/gemma-4-* repo (HF token required)
#   - upgrades transformers + mistral_common in the active env before launch
#     (the bundled transformers in vllm envs typically doesn't know the
#     gemma4 architecture, and newer transformers needs ReasoningEffort
#     from mistral_common)
#   - --reasoning-parser gemma4 is enabled by default so that clients
#     passing chat_template_kwargs={"enable_thinking": true} get the
#     thought trace split into message.reasoning_content. Disable with
#     VLLM_ENABLE_THINKING=0.

MODELS_DIR="${MODELS_DIR:-$SCRATCH/models}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$SCRATCH/.cache/huggingface}"

# Can be either:
#   - a local path, e.g. /models/gemma-4-27b-it
#   - a Hugging Face repo id, e.g. google/gemma-4-27b-it
VLLM_MODEL="${VLLM_MODEL:-google/gemma-4-31B-it}"
VLLM_API_PORT="${VLLM_API_PORT:-9008}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_GPUS="${VLLM_GPUS:-0}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_QUANT="${VLLM_QUANT:-}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
HF_TOKEN="${HF_TOKEN:-}"

# 1 = pass --reasoning-parser gemma4; 0 = omit it.
VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING:-1}"


# Default DP RPC port = API port + 50 (unless explicitly set)
VLLM_DP_RPC_PORT="${VLLM_DP_RPC_PORT:-$((VLLM_API_PORT + 50))}"

count_csv_items() {
    local s="${1// /}"   # remove spaces
    if [[ -z "$s" ]]; then
        echo 0
    else
        awk -F',' '{print NF}' <<< "$s"
    fi
}

VLLM_NUM_GPUS="$(count_csv_items "$VLLM_GPUS")"

VLLM_DP="${VLLM_DP:-}"

if [[ -z "$VLLM_DP" ]]; then
    if (( VLLM_NUM_GPUS % VLLM_TP != 0 )); then
        echo "ERROR: VLLM_NUM_GPUS=$VLLM_NUM_GPUS is not divisible by VLLM_TP=$VLLM_TP"
        exit 1
    fi
    VLLM_DP=$(( VLLM_NUM_GPUS / VLLM_TP ))
fi

if (( VLLM_TP * VLLM_DP != VLLM_NUM_GPUS )); then
    echo "ERROR: VLLM_TP=$VLLM_TP × VLLM_DP=$VLLM_DP does not equal number of visible GPUs=$VLLM_NUM_GPUS"
    echo "       VLLM_GPUS=$VLLM_GPUS"
    exit 1
fi

# Space-separated list of pip packages to upgrade in-env before launch.
# Defaults to empty: the vllm-gemma env already pins the versions we need
# (vllm nightly + transformers 5.x + mistral_common), so re-upgrading on
# every launch wastes time and risks pulling in incompatible new releases.
# Set PIP_UPGRADE_PACKAGES="transformers mistral_common" if bootstrapping
# a fresh env that hasn't had those packages upgraded yet.
# Use ${VAR-default} (no colon) so an explicit empty string disables the
# upgrade; ${VAR:-default} would treat empty as "unset" and re-apply the default.
PIP_UPGRADE_PACKAGES="${PIP_UPGRADE_PACKAGES-}"

VLLM_LOG="${VLLM_LOG:-logs/spf/vllm_gemma_${SLURM_JOB_ID:-interactive}.out}"
VLLM_PID_FILE="${VLLM_PID_FILE:-logs/spf/vllm_gemma_${SLURM_JOB_ID:-interactive}.pid}"
VLLM_PGID_FILE="${VLLM_PGID_FILE:-logs/spf/vllm_gemma_${SLURM_JOB_ID:-interactive}.pgid}"

LOG_DIR="${LOG_DIR:-$(dirname "${VLLM_LOG:-$SCRATCH/vllm.out}")}"
mkdir -p "$LOG_DIR"

############################################
# micromamba + vllm env activation
############################################

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$WORK/micromamba}"

if command -v micromamba >/dev/null 2>&1; then
    MAMBA_EXE="$(command -v micromamba)"
elif [ -x "$WORK/.local/bin/micromamba" ]; then
    MAMBA_EXE="$WORK/.local/bin/micromamba"
elif [ -x "$HOME/.local/bin/micromamba" ]; then
    MAMBA_EXE="$HOME/.local/bin/micromamba"
else
    echo "ERROR: micromamba not found. Install it or add it to PATH."
    exit 1
fi

eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"

ENV_NAME="${ENV_NAME:-vllm-gemma}"
if ! micromamba activate "$ENV_NAME"; then
    echo "ERROR: Failed to activate micromamba env '$ENV_NAME'."
    exit 1
fi

############################################
# Pre-flight: HF token + pip upgrade
############################################

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN is empty. Gemma 4 weights are gated on HuggingFace;"
    echo "         pulling '${VLLM_MODEL}' will fail without an authorized token."
fi

if [[ -n "${PIP_UPGRADE_PACKAGES// /}" ]]; then
    echo "Upgrading pip packages in env '${ENV_NAME}': ${PIP_UPGRADE_PACKAGES}"
    # Intentionally unquoted so bash word-splits into separate package names.
    if ! pip install -q --upgrade $PIP_UPGRADE_PACKAGES; then
        echo "ERROR: pip upgrade of '${PIP_UPGRADE_PACKAGES}' failed."
        exit 1
    fi
fi

############################################
# Cleanup trap
############################################

cleanup_on_error() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        if [[ -f "$VLLM_PGID_FILE" ]]; then
            local pgid
            pgid="$(cat "$VLLM_PGID_FILE" 2>/dev/null || true)"
            if [[ -n "${pgid:-}" ]]; then
                kill -TERM -- "-$pgid" 2>/dev/null || true
                sleep 3
                kill -KILL -- "-$pgid" 2>/dev/null || true
            fi
            rm -f "$VLLM_PGID_FILE"
        elif [[ -f "$VLLM_PID_FILE" ]]; then
            local pid
            pid="$(cat "$VLLM_PID_FILE" 2>/dev/null || true)"
            if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
                wait "$pid" 2>/dev/null || true
            fi
        fi
        rm -f "$VLLM_PID_FILE"
    fi
}
trap cleanup_on_error EXIT

############################################
# Derive served model name
############################################

if [[ "$VLLM_MODEL" == /* ]]; then
    MODEL_SOURCE="local path"
else
    MODEL_SOURCE="Hugging Face repo id"
fi

MODEL_BASENAME="$(basename "$VLLM_MODEL")"
MODEL_NORMALIZED="$(echo "$MODEL_BASENAME" | tr '[:upper:]' '[:lower:]')"
VLLM_SERVED_NAME="${MODEL_NORMALIZED}__${VLLM_DTYPE}"
if [[ -n "${VLLM_QUANT:-}" ]]; then
    VLLM_SERVED_NAME="${VLLM_SERVED_NAME}__${VLLM_QUANT}"
fi

############################################
# Reasoning parser
############################################

# Verified empirically (per start_gemma_vllm_server.sh) that
# enable_thinking=false requests are unaffected by the gemma4 parser,
# so default-on is safe.
VLLM_REASONING_PARSER=""
if [[ "$VLLM_ENABLE_THINKING" == "1" ]]; then
    VLLM_REASONING_PARSER="gemma4"
fi

echo "Starting vLLM (Gemma 4) on GPU(s) [${VLLM_GPUS}]"
echo "  model:           $VLLM_MODEL"
echo "  model source:    $MODEL_SOURCE"
echo "  served name:     $VLLM_SERVED_NAME"
echo "  port:            $VLLM_API_PORT"
echo "  dp port:         $VLLM_DP_RPC_PORT"
echo "  tp:              $VLLM_TP"
echo "  dp:              $VLLM_DP"
echo "  dtype:           $VLLM_DTYPE"
echo "  quant:           ${VLLM_QUANT:-<none>}"
echo "  reasoning parser:${VLLM_REASONING_PARSER:-<none>}"
echo "  log:             $VLLM_LOG"
echo "  pid:             $VLLM_PID_FILE"
echo "  pgid:            $VLLM_PGID_FILE"

export HF_HOME="${HF_CACHE_DIR}"
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

VLLM_SERVE_ARGS="$VLLM_MODEL \
  --host 0.0.0.0 \
  --port $VLLM_API_PORT \
  --data-parallel-rpc-port $VLLM_DP_RPC_PORT \
  --tensor-parallel-size $VLLM_TP \
  --data-parallel-size $VLLM_DP \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --dtype $VLLM_DTYPE \
  --max-model-len $MAX_SEQ_LEN \
  --served-model-name $VLLM_SERVED_NAME \
  --trust-remote-code \
  ${VLLM_QUANT:+--quantization $VLLM_QUANT} \
  ${VLLM_REASONING_PARSER:+--reasoning-parser $VLLM_REASONING_PARSER}"

CUDA_VISIBLE_DEVICES="$VLLM_GPUS" \
setsid bash -lc "exec vllm serve $VLLM_SERVE_ARGS" \
  >"$VLLM_LOG" 2>&1 < /dev/null &
VLLM_PID=$!

echo "$VLLM_PID" > "$VLLM_PID_FILE"
echo "$VLLM_PID" > "$VLLM_PGID_FILE"

echo "Waiting for vLLM on port ${VLLM_API_PORT} (up to 20 min for first-run download/compile)..."
READY=0
for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${VLLM_API_PORT}/health" >/dev/null 2>&1; then
        echo "vLLM ready (waited ~$((i * 5))s)"
        READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM exited unexpectedly. Check: $VLLM_LOG"
        exit 1
    fi
    sleep 5
done

if [[ "$READY" -ne 1 ]]; then
    echo "ERROR: vLLM did not become ready in time. Check: $VLLM_LOG"
    exit 1
fi

trap - EXIT

echo "vLLM started successfully."
echo "  PID:        $VLLM_PID"
echo "  PGID:       $VLLM_PID"
echo "  PID file:   $VLLM_PID_FILE"
echo "  PGID file:  $VLLM_PGID_FILE"
echo "  Log:        $VLLM_LOG"
echo "  Served as:  $VLLM_SERVED_NAME"
