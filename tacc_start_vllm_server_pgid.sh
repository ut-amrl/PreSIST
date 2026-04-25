#!/usr/bin/env bash
set -eo pipefail

MODELS_DIR="${MODELS_DIR:-$SCRATCH/models}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$SCRATCH/.cache/huggingface}"

# Can be either:
#   - a local path, e.g. /models/Qwen3-VL-30B-A3B-Instruct
#   - a Hugging Face repo id, e.g. Qwen/Qwen3-VL-30B-A3B-Instruct
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
VLLM_PORT="${VLLM_PORT:-9006}"
VLLM_TP="${VLLM_TP:-4}"
VLLM_GPUS="${VLLM_GPUS:-0,1,2,3}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_QUANT="${VLLM_QUANT:-}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
HF_TOKEN="${HF_TOKEN:-}"

VLLM_LOG="${VLLM_LOG:-logs/spf/vllm_${SLURM_JOB_ID:-interactive}.out}"
VLLM_PID_FILE="${VLLM_PID_FILE:-logs/spf/vllm_${SLURM_JOB_ID:-interactive}.pid}"
VLLM_PGID_FILE="${VLLM_PGID_FILE:-logs/spf/vllm_${SLURM_JOB_ID:-interactive}.pgid}"

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

ENV_NAME="${ENV_NAME:-vllm2}"
if ! micromamba activate "$ENV_NAME"; then
    echo "ERROR: Failed to activate micromamba env '$ENV_NAME'."
    exit 1
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

if [[ -n "${VLLM_REASONING_PARSER+x}" ]]; then
    : # use caller-supplied value (including empty string to disable)
else
    VLLM_REASONING_PARSER=""
    MODEL_LC="$(echo "$VLLM_MODEL" | tr '[:upper:]' '[:lower:]')"
    if [[ "$MODEL_LC" == *"qwen3"* || "$MODEL_LC" == *"qwen3.5"* ]]; then
        if [[ "$MODEL_LC" == *"thinking"* ]]; then
            VLLM_REASONING_PARSER="deepseek_r1"
        fi
    fi
fi

echo "Starting vLLM on GPU(s) [${VLLM_GPUS}]"
echo "  model:           $VLLM_MODEL"
echo "  model source:    $MODEL_SOURCE"
echo "  served name:     $VLLM_SERVED_NAME"
echo "  port:            $VLLM_PORT"
echo "  tp:              $VLLM_TP"
echo "  dtype:           $VLLM_DTYPE"
echo "  quant:           ${VLLM_QUANT:-<none>}"
echo "  reasoning parser:${VLLM_REASONING_PARSER:-<none>}"
echo "  log:             $VLLM_LOG"
echo "  pid:             $VLLM_PID_FILE"
echo "  pgid:            $VLLM_PGID_FILE"

export HF_HOME="${HF_CACHE_DIR}"
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
fi

VLLM_SERVE_ARGS="$VLLM_MODEL \
  --host 0.0.0.0 \
  --port $VLLM_PORT \
  --tensor-parallel-size $VLLM_TP \
  --gpu-memory-utilization 0.90 \
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

echo "Waiting for vLLM on port ${VLLM_PORT} (up to 20 min for first-run download/compile)..."
READY=0
for i in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
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
