#!/usr/bin/env bash
set -eo pipefail

JOB_START=$(date +%s)

echo "=== vLLM server smoke test ==="
echo "  Date:   $(date)"
echo "  Host:   $(hostname)"
echo

############################################
# Modules
############################################

module load tacc-apptainer

############################################
# Paths (override via env)
############################################

export CODE_DIR="${CODE_DIR:-$WORK/workspaces/semantic-persistence-filter}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$SCRATCH/.cache/huggingface}"

export JOB_ID_TAG="${SLURM_JOB_ID:-interactive}"
export JOB_ROOT="${JOB_ROOT:-$SCRATCH/jobs/spf/${JOB_ID_TAG}}"
export LOG_DIR="${LOG_DIR:-$JOB_ROOT/logs}"
export JOB_CACHE_DIR="${JOB_CACHE_DIR:-$JOB_ROOT/cache}"
export JOB_TMP_DIR="${JOB_TMP_DIR:-$JOB_ROOT/tmp}"
export JOB_RUN_DIR="${JOB_RUN_DIR:-$JOB_ROOT/run}"

MODELS_DIR="${MODELS_DIR:-$SCRATCH/models}"

TRTLLM_PORT="${TRTLLM_PORT:-9011}"
TRTLLM_MODEL="${TRTLLM_MODEL:-Qwen/Qwen3-VL-8B-Thinking}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
HF_TOKEN="${HF_TOKEN:-}"

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HF_CACHE_DIR/token}"

if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_TOKEN_FILE" ]; then
    HF_TOKEN="$(<"$HF_TOKEN_FILE")"
    HF_TOKEN="${HF_TOKEN//$'\n'/}"
    HF_TOKEN="${HF_TOKEN//$'\r'/}"
fi

export HF_TOKEN

if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF token loaded from ${HF_TOKEN_FILE}"
else
    echo "WARNING: No HF token found in \$HF_TOKEN or ${HF_TOKEN_FILE}"
fi

TRTLLM_TP="${TRTLLM_TP:-1}"
TOTAL_GPUS="${TOTAL_GPUS:-8}"

VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_QUANT="${VLLM_QUANT:-}"

############################################
# Derive GPU layout from TP
############################################

TRTLLM_GPUS=$(seq -s, 0 $((TRTLLM_TP - 1)))

############################################
# Create directories + set cache env vars
############################################

mkdir -p \
  "${HF_CACHE_DIR}" \
  "${MODELS_DIR}" \
  "${LOG_DIR}" \
  "${JOB_RUN_DIR}" \
  "${JOB_CACHE_DIR}/xdg" \
  "${JOB_CACHE_DIR}/flashinfer" \
  "${JOB_CACHE_DIR}/triton" \
  "${JOB_CACHE_DIR}/tmp" \
  "${JOB_CACHE_DIR}/torch" \
  "${JOB_CACHE_DIR}/cuda" \
  "${JOB_CACHE_DIR}/numba" \
  "${JOB_CACHE_DIR}/matplotlib" \
  "${JOB_CACHE_DIR}/pycache" \
  "${JOB_CACHE_DIR}/apptainer" \
  "${JOB_TMP_DIR}"

export HF_HOME="${HF_CACHE_DIR}"
export XDG_CACHE_HOME="${JOB_CACHE_DIR}/xdg"
export FLASHINFER_CACHE_DIR="${JOB_CACHE_DIR}/flashinfer"
export TRITON_CACHE_DIR="${JOB_CACHE_DIR}/triton"

export TORCH_HOME="${JOB_CACHE_DIR}/torch"
export CUDA_CACHE_PATH="${JOB_CACHE_DIR}/cuda"
export NUMBA_CACHE_DIR="${JOB_CACHE_DIR}/numba"
export MPLCONFIGDIR="${JOB_CACHE_DIR}/matplotlib"
export PYTHONPYCACHEPREFIX="${JOB_CACHE_DIR}/pycache"

export TMPDIR="${JOB_TMP_DIR}"
export TEMP="${JOB_TMP_DIR}"
export TMP="${JOB_TMP_DIR}"
export APPTAINER_CACHEDIR="${JOB_CACHE_DIR}/apptainer"
export APPTAINER_TMPDIR="${JOB_TMP_DIR}"
export SINGULARITY_CACHEDIR="${JOB_CACHE_DIR}/apptainer"
export SINGULARITY_TMPDIR="${JOB_TMP_DIR}"

export APPTAINERENV_HF_HOME="${HF_HOME}"
export APPTAINERENV_XDG_CACHE_HOME="${XDG_CACHE_HOME}"
export APPTAINERENV_FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR}"
export APPTAINERENV_TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
export APPTAINERENV_TORCH_HOME="${TORCH_HOME}"
export APPTAINERENV_CUDA_CACHE_PATH="${CUDA_CACHE_PATH}"
export APPTAINERENV_NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR}"
export APPTAINERENV_MPLCONFIGDIR="${MPLCONFIGDIR}"
export APPTAINERENV_PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}"
export APPTAINERENV_TMPDIR="${TMPDIR}"
export APPTAINERENV_TEMP="${TEMP}"
export APPTAINERENV_TMP="${TMP}"

unset TRANSFORMERS_CACHE

echo "GPU layout (TP=$TRTLLM_TP, $TOTAL_GPUS total GPUs):"
echo "  vLLM:  GPUs $TRTLLM_GPUS  (model: $TRTLLM_MODEL, port: $TRTLLM_PORT)"
echo "  Log:   $LOG_DIR"
echo "  Job root: $JOB_ROOT"
echo

############################################
# Start vLLM server
############################################

VLLM_START_SCRIPT="${VLLM_START_SCRIPT:-$CODE_DIR/tacc_start_vllm_server_pgid.sh}"
if [ ! -x "$VLLM_START_SCRIPT" ]; then
    echo "ERROR: vLLM start script not found or not executable: $VLLM_START_SCRIPT"
    exit 1
fi

TRTLLM_LOG="${LOG_DIR}/vllm_test_${JOB_ID_TAG}.out"
TRTLLM_PID_FILE="${JOB_RUN_DIR}/vllm_test_${JOB_ID_TAG}.pid"
TRTLLM_PGID_FILE="${JOB_RUN_DIR}/vllm_test_${JOB_ID_TAG}.pgid"

############################################
# Cleanup trap
############################################

cleanup() {
    echo "Cleaning up..."
    if [ -f "$TRTLLM_PGID_FILE" ]; then
        TRTLLM_PGID="$(cat "$TRTLLM_PGID_FILE" 2>/dev/null || true)"
        if [ -n "${TRTLLM_PGID:-}" ]; then
            echo "Stopping vLLM process group $TRTLLM_PGID ..."
            kill -TERM -- "-$TRTLLM_PGID" 2>/dev/null || true
            sleep 5
            kill -KILL -- "-$TRTLLM_PGID" 2>/dev/null || true
        fi
        rm -f "$TRTLLM_PGID_FILE"
    elif [ -f "$TRTLLM_PID_FILE" ]; then
        TRTLLM_PID="$(cat "$TRTLLM_PID_FILE" 2>/dev/null || true)"
        if [ -n "${TRTLLM_PID:-}" ] && kill -0 "$TRTLLM_PID" 2>/dev/null; then
            echo "Stopping vLLM server (PID $TRTLLM_PID)..."
            kill "$TRTLLM_PID" 2>/dev/null || true
            wait "$TRTLLM_PID" 2>/dev/null || true
        fi
    fi
    rm -f "$TRTLLM_PID_FILE"
}
trap cleanup EXIT

echo "Starting vLLM server via $VLLM_START_SCRIPT ..."
MODELS_DIR="$MODELS_DIR" \
HF_CACHE_DIR="$HF_CACHE_DIR" \
LOG_DIR="$LOG_DIR" \
VLLM_MODEL="$TRTLLM_MODEL" \
VLLM_PORT="$TRTLLM_PORT" \
VLLM_TP="$TRTLLM_TP" \
VLLM_GPUS="$TRTLLM_GPUS" \
VLLM_DTYPE="$VLLM_DTYPE" \
VLLM_QUANT="$VLLM_QUANT" \
MAX_SEQ_LEN="$MAX_SEQ_LEN" \
HF_TOKEN="$HF_TOKEN" \
VLLM_LOG="$TRTLLM_LOG" \
VLLM_PID_FILE="$TRTLLM_PID_FILE" \
VLLM_PGID_FILE="$TRTLLM_PGID_FILE" \
"$VLLM_START_SCRIPT"

############################################
# Smoke test: /v1/models
############################################

echo
echo "vLLM server is up. Running smoke test against http://localhost:${TRTLLM_PORT}/v1/models ..."

MODELS_RESPONSE="$(curl -sf "http://localhost:${TRTLLM_PORT}/v1/models" 2>&1)"
CURL_EXIT=$?

if [ "$CURL_EXIT" -ne 0 ]; then
    echo "ERROR: /v1/models request failed (curl exit $CURL_EXIT)"
    exit 1
fi

echo "Response from /v1/models:"
echo "$MODELS_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$MODELS_RESPONSE"

echo
echo "Smoke test passed. vLLM is serving models correctly."
echo "End time: $(date)"
ELAPSED=$(( $(date +%s) - JOB_START ))
echo "Total elapsed: ${ELAPSED}s (~$((ELAPSED/60))m)"
