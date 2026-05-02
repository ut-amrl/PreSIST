#!/usr/bin/env bash
set -eo pipefail
# Smoke test for Gemma 4 vLLM server on TACC.
#
# Starts the server via tacc_start_gemma_vllm_server_pgid.sh, then sends a
# chat completion with chat_template_kwargs.enable_thinking=true and prints
# both message.content and message.reasoning_content (the split thought
# trace) so we can confirm the gemma4 reasoning parser is wired up.

JOB_START=$(date +%s)

echo "=== Gemma 4 vLLM server smoke test ==="
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

VLLM_PORT="${VLLM_PORT:-9008}"
VLLM_MODEL="${VLLM_MODEL:-google/gemma-4-31B-it}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_QUANT="${VLLM_QUANT:-}"
VLLM_ENABLE_THINKING="${VLLM_ENABLE_THINKING:-1}"

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
    echo "         Gemma 4 is gated; the server will fail to download weights."
fi

############################################
# Derive GPU layout from TP
############################################

VLLM_GPUS="${VLLM_GPUS:-$(seq -s, 0 $((VLLM_TP - 1)))}"

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

unset TRANSFORMERS_CACHE

echo "GPU layout (TP=$VLLM_TP):"
echo "  vLLM:  GPUs $VLLM_GPUS  (model: $VLLM_MODEL, port: $VLLM_PORT)"
echo "  Log:   $LOG_DIR"
echo "  Job root: $JOB_ROOT"
echo

############################################
# Start vLLM (Gemma 4) server
############################################

VLLM_START_SCRIPT="${VLLM_START_SCRIPT:-$CODE_DIR/tacc_start_gemma_vllm_server_pgid.sh}"
if [ ! -x "$VLLM_START_SCRIPT" ]; then
    echo "ERROR: vLLM start script not found or not executable: $VLLM_START_SCRIPT"
    exit 1
fi

VLLM_LOG="${LOG_DIR}/vllm_gemma_test_${JOB_ID_TAG}.out"
VLLM_PID_FILE="${JOB_RUN_DIR}/vllm_gemma_test_${JOB_ID_TAG}.pid"
VLLM_PGID_FILE="${JOB_RUN_DIR}/vllm_gemma_test_${JOB_ID_TAG}.pgid"

############################################
# Cleanup trap
############################################

cleanup() {
    echo "Cleaning up..."
    if [ -f "$VLLM_PGID_FILE" ]; then
        VLLM_PGID="$(cat "$VLLM_PGID_FILE" 2>/dev/null || true)"
        if [ -n "${VLLM_PGID:-}" ]; then
            echo "Stopping vLLM process group $VLLM_PGID ..."
            kill -TERM -- "-$VLLM_PGID" 2>/dev/null || true
            sleep 5
            kill -KILL -- "-$VLLM_PGID" 2>/dev/null || true
        fi
        rm -f "$VLLM_PGID_FILE"
    elif [ -f "$VLLM_PID_FILE" ]; then
        VLLM_PID_VAL="$(cat "$VLLM_PID_FILE" 2>/dev/null || true)"
        if [ -n "${VLLM_PID_VAL:-}" ] && kill -0 "$VLLM_PID_VAL" 2>/dev/null; then
            echo "Stopping vLLM server (PID $VLLM_PID_VAL)..."
            kill "$VLLM_PID_VAL" 2>/dev/null || true
            wait "$VLLM_PID_VAL" 2>/dev/null || true
        fi
    fi
    rm -f "$VLLM_PID_FILE"
}
trap cleanup EXIT

echo "Starting Gemma 4 vLLM server via $VLLM_START_SCRIPT ..."
MODELS_DIR="$MODELS_DIR" \
HF_CACHE_DIR="$HF_CACHE_DIR" \
LOG_DIR="$LOG_DIR" \
VLLM_MODEL="$VLLM_MODEL" \
VLLM_PORT="$VLLM_PORT" \
VLLM_TP="$VLLM_TP" \
VLLM_GPUS="$VLLM_GPUS" \
VLLM_DTYPE="$VLLM_DTYPE" \
VLLM_QUANT="$VLLM_QUANT" \
VLLM_ENABLE_THINKING="$VLLM_ENABLE_THINKING" \
MAX_SEQ_LEN="$MAX_SEQ_LEN" \
HF_TOKEN="$HF_TOKEN" \
VLLM_LOG="$VLLM_LOG" \
VLLM_PID_FILE="$VLLM_PID_FILE" \
VLLM_PGID_FILE="$VLLM_PGID_FILE" \
"$VLLM_START_SCRIPT"

############################################
# Smoke test 1: /v1/models -> capture served name
############################################

echo
echo "Querying http://localhost:${VLLM_PORT}/v1/models ..."
MODELS_RESPONSE="$(curl -sf "http://localhost:${VLLM_PORT}/v1/models" 2>&1)"
if [ -z "$MODELS_RESPONSE" ]; then
    echo "ERROR: empty response from /v1/models"
    exit 1
fi

SERVED_NAME="$(python3 -c '
import json, sys
data = json.loads(sys.stdin.read())
print(data["data"][0]["id"])
' <<<"$MODELS_RESPONSE")"

echo "Served model id: $SERVED_NAME"

############################################
# Smoke test 2: chat completion w/ thinking
############################################

echo
echo "Sending chat completion (enable_thinking=${VLLM_ENABLE_THINKING}) ..."

REQ_BODY="$(SERVED_NAME="$SERVED_NAME" VLLM_ENABLE_THINKING="$VLLM_ENABLE_THINKING" python3 -c '
import json, os
body = {
    "model": os.environ["SERVED_NAME"],
    "messages": [
        {"role": "user", "content": "A farmer has 17 sheep and all but 9 die. How many are left? Think step by step."}
    ],
    "max_tokens": 512,
    "temperature": 0.0,
}
if os.environ.get("VLLM_ENABLE_THINKING", "1") == "1":
    body["chat_template_kwargs"] = {"enable_thinking": True}
print(json.dumps(body))
')"

CHAT_RESPONSE="$(curl -sf -X POST \
    -H "Content-Type: application/json" \
    -d "$REQ_BODY" \
    "http://localhost:${VLLM_PORT}/v1/chat/completions" 2>&1)" || {
    echo "ERROR: /v1/chat/completions request failed"
    echo "$CHAT_RESPONSE"
    exit 1
}

echo
echo "=== Parsed response ==="
CHAT_RESPONSE="$CHAT_RESPONSE" python3 - <<'PY'
import json, os, sys, textwrap

raw = os.environ["CHAT_RESPONSE"]
try:
    data = json.loads(raw)
except json.JSONDecodeError as e:
    print(f"ERROR: response was not valid JSON: {e}")
    print(raw)
    sys.exit(1)

choice = data["choices"][0]
msg = choice["message"]

content = msg.get("content") or ""
# vLLM reasoning parsers split the thought trace into reasoning_content
# (or, on some versions, "reasoning"). Try both.
reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""

def show(label, text):
    print(f"--- {label} ({len(text)} chars) ---")
    if text:
        print(textwrap.indent(text.strip(), "  "))
    else:
        print("  <empty>")

show("reasoning", reasoning)
show("content", content)

usage = data.get("usage", {})
if usage:
    print(f"\nusage: {usage}")

if not content and not reasoning:
    print("\nERROR: both content and reasoning were empty.")
    sys.exit(1)
PY

echo
echo "Smoke test passed."
echo "End time: $(date)"
ELAPSED=$(( $(date +%s) - JOB_START ))
echo "Total elapsed: ${ELAPSED}s (~$((ELAPSED/60))m)"
