#!/usr/bin/env bash
set -eo pipefail

module load tacc-apptainer

# mkdir -p logs/spf

TRTLLM_IMAGE="${TRTLLM_IMAGE:-$SCRATCH/containers/trtllm-qwen3-vl-8b-thinking-26.03-openai.sif}"
MODELS_DIR="${MODELS_DIR:-$SCRATCH/models}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$SCRATCH/.cache/huggingface}"
JOB_CACHE_DIR="${JOB_CACHE_DIR:-$SCRATCH/.job_cache/${SLURM_JOB_ID:-interactive}}"

# Can be either:
#   - a local path inside the container, e.g. /models/Qwen3-VL-8B-Thinking
#   - a Hugging Face repo id, e.g. Qwen/Qwen3-VL-8B-Thinking
TRTLLM_MODEL="${TRTLLM_MODEL:-Qwen/Qwen3-VL-8B-Thinking}"
TRTLLM_PORT="${TRTLLM_PORT:-9011}"
TRTLLM_TP="${TRTLLM_TP:-1}"
TRTLLM_GPUS="${TRTLLM_GPUS:-0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
HF_TOKEN="${HF_TOKEN:-}"

TRTLLM_LOG="${TRTLLM_LOG:-logs/spf/trtllm_${SLURM_JOB_ID:-interactive}.out}"
TRTLLM_PID_FILE="${TRTLLM_PID_FILE:-logs/spf/trtllm_${SLURM_JOB_ID:-interactive}.pid}"
TRTLLM_PGID_FILE="${TRTLLM_PGID_FILE:-logs/spf/trtllm_${SLURM_JOB_ID:-interactive}.pgid}"

# mkdir -p \
#  "${HF_CACHE_DIR}" \
#  "${MODELS_DIR}" \
#  "${JOB_CACHE_DIR}/xdg" \
#  "${JOB_CACHE_DIR}/flashinfer" \
#  "${JOB_CACHE_DIR}/triton" \
#  "${JOB_CACHE_DIR}/tmp"


# add this near the top
JOB_TMP_DIR="${JOB_TMP_DIR:-${TMPDIR:-$SCRATCH/.job_tmp/${SLURM_JOB_ID:-interactive}}}"
LOG_DIR="${LOG_DIR:-$(dirname "${TRTLLM_LOG:-$SCRATCH/trtllm.out}")}"

mkdir -p \
  "$LOG_DIR" \
  "$JOB_TMP_DIR" \
  "${HF_CACHE_DIR}" \
  "${MODELS_DIR}" \
  "${JOB_CACHE_DIR}/xdg" \
  "${JOB_CACHE_DIR}/flashinfer" \
  "${JOB_CACHE_DIR}/triton" \
  "${JOB_CACHE_DIR}/tmp"

TRTLLM_CFG_FILE="${JOB_TMP_DIR}/trtllm-kvcache-$$.yaml"


#TRTLLM_CFG_FILE="/tmp/trtllm-kvcache-$$.yaml"
cat > "$TRTLLM_CFG_FILE" <<'EOF'
kv_cache_config:
  enable_block_reuse: false
EOF

cleanup_on_error() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        if [[ -f "$TRTLLM_PGID_FILE" ]]; then
            local pgid
            pgid="$(cat "$TRTLLM_PGID_FILE" 2>/dev/null || true)"
            if [[ -n "${pgid:-}" ]]; then
                kill -TERM -- "-$pgid" 2>/dev/null || true
                sleep 3
                kill -KILL -- "-$pgid" 2>/dev/null || true
            fi
            rm -f "$TRTLLM_PGID_FILE"
        elif [[ -f "$TRTLLM_PID_FILE" ]]; then
            local pid
            pid="$(cat "$TRTLLM_PID_FILE" 2>/dev/null || true)"
            if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
                wait "$pid" 2>/dev/null || true
            fi
        fi
        rm -f "$TRTLLM_PID_FILE"
    fi
    rm -f "$TRTLLM_CFG_FILE"
}
trap cleanup_on_error EXIT

if [[ "$TRTLLM_MODEL" == /* ]]; then
    MODEL_SOURCE="local path"
else
    MODEL_SOURCE="Hugging Face repo id"
fi

TRTLLM_SERVE_CMD="unset PMI_FD PMI_RANK PMI_SIZE PMI_JOBID SLURM_STEP_ID SLURM_STEP_NUM_NODES SLURM_PROCID SLURM_NODEID && \
  pip install -q --upgrade openai && \
  exec trtllm-llmapi-launch trtllm-serve ${TRTLLM_MODEL} \
  --host 0.0.0.0 \
  --port ${TRTLLM_PORT} \
  --tp_size ${TRTLLM_TP} \
  --kv_cache_free_gpu_memory_fraction 0.90 \
  --max_seq_len ${MAX_SEQ_LEN} \
  --max_num_tokens ${MAX_SEQ_LEN} \
  --backend pytorch \
  --reasoning_parser deepseek-r1 \
  --config /tmp/trtllm-kvcache.yaml \
  --trust_remote_code"

echo "Starting TRT-LLM on GPU(s) [${TRTLLM_GPUS}]"
echo "  image:        $TRTLLM_IMAGE"
echo "  model:        $TRTLLM_MODEL"
echo "  model source: $MODEL_SOURCE"
echo "  port:         $TRTLLM_PORT"
echo "  tp:           $TRTLLM_TP"
echo "  log:          $TRTLLM_LOG"
echo "  pid:          $TRTLLM_PID_FILE"
echo "  pgid:         $TRTLLM_PGID_FILE"

#APPTAINER_ENV="HF_HOME=/cache/hf,XDG_CACHE_HOME=/cache/job/xdg,FLASHINFER_CACHE_DIR=/cache/job/flashinfer,TRITON_CACHE_DIR=/cache/job/triton,TMPDIR=/cache/job/tmp"

APPTAINER_ENV="HF_HOME=/cache/hf,\
XDG_CACHE_HOME=/cache/job/xdg,\
FLASHINFER_CACHE_DIR=/cache/job/flashinfer,\
TRITON_CACHE_DIR=/cache/job/triton,\
TORCH_HOME=/cache/job/torch,\
CUDA_CACHE_PATH=/cache/job/cuda,\
NUMBA_CACHE_DIR=/cache/job/numba,\
MPLCONFIGDIR=/cache/job/matplotlib,\
PYTHONPYCACHEPREFIX=/cache/job/pycache,\
PIP_CACHE_DIR=/cache/job/pip,\
TMPDIR=/cache/job/tmp,\
TMP=/cache/job/tmp,\
TEMP=/cache/job/tmp,\
HOME=/cache/job/home"

mkdir -p \
  "${JOB_CACHE_DIR}/torch" \
  "${JOB_CACHE_DIR}/cuda" \
  "${JOB_CACHE_DIR}/numba" \
  "${JOB_CACHE_DIR}/matplotlib" \
  "${JOB_CACHE_DIR}/pycache" \
  "${JOB_CACHE_DIR}/pip" \
  "${JOB_CACHE_DIR}/home"

if [[ -n "$HF_TOKEN" ]]; then
    APPTAINER_ENV="${APPTAINER_ENV},HF_TOKEN=${HF_TOKEN}"
fi

#CUDA_VISIBLE_DEVICES="$TRTLLM_GPUS" \
#setsid bash -c '
#  exec apptainer exec --nv \
#    --env "'"$APPTAINER_ENV"'" \
#    -B "'"${HF_CACHE_DIR}"':/cache/hf" \
#    -B "'"${MODELS_DIR}"':/models:ro" \
#    -B "'"${JOB_CACHE_DIR}"':/cache/job" \
#    -B "'"${TRTLLM_CFG_FILE}"':/tmp/trtllm-kvcache.yaml:ro" \
#    "'"$TRTLLM_IMAGE"'" \
#    bash -lc '"'"$TRTLLM_SERVE_CMD"'"'
#' >"$TRTLLM_LOG" 2>&1 &


CUDA_VISIBLE_DEVICES="$TRTLLM_GPUS" \
setsid apptainer exec --nv \
  --env "$APPTAINER_ENV" \
  -B "${HF_CACHE_DIR}:/cache/hf" \
  -B "${MODELS_DIR}:/models:ro" \
  -B "${JOB_CACHE_DIR}:/cache/job" \
  -B "${TRTLLM_CFG_FILE}:/tmp/trtllm-kvcache.yaml:ro" \
  "$TRTLLM_IMAGE" \
  bash -lc "$TRTLLM_SERVE_CMD" \
  >"$TRTLLM_LOG" 2>&1 < /dev/null &
TRTLLM_PID=$!

# The setsid-launched background PID is the process-group leader.
echo "$TRTLLM_PID" > "$TRTLLM_PID_FILE"
echo "$TRTLLM_PID" > "$TRTLLM_PGID_FILE"

echo "Waiting for TRT-LLM on port ${TRTLLM_PORT} (up to 30 min for first-run JIT compile/download)..."
READY=0
for i in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:${TRTLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "TRT-LLM ready (waited ~$((i * 5))s)"
        READY=1
        break
    fi
    if ! kill -0 "$TRTLLM_PID" 2>/dev/null; then
        echo "ERROR: TRT-LLM exited unexpectedly. Check: $TRTLLM_LOG"
        exit 1
    fi
    sleep 5
done

if [[ "$READY" -ne 1 ]]; then
    echo "ERROR: TRT-LLM did not become ready in time. Check: $TRTLLM_LOG"
    exit 1
fi

rm -f "$TRTLLM_CFG_FILE"
trap - EXIT

echo "TRT-LLM started successfully."
echo "  PID:      $TRTLLM_PID"
echo "  PGID:     $TRTLLM_PID"
echo "  PID file: $TRTLLM_PID_FILE"
echo "  PGID file:$TRTLLM_PGID_FILE"
echo "  Log:      $TRTLLM_LOG"