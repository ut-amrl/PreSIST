#!/usr/bin/env bash
set -eo pipefail
# Smoke-test TRT-LLM on TACC via apptainer.
# Starts the server, waits for readiness, sends one inference request, then shuts down.

mkdir -p logs/spf

echo "=== TRT-LLM smoke test ==="
echo "  Date: $(date)"
echo "  Host: $(hostname)"
echo

############################################
# Modules
############################################

module load tacc-apptainer

############################################
# Config (override via env)
############################################

HF_CACHE_DIR="${HF_CACHE_DIR:-$SCRATCH/.cache/huggingface}"
JOB_CACHE_DIR="${JOB_CACHE_DIR:-$SCRATCH/.job_cache/${SLURM_JOB_ID:-interactive}}"
MODELS_DIR="${MODELS_DIR:-$SCRATCH/models}"

TRTLLM_IMAGE="${TRTLLM_IMAGE:-$SCRATCH/containers/trtllm-qwen3-vl-8b-thinking-26.03-openai.sif}"

TRTLLM_PORT="${TRTLLM_PORT:-9011}"
# Use local path inside container; HF model ID accepted as fallback.
TRTLLM_MODEL="${TRTLLM_MODEL:-/models/Qwen3-VL-8B-Thinking}"
TRTLLM_TP="${TRTLLM_TP:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

# Which GPU(s) to use — comma-separated (e.g. "0" or "0,1")
TRTLLM_GPUS="${TRTLLM_GPUS:-0}"

############################################
# Setup
############################################

mkdir -p \
  "${HF_CACHE_DIR}" \
  "${MODELS_DIR}" \
  "${JOB_CACHE_DIR}/xdg" \
  "${JOB_CACHE_DIR}/flashinfer" \
  "${JOB_CACHE_DIR}/triton" \
  "${JOB_CACHE_DIR}/tmp"

TRTLLM_LOG="logs/spf/trtllm_test_${SLURM_JOB_ID:-interactive}.out"

TRTLLM_CFG_FILE="/tmp/trtllm-kvcache-test-$$.yaml"
cat > "$TRTLLM_CFG_FILE" <<'EOF'
kv_cache_config:
  enable_block_reuse: false
EOF

############################################
# Cleanup
############################################

cleanup() {
    echo "Stopping TRT-LLM (PID ${TRTLLM_PID:-?})..."
    kill "${TRTLLM_PID:-}" 2>/dev/null || true
    wait "${TRTLLM_PID:-}" 2>/dev/null || true
    rm -f "$TRTLLM_CFG_FILE"
}
trap cleanup EXIT

############################################
# Start TRT-LLM
############################################

echo "Config:"
echo "  Image:   $TRTLLM_IMAGE"
echo "  Model:   $TRTLLM_MODEL"
echo "  GPUs:    $TRTLLM_GPUS"
echo "  TP:      $TRTLLM_TP"
echo "  Port:    $TRTLLM_PORT"
echo "  MaxSeq:  $MAX_SEQ_LEN"
echo "  Models:  $MODELS_DIR → /models"
echo "  Log:     $TRTLLM_LOG"
echo

APPTAINER_FLAGS=(
  --nv
  --env "HF_HOME=/cache/hf,XDG_CACHE_HOME=/cache/job/xdg,FLASHINFER_CACHE_DIR=/cache/job/flashinfer,TRITON_CACHE_DIR=/cache/job/triton,TMPDIR=/cache/job/tmp,CUDA_VISIBLE_DEVICES=${TRTLLM_GPUS}"
  -B "${HF_CACHE_DIR}:/cache/hf"
  -B "${MODELS_DIR}:/models:ro"
  -B "${JOB_CACHE_DIR}:/cache/job"
  -B "${TRTLLM_CFG_FILE}:/tmp/trtllm-kvcache.yaml:ro"
)

TRTLLM_SERVE_CMD="pip install -q --upgrade openai && trtllm-llmapi-launch trtllm-serve ${TRTLLM_MODEL} \
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

if [ "$TRTLLM_TP" -gt 1 ]; then
  # TP>1 requires real MPI worker spawning. srun --mpi=pmi2 provides the PMI2
  # context the container's OpenMPI needs; without it, MPI_Comm_create_group segfaults.
  # srun creates TRTLLM_TP tasks on this node; trtllm-llmapi-launch assigns
  # rank 0 as the API server and the rest as TP workers.
  # --preserve-fds passes the PMI2 file descriptor (PMI_FD) into the container;
  # apptainer blocks it by default, which causes PMI2_Init to fail inside.
  srun -n "$TRTLLM_TP" --ntasks-per-node="$TRTLLM_TP" --mpi=pmi2 \
    apptainer exec --preserve-fds 20 "${APPTAINER_FLAGS[@]}" \
    "$TRTLLM_IMAGE" \
    bash -c "$TRTLLM_SERVE_CMD" \
    >"$TRTLLM_LOG" 2>&1 &
else
  # TP=1: trtllm-llmapi-launch uses IPC (no MPI needed); unset SLURM PMI vars
  # so OpenMPI doesn't attempt PMI2 init and crash.
  TRTLLM_SERVE_CMD="unset PMI_FD PMI_RANK PMI_SIZE PMI_JOBID SLURM_STEP_ID SLURM_STEP_NUM_NODES SLURM_PROCID SLURM_NODEID && $TRTLLM_SERVE_CMD"
  apptainer exec "${APPTAINER_FLAGS[@]}" \
    "$TRTLLM_IMAGE" \
    bash -lc "$TRTLLM_SERVE_CMD" \
    >"$TRTLLM_LOG" 2>&1 &
fi
TRTLLM_PID=$!

echo "TRT-LLM started (PID $TRTLLM_PID). Waiting for readiness (up to 30 min)..."

READY=0
for i in $(seq 1 360); do
    if curl -sf "http://localhost:${TRTLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "TRT-LLM ready after ~$((i * 5))s"
        READY=1
        break
    fi
    if ! kill -0 "$TRTLLM_PID" 2>/dev/null; then
        echo "ERROR: TRT-LLM exited early. Tail of log:"
        tail -40 "$TRTLLM_LOG"
        exit 1
    fi
    sleep 5
done

if [ "$READY" -eq 0 ]; then
    echo "ERROR: TRT-LLM did not become ready in 30 min. Tail of log:"
    tail -40 "$TRTLLM_LOG"
    exit 1
fi

############################################
# Smoke test — one short inference
############################################

echo
echo "Running inference smoke test..."

MODEL_ID=$(curl -sf "http://localhost:${TRTLLM_PORT}/v1/models" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data['data'][0]['id'])
" 2>/dev/null || echo "$TRTLLM_MODEL")

RESPONSE=$(curl -sf "http://localhost:${TRTLLM_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL_ID}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: OK\"}],
    \"max_tokens\": 32
  }")

echo "Model ID in server: $MODEL_ID"
echo "Response:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"

echo
echo "=== TRT-LLM smoke test PASSED ==="
