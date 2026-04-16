#!/usr/bin/env bash
set -eo pipefail

mkdir -p logs/spf

echo "=== SPF TRT-LLM + DAM seg/desc driver ==="
echo "  Date:   $(date)"
echo "  Host:   $(hostname)"
echo

############################################
# Modules
############################################

module load tacc-apptainer

############################################
# Paths (can override via env)
############################################

export IMAGE="${IMAGE:-$SCRATCH/containers/spf_latest.sif}"
export CODE_DIR="${CODE_DIR:-$WORK/workspaces/semantic-persistence-filter}"
export DATA_DIR="${DATA_DIR:-$SCRATCH/data/semantic-persistence-filter}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$SCRATCH/.cache/huggingface}"

TRTLLM_IMAGE="${TRTLLM_IMAGE:-$SCRATCH/containers/tritonserver_26.03-trtllm-python-py3.sif}"
DAM_IMAGE="${DAM_IMAGE:-$SCRATCH/containers/describe-anything_latest.sif}"

TRTLLM_PORT="${TRTLLM_PORT:-9011}"
TRTLLM_MODEL="${TRTLLM_MODEL:-Qwen/Qwen3-VL-8B-Thinking}"
TRTLLM_TP="${TRTLLM_TP:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"

DAM_PORT="${DAM_PORT:-9014}"

CONFIG_NAME="${CONFIG_NAME:-qwen3_30b_t_lang_q3T_pU3_sam3batch_maxObj_30_minConf_0p35_mA0p001_dedupe0p9_v3_iqr_only}"
DATASET_NAME="${DATASET_NAME:-train_mix_0}"

############################################
# GPU assignment (can override via env)
############################################

TRTLLM_GPUS="${TRTLLM_GPUS:-0,1,2,3}"
DAM_GPUS="${DAM_GPUS:-4}"
PIPELINE_GPUS="${PIPELINE_GPUS:-5,6,7}"

############################################
# Derived config
############################################

mkdir -p "$HF_CACHE_DIR"

echo "Using:"
echo "  SPF image:      $IMAGE"
echo "  TRT-LLM image:  $TRTLLM_IMAGE"
echo "  DAM image:      $DAM_IMAGE"
echo "  Code dir:       $CODE_DIR"
echo "  Data dir:       $DATA_DIR"
echo "  HF cache:       $HF_CACHE_DIR"
echo "  TRT-LLM GPUs:   $TRTLLM_GPUS  (model: $TRTLLM_MODEL, port: $TRTLLM_PORT)"
echo "  DAM GPUs:       $DAM_GPUS  (port: $DAM_PORT)"
echo "  Pipeline GPUs:  $PIPELINE_GPUS"
echo "  Config:         $CONFIG_NAME"
echo "  Dataset:        $DATASET_NAME"
echo

############################################
# Start TRT-LLM in the background
############################################

TRTLLM_LOG="logs/spf/trtllm_${SLURM_JOB_ID:-interactive}.out"

# Qwen3-VL requires pytorch backend and KV cache block reuse disabled
TRTLLM_CFG_FILE="/tmp/trtllm-kvcache-$$.yaml"
cat > "$TRTLLM_CFG_FILE" <<EOF
kv_cache_config:
  enable_block_reuse: false
EOF

TRTLLM_SERVE_CMD="pip install -q --upgrade openai && trtllm-serve ${TRTLLM_MODEL} \
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

echo "Starting TRT-LLM server on GPUs [$TRTLLM_GPUS] (logs: $TRTLLM_LOG)..."
CUDA_VISIBLE_DEVICES="$TRTLLM_GPUS" \
apptainer exec --nv \
  -B "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  -B "${TRTLLM_CFG_FILE}:/tmp/trtllm-kvcache.yaml:ro" \
  "$TRTLLM_IMAGE" \
  bash -lc "$TRTLLM_SERVE_CMD" \
  >"$TRTLLM_LOG" 2>&1 &

TRTLLM_PID=$!

############################################
# Start DAM in the background
############################################

DAM_LOG="logs/spf/dam_${SLURM_JOB_ID:-interactive}.out"

echo "Starting DAM server on GPU [$DAM_GPUS] (logs: $DAM_LOG)..."
CUDA_VISIBLE_DEVICES="$DAM_GPUS" \
apptainer exec --nv \
  -B "${HF_CACHE_DIR}:/workspace/hf_home" \
  "$DAM_IMAGE" \
  bash -lc "python /workspace/project/dam_server.py --host 0.0.0.0 --port ${DAM_PORT}" \
  >"$DAM_LOG" 2>&1 &

DAM_PID=$!

############################################
# Cleanup trap
############################################

cleanup() {
  for pid_var in TRTLLM_PID DAM_PID; do
    local pid="${!pid_var:-}"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "Stopping server (PID $pid)..."
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  rm -f "$TRTLLM_CFG_FILE"
}
trap cleanup EXIT

############################################
# Wait for TRT-LLM
############################################

echo "Waiting for TRT-LLM server on port $TRTLLM_PORT (up to 30 min for first-run JIT compile)..."
READY=0
for i in $(seq 1 360); do
  if curl -s "http://localhost:${TRTLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "TRT-LLM server is ready (waited ~$((i * 5)) seconds)"
    READY=1
    break
  fi
  if ! kill -0 "$TRTLLM_PID" 2>/dev/null; then
    echo "ERROR: TRT-LLM process exited unexpectedly. Check: $TRTLLM_LOG"
    exit 1
  fi
  sleep 5
done

if [ "$READY" -ne 1 ]; then
  echo "ERROR: TRT-LLM did not become ready in time. Check: $TRTLLM_LOG"
  exit 1
fi

############################################
# Wait for DAM
############################################

echo "Waiting for DAM server on port $DAM_PORT..."
READY=0
for i in $(seq 1 60); do
  if curl -s "http://localhost:${DAM_PORT}/health" >/dev/null 2>&1 || \
     curl -s "http://localhost:${DAM_PORT}/" >/dev/null 2>&1; then
    echo "DAM server is ready (waited ~$((i * 5)) seconds)"
    READY=1
    break
  fi
  if ! kill -0 "$DAM_PID" 2>/dev/null; then
    echo "ERROR: DAM process exited unexpectedly. Check: $DAM_LOG"
    exit 1
  fi
  sleep 5
done

if [ "$READY" -ne 1 ]; then
  echo "ERROR: DAM server did not become ready in time. Check: $DAM_LOG"
  exit 1
fi

echo
echo "Both servers are up. Starting containerized pipeline..."
echo

############################################
# Run the pipeline
############################################

chmod +x "$CODE_DIR/tacc_run_spf_container.sh"

export CUDA_VISIBLE_DEVICES="$PIPELINE_GPUS"

CONTAINER_CMD="cd /workspace/code/semantic-persistence-filter && \
HYDRA_FULL_ERROR=1 python3 run_llm_labeling_pipeline.py \
  --config-name ${CONFIG_NAME} \
  +image_dataset=${DATASET_NAME} \
  ++dataset_generator.run_phases=[segmentation,object_descriptions] \
  ++dataset_generator.instanceSegmenter.object_discoverer.use_vllm_server=True \
  ++dataset_generator.instanceSegmenter.object_discoverer.vllm_url_root=http://localhost \
  ++dataset_generator.instanceSegmenter.object_discoverer.vllm_port=${TRTLLM_PORT} \
  ++dataset_generator.objectDescriber.server_port=${DAM_PORT}"

echo "Running pipeline on GPUs [$PIPELINE_GPUS]:"
echo "  $CONTAINER_CMD"
echo

IMAGE="$IMAGE" CODE_DIR="$CODE_DIR" DATA_DIR="$DATA_DIR" HF_CACHE_DIR="$HF_CACHE_DIR" \
  "$CODE_DIR/tacc_run_spf_container.sh" "$CONTAINER_CMD"

PIPELINE_EXIT=$?

echo
echo "Pipeline finished with exit code: $PIPELINE_EXIT"
echo "End time: $(date)"

exit $PIPELINE_EXIT
