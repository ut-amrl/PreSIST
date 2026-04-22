#!/usr/bin/env bash
set -eo pipefail

mkdir -p logs/spf

JOB_START=$(date +%s)

echo "=== SPF TRT-LLM + DAM multi-worker seg/caption/desc driver ==="
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

export IMAGE="${IMAGE:-$SCRATCH/containers/spf_latest.sif}"
export CODE_DIR="${CODE_DIR:-$WORK/workspaces/semantic-persistence-filter}"
export DATA_DIR="${DATA_DIR:-$SCRATCH/data/semantic-persistence-filter}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$SCRATCH/.cache/huggingface}"

TRTLLM_IMAGE="${TRTLLM_IMAGE:-$SCRATCH/containers/tritonserver_26.03-trtllm-python-py3.sif}"
DAM_IMAGE="${DAM_IMAGE:-$SCRATCH/containers/describe-anything_latest.sif}"

TRTLLM_PORT="${TRTLLM_PORT:-9011}"
TRTLLM_MODEL="${TRTLLM_MODEL:-Qwen/Qwen3-VL-8B-Thinking}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"

DAM_PORT="${DAM_PORT:-9014}"

# CONFIG_NAME must use a sceneCaptioner that is QwenVLSceneLevelCaptioner (not dummy).
# Configs with "cap" in their name (e.g. capQ38bTlong) have this pre-configured.
CONFIG_NAME="${CONFIG_NAME:-q3_30bt_lang_q3T_pU3_sam3b_maxOb_30_mConf_0p35_mA0p001_dd0p9_capQ38bTproc2_v3_iqr_only_behav}"

# Space-separated list; workers round-robin through these in order.
DATASETS="${DATASETS:-train_mix_0 train_mix_1 train_mix_2 train_mix_3 train_mix_4 train_mix_5 train_mix_6 train_mix_7 train_mix_8 train_mix_9 train_mix_10}"

# TP controls how many GPUs TRT-LLM uses; remaining GPUs go to pipeline workers.
#   TP=1 → 1 TRT-LLM GPU, 1 DAM GPU, 6 workers
#   TP=2 → 2 TRT-LLM GPUs, 1 DAM GPU, 5 workers  (default)
#   TP=4 → 4 TRT-LLM GPUs, 1 DAM GPU, 3 workers
TRTLLM_TP="${TRTLLM_TP:-2}"
TOTAL_GPUS="${TOTAL_GPUS:-8}"

# Wall time in hours — used to compute hold-allocation sleep at the end.
WALL_HOURS="${WALL_HOURS:-48}"

############################################
# Derive GPU layout from TP
############################################

# TRT-LLM: GPUs 0..(TP-1)
# DAM:     GPU TP
# Workers: GPUs TP+1..TOTAL_GPUS-1

TRTLLM_GPUS=$(seq -s, 0 $((TRTLLM_TP - 1)))
DAM_GPU_IDX=$TRTLLM_TP
DAM_GPUS="$DAM_GPU_IDX"

PIPELINE_GPU_LIST=()
for i in $(seq $((DAM_GPU_IDX + 1)) $((TOTAL_GPUS - 1))); do
    PIPELINE_GPU_LIST+=("$i")
done
NUM_WORKERS=${#PIPELINE_GPU_LIST[@]}

if [ "$NUM_WORKERS" -lt 1 ]; then
    echo "ERROR: No GPUs left for pipeline workers (TP=$TRTLLM_TP, TOTAL_GPUS=$TOTAL_GPUS)."
    exit 1
fi

mkdir -p "$HF_CACHE_DIR"

echo "GPU layout (TP=$TRTLLM_TP, $TOTAL_GPUS total GPUs):"
echo "  TRT-LLM:  GPUs $TRTLLM_GPUS  (model: $TRTLLM_MODEL, port: $TRTLLM_PORT)"
echo "  DAM:      GPU  $DAM_GPUS  (port: $DAM_PORT)"
echo "  Workers:  GPUs ${PIPELINE_GPU_LIST[*]}  ($NUM_WORKERS workers)"
echo "  Config:   $CONFIG_NAME"
echo "  Datasets: $DATASETS"
echo "  Wall:     ${WALL_HOURS}h"
echo

############################################
# Start TRT-LLM
############################################

TRTLLM_LOG="logs/spf/trtllm_${SLURM_JOB_ID:-interactive}.out"

TRTLLM_CFG_FILE="/tmp/trtllm-kvcache-$$.yaml"
cat > "$TRTLLM_CFG_FILE" <<'EOF'
kv_cache_config:
  enable_block_reuse: false
EOF

TRTLLM_SERVE_CMD="pip install -q --user --upgrade openai && trtllm-serve ${TRTLLM_MODEL} \
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
# Start DAM
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

WORKER_PIDS=()

cleanup() {
    echo "Cleaning up..."
    for pid in "${WORKER_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
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

echo "Waiting for TRT-LLM on port $TRTLLM_PORT (up to 30 min for first-run JIT compile)..."
READY=0
for i in $(seq 1 360); do
    if curl -s "http://localhost:${TRTLLM_PORT}/v1/models" >/dev/null 2>&1; then
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
[ "$READY" -eq 1 ] || { echo "ERROR: TRT-LLM did not become ready in time. Check: $TRTLLM_LOG"; exit 1; }

############################################
# Wait for DAM
############################################

echo "Waiting for DAM on port $DAM_PORT..."
READY=0
for i in $(seq 1 60); do
    if curl -s "http://localhost:${DAM_PORT}/health" >/dev/null 2>&1 || \
       curl -s "http://localhost:${DAM_PORT}/" >/dev/null 2>&1; then
        echo "DAM ready (waited ~$((i * 5))s)"
        READY=1
        break
    fi
    if ! kill -0 "$DAM_PID" 2>/dev/null; then
        echo "ERROR: DAM exited unexpectedly. Check: $DAM_LOG"
        exit 1
    fi
    sleep 5
done
[ "$READY" -eq 1 ] || { echo "ERROR: DAM did not become ready in time. Check: $DAM_LOG"; exit 1; }

echo
echo "Both servers up. Launching $NUM_WORKERS pipeline workers..."
echo

############################################
# Worker function
# Each worker processes its dataset list sequentially on one GPU.
# A failed dataset logs the error and continues to the next one.
############################################

run_worker() {
    set +e  # don't abort worker on individual dataset failure
    local worker_id=$1
    local gpu=$2
    shift 2
    local datasets=("$@")
    local worker_exit=0

    for dataset in "${datasets[@]}"; do
        local log="logs/spf/pipeline_w${worker_id}_${dataset}_${SLURM_JOB_ID:-interactive}.out"
        echo "[worker $worker_id / GPU $gpu] Starting $dataset  →  $log"

        local container_cmd
        # shellcheck disable=SC2089
        container_cmd="cd /workspace/code/semantic-persistence-filter && \
HYDRA_FULL_ERROR=1 python3 run_llm_labeling_pipeline.py \
  --config-name ${CONFIG_NAME} \
  +image_dataset=${dataset} \
  ++dataset_generator.run_phases=[segmentation,scene_caption,object_descriptions] \
  ++dataset_generator.instanceSegmenter.object_discoverer.use_vllm_server=True \
  ++dataset_generator.instanceSegmenter.object_discoverer.vllm_url_root=http://localhost \
  ++dataset_generator.instanceSegmenter.object_discoverer.vllm_port=${TRTLLM_PORT} \
  ++dataset_generator.sceneCaptioner.use_vllm_server=True \
  ++dataset_generator.sceneCaptioner.vllm_url_root=http://localhost \
  ++dataset_generator.sceneCaptioner.vllm_port=${TRTLLM_PORT} \
  ++dataset_generator.objectDescriber.server_port=${DAM_PORT}"

        CUDA_VISIBLE_DEVICES="$gpu" \
        IMAGE="$IMAGE" CODE_DIR="$CODE_DIR" DATA_DIR="$DATA_DIR" HF_CACHE_DIR="$HF_CACHE_DIR" \
            "$CODE_DIR/tacc_run_spf_container.sh" "$container_cmd" \
            >"$log" 2>&1
        local exit_code=$?

        if [ "$exit_code" -ne 0 ]; then
            echo "[worker $worker_id / GPU $gpu] ERROR: $dataset failed (exit $exit_code) — continuing"
            worker_exit=1
        else
            echo "[worker $worker_id / GPU $gpu] Finished $dataset OK"
        fi
    done

    return $worker_exit
}

############################################
# Distribute datasets round-robin
############################################

read -ra DATASET_ARR <<< "$DATASETS"

declare -a WORKER_DATASET_LISTS
for ((i=0; i<NUM_WORKERS; i++)); do
    WORKER_DATASET_LISTS[$i]=""
done
for ((i=0; i<${#DATASET_ARR[@]}; i++)); do
    w=$((i % NUM_WORKERS))
    WORKER_DATASET_LISTS[$w]+="${DATASET_ARR[$i]} "
done

echo "Dataset distribution:"
for ((w=0; w<NUM_WORKERS; w++)); do
    echo "  Worker $w (GPU ${PIPELINE_GPU_LIST[$w]}): ${WORKER_DATASET_LISTS[$w]}"
done
echo

############################################
# Launch workers in parallel
############################################

chmod +x "$CODE_DIR/tacc_run_spf_container.sh"

for ((w=0; w<NUM_WORKERS; w++)); do
    read -ra ds_arr <<< "${WORKER_DATASET_LISTS[$w]}"
    run_worker "$w" "${PIPELINE_GPU_LIST[$w]}" "${ds_arr[@]}" &
    WORKER_PIDS+=($!)
done

############################################
# Wait for all workers
############################################

OVERALL_EXIT=0
for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" || OVERALL_EXIT=1
done

echo
echo "All workers finished  (overall exit: $OVERALL_EXIT)"
echo "End time: $(date)"

############################################
# Sleep to hold allocation until near wall time
############################################

ELAPSED=$(( $(date +%s) - JOB_START ))
HOLD_SECS=$(( WALL_HOURS * 3600 - ELAPSED - 600 ))  # 10-min safety buffer
if [ "$HOLD_SECS" -gt 0 ]; then
    echo "Sleeping ${HOLD_SECS}s to hold allocation (~$((HOLD_SECS/3600))h $((HOLD_SECS%3600/60))m, releases 10 min before wall time)..."
    sleep "$HOLD_SECS"
else
    echo "No time left for hold sleep (elapsed ${ELAPSED}s, wall ${WALL_HOURS}h)."
fi

exit $OVERALL_EXIT
