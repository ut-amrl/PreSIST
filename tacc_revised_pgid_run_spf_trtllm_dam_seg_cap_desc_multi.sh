#!/usr/bin/env bash
set -eo pipefail

#mkdir -p logs/spf

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

# Per-job cache root: JIT/temp/write-heavy caches
#export JOB_CACHE_DIR="${JOB_CACHE_DIR:-$SCRATCH/.job_cache/${SLURM_JOB_ID:-interactive}}"
# Per-job writable root on SCRATCH
export JOB_ID_TAG="${SLURM_JOB_ID:-interactive}"
export JOB_ROOT="${JOB_ROOT:-$SCRATCH/jobs/spf/${JOB_ID_TAG}}"
export LOG_DIR="${LOG_DIR:-$JOB_ROOT/logs}"
export JOB_CACHE_DIR="${JOB_CACHE_DIR:-$JOB_ROOT/cache}"
export JOB_TMP_DIR="${JOB_TMP_DIR:-$JOB_ROOT/tmp}"
export JOB_RUN_DIR="${JOB_RUN_DIR:-$JOB_ROOT/run}"


TRTLLM_IMAGE="${TRTLLM_IMAGE:-$SCRATCH/containers/trtllm-qwen3-vl-8b-thinking-26.03-openai.sif}"
DAM_IMAGE="${DAM_IMAGE:-$SCRATCH/containers/describe-anything_latest.sif}"

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


DAM_PORT="${DAM_PORT:-9014}"
DAM_WORKERS="${DAM_WORKERS:-3}"

CONFIG_NAME="${CONFIG_NAME:-q3_30bt_lang_q3T_pU3_sam3b_maxOb_30_mConf_0p35_mA0p001_dd0p9_capQ38bTproc2_v3_iqr_only_behav}"

DATASETS="${DATASETS:-train_mix_0 train_mix_1 train_mix_2 train_mix_3 train_mix_4 train_mix_5 train_mix_6 train_mix_7 train_mix_8 train_mix_9 train_mix_10}"

TRTLLM_TP="${TRTLLM_TP:-1}"
TOTAL_GPUS="${TOTAL_GPUS:-8}"

WALL_HOURS="${WALL_HOURS:-48}"

# Set USE_VLLM=1 to use vLLM (micromamba) instead of TRT-LLM (apptainer).
# Auto-enabled for multi-GPU since TRT-LLM does not support it.
if [[ "${TRTLLM_TP:-1}" -gt 1 && -z "${USE_VLLM:-}" ]]; then
    echo "NOTE: TRTLLM_TP=$TRTLLM_TP > 1; TRT-LLM does not support multi-GPU — switching to vLLM."
    USE_VLLM=1
fi
USE_VLLM="${USE_VLLM:-}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_QUANT="${VLLM_QUANT:-}"

############################################
# Derive GPU layout from TP
############################################

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

# Ensure container env sees the same paths even if wrapper uses apptainer with its own env handling.
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

LLM_SERVER="${USE_VLLM:+vLLM}"
LLM_SERVER="${LLM_SERVER:-TRT-LLM}"
echo "GPU layout (TP=$TRTLLM_TP, $TOTAL_GPUS total GPUs):"
echo "  $LLM_SERVER:  GPUs $TRTLLM_GPUS  (model: $TRTLLM_MODEL, port: $TRTLLM_PORT)"
echo "  DAM:      GPU  $DAM_GPUS  (port: $DAM_PORT, workers $DAM_WORKERS)"
echo "  Workers:  GPUs ${PIPELINE_GPU_LIST[*]}  ($NUM_WORKERS workers)"
echo "  Config:   $CONFIG_NAME"
echo "  Datasets: $DATASETS"
echo "  Wall:     ${WALL_HOURS}h"
echo "  Job root: $JOB_ROOT"
echo "  Logs:     $LOG_DIR"
echo

############################################
# Start LLM server (TRT-LLM or vLLM)
############################################

TRTLLM_LOG="${LOG_DIR}/trtllm_${JOB_ID_TAG}.out"
TRTLLM_PID_FILE="${JOB_RUN_DIR}/trtllm_${JOB_ID_TAG}.pid"
TRTLLM_PGID_FILE="${JOB_RUN_DIR}/trtllm_${JOB_ID_TAG}.pgid"

if [[ -n "$USE_VLLM" ]]; then
    VLLM_START_SCRIPT="${VLLM_START_SCRIPT:-$CODE_DIR/tacc_start_vllm_server_pgid.sh}"
    if [ ! -x "$VLLM_START_SCRIPT" ]; then
        echo "ERROR: vLLM start script not found or not executable: $VLLM_START_SCRIPT"
        exit 1
    fi
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
else
    TRTLLM_START_SCRIPT="${TRTLLM_START_SCRIPT:-$CODE_DIR/tacc_start_trtllm_server_pgid.sh}"
    if [ ! -x "$TRTLLM_START_SCRIPT" ]; then
        echo "ERROR: TRT-LLM start script not found or not executable: $TRTLLM_START_SCRIPT"
        exit 1
    fi
    echo "Starting TRT-LLM server via $TRTLLM_START_SCRIPT ..."
    TRTLLM_IMAGE="$TRTLLM_IMAGE" \
    MODELS_DIR="$MODELS_DIR" \
    HF_CACHE_DIR="$HF_CACHE_DIR" \
    JOB_CACHE_DIR="$JOB_CACHE_DIR" \
    JOB_TMP_DIR="$JOB_TMP_DIR" \
    JOB_ROOT="$JOB_ROOT" \
    LOG_DIR="$LOG_DIR" \
    TRTLLM_MODEL="$TRTLLM_MODEL" \
    TRTLLM_PORT="$TRTLLM_PORT" \
    TRTLLM_TP="$TRTLLM_TP" \
    TRTLLM_GPUS="$TRTLLM_GPUS" \
    MAX_SEQ_LEN="$MAX_SEQ_LEN" \
    HF_TOKEN="$HF_TOKEN" \
    TRTLLM_LOG="$TRTLLM_LOG" \
    TRTLLM_PID_FILE="$TRTLLM_PID_FILE" \
    TRTLLM_PGID_FILE="$TRTLLM_PGID_FILE" \
    "$TRTLLM_START_SCRIPT"
fi

############################################
# Start DAM
############################################

DAM_LOG="${LOG_DIR}/dam_${JOB_ID_TAG}.out"


echo "Starting DAM server on GPU [$DAM_GPUS] (logs: $DAM_LOG)..."
CUDA_VISIBLE_DEVICES="$DAM_GPUS" \
apptainer exec --nv \
  -B "${HF_CACHE_DIR}:/workspace/hf_home" \
  -B "${JOB_TMP_DIR}:${JOB_TMP_DIR}" \
  "$DAM_IMAGE" \
  bash -lc "python /workspace/project/dam_server.py --host 0.0.0.0 --port ${DAM_PORT} --workers ${DAM_WORKERS:-1}" \
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

    if [ -f "$TRTLLM_PGID_FILE" ]; then
        TRTLLM_PGID="$(cat "$TRTLLM_PGID_FILE" 2>/dev/null || true)"
        if [ -n "${TRTLLM_PGID:-}" ]; then
            echo "Stopping TRT-LLM process group $TRTLLM_PGID ..."
            kill -TERM -- "-$TRTLLM_PGID" 2>/dev/null || true
            sleep 5
            kill -KILL -- "-$TRTLLM_PGID" 2>/dev/null || true
        fi
        rm -f "$TRTLLM_PGID_FILE"
    elif [ -f "$TRTLLM_PID_FILE" ]; then
        TRTLLM_PID="$(cat "$TRTLLM_PID_FILE" 2>/dev/null || true)"
        if [ -n "${TRTLLM_PID:-}" ] && kill -0 "$TRTLLM_PID" 2>/dev/null; then
            echo "Stopping TRT-LLM server (PID $TRTLLM_PID)..."
            kill "$TRTLLM_PID" 2>/dev/null || true
            wait "$TRTLLM_PID" 2>/dev/null || true
        fi
    fi
    rm -f "$TRTLLM_PID_FILE"

    if [ -n "${DAM_PID:-}" ] && kill -0 "$DAM_PID" 2>/dev/null; then
        echo "Stopping DAM server (PID $DAM_PID)..."
        kill "$DAM_PID" 2>/dev/null || true
        wait "$DAM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

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

run_worker() {
    set +e
    local worker_id=$1
    local gpu=$2
    shift 2
    local datasets=("$@")
    local worker_exit=0

    for dataset in "${datasets[@]}"; do
        local log="${LOG_DIR}/pipeline_w${worker_id}_${dataset}_${JOB_ID_TAG}.out"

	echo "[worker $worker_id / GPU $gpu] Starting $dataset  →  $log"

        local container_cmd
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
        IMAGE="$IMAGE" \
        CODE_DIR="$CODE_DIR" \
        DATA_DIR="$DATA_DIR" \
        HF_CACHE_DIR="$HF_CACHE_DIR" \
        JOB_ROOT="$JOB_ROOT" \
        JOB_CACHE_DIR="$JOB_CACHE_DIR" \
        JOB_TMP_DIR="$JOB_TMP_DIR" \
        LOG_DIR="$LOG_DIR" \
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

chmod +x "$CODE_DIR/tacc_run_spf_container.sh"

for ((w=0; w<NUM_WORKERS; w++)); do
    read -ra ds_arr <<< "${WORKER_DATASET_LISTS[$w]}"
    run_worker "$w" "${PIPELINE_GPU_LIST[$w]}" "${ds_arr[@]}" &
    WORKER_PIDS+=($!)
done

OVERALL_EXIT=0
for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" || OVERALL_EXIT=1
done

echo
echo "All workers finished  (overall exit: $OVERALL_EXIT)"
echo "End time: $(date)"

ELAPSED=$(( $(date +%s) - JOB_START ))
HOLD_SECS=$(( WALL_HOURS * 3600 - ELAPSED - 600 ))
if [ "$HOLD_SECS" -gt 0 ]; then
    echo "Sleeping ${HOLD_SECS}s to hold allocation (~$((HOLD_SECS/3600))h $((HOLD_SECS%3600/60))m, releases 10 min before wall time)..."
    sleep "$HOLD_SECS"
else
    echo "No time left for hold sleep (elapsed ${ELAPSED}s, wall ${WALL_HOURS}h)."
fi

exit $OVERALL_EXIT

