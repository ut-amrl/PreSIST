#!/usr/bin/env bash
set -eo pipefail

# ============================================================
# Workers-only driver: assumes Qwen / Gemma / DAM servers are
# already running (started by another job or shell), and just
# launches pipeline workers for the given datasets.
#
# Specify worker GPUs explicitly via WORKER_GPUS (comma- or
# space-separated). One worker per GPU; if there are more
# datasets than GPUs, datasets are round-robined across workers.
#
# Example:
#   WORKER_GPUS="0,1,2,3" \
#   DATASETS="train_mix_3 train_mix_4 train_mix_5" \
#   ./tacc_run_spf_workers_only.sh
# ============================================================

JOB_START=$(date +%s)

echo "=== SPF workers-only driver (servers assumed up) ==="
echo "  Date: $(date)"
echo "  Host: $(hostname)"
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

export JOB_ID_TAG="${SLURM_JOB_ID:-interactive}"
export JOB_ROOT="${JOB_ROOT:-$SCRATCH/jobs/spf/${JOB_ID_TAG}}"
export LOG_DIR="${LOG_DIR:-$JOB_ROOT/logs}"
export JOB_CACHE_DIR="${JOB_CACHE_DIR:-$JOB_ROOT/cache}"
export JOB_TMP_DIR="${JOB_TMP_DIR:-$JOB_ROOT/tmp}"
export JOB_RUN_DIR="${JOB_RUN_DIR:-$JOB_ROOT/run}"

############################################
# Server ports (must match the running servers)
############################################

TRTLLM_PORT="${TRTLLM_PORT:-9011}"
GEMMA_PORT="${GEMMA_PORT:-9008}"
DAM_PORT="${DAM_PORT:-9014}"

############################################
# What to run
############################################

CONFIG_NAME="${CONFIG_NAME:-q3_30bt_lang_q3T_pU3_sam3b_maxOb_30_mConf_0p35_mA0p001_dd0p9_capQ38bTproc2_idGemmaBBgCCMTD1HL_v3_iqr_only_behav}"
RUN_PHASES="${RUN_PHASES:-object_descriptions}"
DATASETS="${DATASETS:-train_mix_3 train_mix_4 train_mix_5 train_mix_6 train_mix_7}"

LANG_PROMPT_BACKEND="${LANG_PROMPT_BACKEND:-gemma}"
SCENE_CAPTION_BACKEND="${SCENE_CAPTION_BACKEND:-gemma}"
OBJECT_DESC_BACKEND="${OBJECT_DESC_BACKEND:-gemma}"
PERSISTENCE_BACKEND="${PERSISTENCE_BACKEND:-}"

############################################
# Worker GPU list (required)
#   WORKER_GPUS="0,1,2,3"  or  WORKER_GPUS="0 1 2 3"
# One worker per GPU; datasets round-robin across workers.
############################################

WORKER_GPUS="${WORKER_GPUS:-}"
if [[ -z "$WORKER_GPUS" ]]; then
    echo "ERROR: WORKER_GPUS must be set (e.g. WORKER_GPUS=\"0,1,2,3\")."
    exit 1
fi

# Normalize commas to spaces, then split.
WORKER_GPUS_NORM="${WORKER_GPUS//,/ }"
read -ra PIPELINE_GPU_LIST <<< "$WORKER_GPUS_NORM"
NUM_WORKERS=${#PIPELINE_GPU_LIST[@]}
if (( NUM_WORKERS < 1 )); then
    echo "ERROR: WORKER_GPUS parsed to zero GPUs."
    exit 1
fi

############################################
# Backend port lookup (no availability check —
# we assume the servers are up).
############################################

backend_port() {
    case "$1" in
        qwen)  echo "$TRTLLM_PORT" ;;
        gemma) echo "$GEMMA_PORT" ;;
        dam)   echo "$DAM_PORT" ;;
        "")    echo "" ;;
        *)     echo "ERROR: unknown backend '$1'" >&2; return 1 ;;
    esac
}

phase_enabled() {
    case ",$RUN_PHASES," in
        *",$1,"*) return 0 ;;
        *)        return 1 ;;
    esac
}

############################################
# Caches / tmp dirs
############################################

mkdir -p \
  "${HF_CACHE_DIR}" \
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

echo "Worker layout:"
echo "  Worker GPUs: ${PIPELINE_GPU_LIST[*]}  ($NUM_WORKERS workers)"
echo "  Backends (assumed running):"
echo "    lang prompt:          ${LANG_PROMPT_BACKEND:-<config default>}  (qwen:$TRTLLM_PORT gemma:$GEMMA_PORT dam:$DAM_PORT)"
echo "    scene caption:        ${SCENE_CAPTION_BACKEND:-<config default>}"
echo "    object description:   ${OBJECT_DESC_BACKEND:-<config default>}"
echo "    persistence estimate: ${PERSISTENCE_BACKEND:-<config default>}"
echo "  Config:   $CONFIG_NAME"
echo "  Phases:   $RUN_PHASES"
echo "  Datasets: $DATASETS"
echo "  Job root: $JOB_ROOT"
echo "  Logs:     $LOG_DIR"
echo

############################################
# Worker
############################################

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

        local extra_args=""
        emit_vllm_overrides() {
            local prefix="$1"
            local port
            port="$(backend_port "$2")"
            extra_args+=" ++${prefix}.use_vllm_server=True"
            extra_args+=" ++${prefix}.vllm_url_root=http://localhost"
            extra_args+=" ++${prefix}.vllm_port=${port}"
        }

        if [[ -n "$LANG_PROMPT_BACKEND" ]] && phase_enabled segmentation; then
            emit_vllm_overrides "dataset_generator.instanceSegmenter.object_discoverer" "$LANG_PROMPT_BACKEND"
        fi
        if [[ -n "$SCENE_CAPTION_BACKEND" ]] && phase_enabled scene_caption; then
            emit_vllm_overrides "dataset_generator.sceneCaptioner" "$SCENE_CAPTION_BACKEND"
        fi
        if phase_enabled object_descriptions; then
            case "$OBJECT_DESC_BACKEND" in
                dam)
                    extra_args+=" ++dataset_generator.objectDescriber.server_port=${DAM_PORT}"
                    ;;
                qwen|gemma)
                    emit_vllm_overrides "dataset_generator.objectDescriber" "$OBJECT_DESC_BACKEND"
                    ;;
                "") ;;
            esac
        fi
        if [[ -n "$PERSISTENCE_BACKEND" ]] && phase_enabled persistence; then
            emit_vllm_overrides "dataset_generator.llmPersistenceQuerier.llmQuerier" "$PERSISTENCE_BACKEND"
        fi

        local container_cmd
        container_cmd="cd /workspace/code/semantic-persistence-filter && \
HYDRA_FULL_ERROR=1 python3 run_llm_labeling_pipeline.py \
  --config-name ${CONFIG_NAME} \
  +image_dataset=${dataset} \
  ++dataset_generator.run_phases=[${RUN_PHASES}]${extra_args}"

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

############################################
# Round-robin datasets across workers
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

chmod +x "$CODE_DIR/tacc_run_spf_container.sh"

WORKER_PIDS=()
cleanup() {
    echo "Cleaning up workers..."
    for pid in "${WORKER_PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

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
echo "Elapsed: $(( $(date +%s) - JOB_START ))s"

exit $OVERALL_EXIT
