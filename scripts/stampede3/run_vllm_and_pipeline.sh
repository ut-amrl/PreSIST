#!/usr/bin/env bash
# Shared logic for starting vLLM and running the LLM labeling pipeline on Stampede3.
# Source this from sbatch_h100_vllm_pipeline.sh, or run standalone from an idev session.
#
# Where code and data live (mirrors docker_start_spf.sh volumes):
#   Host PROJECT_ROOT     -> container /workspace/code/semantic-persistence-filter  (repo code)
#   Host SPF_DATA         -> container /workspace/data                               (datasets; optional)
#   Host HF_CACHE         -> container /opt/hf                                      (HuggingFace cache)
# Put code on $WORK or $SCRATCH; put data and HF cache on $SCRATCH (or $WORK). See sbatch/idev comments.
#
# Required env: APPTAINER_IMAGE, PROJECT_ROOT.
# Optional env: SPF_DATA, HF_CACHE, CONFIG_NAME, IMAGE_DATASET, VLLM_*, RUN_LOG_DIR, PIPELINE_EXTRA_ARGS.
#
# Example (idev, after allocation):
#   export APPTAINER_IMAGE=$SCRATCH/spf.sif
#   export PROJECT_ROOT=$WORK/semantic-persistence-filter
#   export SPF_DATA=$SCRATCH/data/semantic-persistence-filter
#   export HF_CACHE=$SCRATCH/hf_cache
#   export IMAGE_DATASET=openloris_location_all
#   ./scripts/stampede3/run_vllm_and_pipeline.sh
#
# Example (override config/dataset/vLLM):
#   CONFIG_NAME=my_config IMAGE_DATASET=other_dataset VLLM_PORT=9002 ./scripts/stampede3/run_vllm_and_pipeline.sh

set -euo pipefail

# -----------------------------------------------------------------------------
# Defaults and derived vars (overridden by caller when sourcing)
# -----------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
SPF_DATA="${SPF_DATA:-}"
CONFIG_NAME="${CONFIG_NAME:-init_dataset_gen_qwen3_30b_instruct_semantic_sam_minArea0p001}"
IMAGE_DATASET="${IMAGE_DATASET:-openloris_location_all}"

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_PORT="${VLLM_PORT:-9001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

HF_CACHE="${HF_CACHE:-${SCRATCH:-/tmp}/hf_cache}"
RUN_LOG_DIR="${RUN_LOG_DIR:-}"
PIPELINE_EXTRA_ARGS="${PIPELINE_EXTRA_ARGS:-}"

# Derived: served model name must match qwen_llm_querier._derive_served_model_name()
MODEL_BASENAME="$(basename "${VLLM_MODEL}")"
MODEL_NORMALIZED="$(echo "${MODEL_BASENAME}" | tr '[:upper:]' '[:lower:]')"
SERVED_MODEL_NAME="${MODEL_NORMALIZED}__${VLLM_DTYPE}"

# Tensor parallel: use all GPUs on the node when in Slurm, else 1
GPU_COUNT="${SLURM_GPUS_ON_NODE:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${GPU_COUNT}}"

CONTAINER_BIND="-B ${PROJECT_ROOT}:/workspace/code/semantic-persistence-filter -B ${HF_CACHE}:/opt/hf"
[[ -n "${SPF_DATA:-}" ]] && [[ -d "${SPF_DATA}" ]] && CONTAINER_BIND="${CONTAINER_BIND} -B ${SPF_DATA}:/workspace/data"
CONTAINER_PWD="/workspace/code/semantic-persistence-filter"

# PID of background vLLM process (set by start_vllm_background)
vllm_pid=""

# -----------------------------------------------------------------------------
# Functions (used by sbatch and optionally by idev)
# -----------------------------------------------------------------------------

start_vllm_background() {
  if [[ -z "${APPTAINER_IMAGE:-}" ]] || [[ ! -f "${APPTAINER_IMAGE}" ]]; then
    echo "ERROR: APPTAINER_IMAGE must be set and point to an existing SIF." >&2
    return 1
  fi
  if [[ ! -d "${PROJECT_ROOT}" ]]; then
    echo "ERROR: PROJECT_ROOT not found: ${PROJECT_ROOT}" >&2
    return 1
  fi
  echo "Launching vLLM..."
  if [[ -n "${RUN_LOG_DIR:-}" ]]; then
    apptainer exec --nv ${CONTAINER_BIND} --pwd "${CONTAINER_PWD}" "${APPTAINER_IMAGE}" \
      bash -lc "vllm serve \
        --model '${VLLM_MODEL}' \
        --host 0.0.0.0 \
        --port '${VLLM_PORT}' \
        --trust-remote-code \
        --served-model-name '${SERVED_MODEL_NAME}' \
        --dtype '${VLLM_DTYPE}' \
        --tensor-parallel-size '${TENSOR_PARALLEL_SIZE}' \
        --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \
        --max-model-len '${MAX_MODEL_LEN}'" \
      > "${RUN_LOG_DIR}/vllm.log" 2>&1 &
  else
    apptainer exec --nv ${CONTAINER_BIND} --pwd "${CONTAINER_PWD}" "${APPTAINER_IMAGE}" \
      bash -lc "vllm serve \
        --model '${VLLM_MODEL}' \
        --host 0.0.0.0 \
        --port '${VLLM_PORT}' \
        --trust-remote-code \
        --served-model-name '${SERVED_MODEL_NAME}' \
        --dtype '${VLLM_DTYPE}' \
        --tensor-parallel-size '${TENSOR_PARALLEL_SIZE}' \
        --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \
        --max-model-len '${MAX_MODEL_LEN}'" &
  fi
  vllm_pid=$!
}

wait_for_vllm() {
  local max_wait_s="${1:-600}"
  local elapsed=0
  echo "Waiting for vLLM readiness on /v1/models ..."
  until curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
    sleep 2
    elapsed=$((elapsed + 2))
    if (( elapsed >= max_wait_s )); then
      echo "ERROR: vLLM did not become ready within ${max_wait_s}s" >&2
      [[ -n "${RUN_LOG_DIR:-}" ]] && echo "See ${RUN_LOG_DIR}/vllm.log" >&2
      return 1
    fi
  done
  echo "vLLM is ready."
}

run_pipeline() {
  if [[ -z "${APPTAINER_IMAGE:-}" ]] || [[ ! -f "${APPTAINER_IMAGE}" ]]; then
    echo "ERROR: APPTAINER_IMAGE must be set and point to an existing SIF." >&2
    return 1
  fi
  echo "Starting pipeline..."
  local pipeline_cmd="HYDRA_FULL_ERROR=1 python3 run_llm_labeling_pipeline.py \
    --config-name '${CONFIG_NAME}' \
    +image_dataset='${IMAGE_DATASET}' \
    ++dataset_generator.llmPersistenceQuerier.llmQuerier.use_vllm_server=True \
    ++dataset_generator.llmPersistenceQuerier.llmQuerier.vllm_url_root=http://localhost \
    ++dataset_generator.llmPersistenceQuerier.llmQuerier.vllm_port=${VLLM_PORT} \
    +dataset_generator/objectDescriber=dummyObjectDescriber \
    ${PIPELINE_EXTRA_ARGS}"
  if [[ -n "${RUN_LOG_DIR:-}" ]]; then
    apptainer exec --nv ${CONTAINER_BIND} --pwd "${CONTAINER_PWD}" "${APPTAINER_IMAGE}" \
      bash -lc "${pipeline_cmd}" \
      > "${RUN_LOG_DIR}/pipeline.log" 2>&1
  else
    apptainer exec --nv ${CONTAINER_BIND} --pwd "${CONTAINER_PWD}" "${APPTAINER_IMAGE}" \
      bash -lc "${pipeline_cmd}"
  fi
}

cleanup_vllm() {
  if [[ -n "${vllm_pid:-}" ]] && kill -0 "${vllm_pid}" 2>/dev/null; then
    echo "Stopping vLLM (pid=${vllm_pid})"
    kill "${vllm_pid}" || true
    wait "${vllm_pid}" || true
  fi
}

# -----------------------------------------------------------------------------
# Standalone mode: run full flow (for idev: ./run_vllm_and_pipeline.sh)
# -----------------------------------------------------------------------------
main() {
  if [[ -z "${APPTAINER_IMAGE:-}" ]] || [[ ! -f "${APPTAINER_IMAGE}" ]]; then
    echo "ERROR: Set APPTAINER_IMAGE to your SIF path. PROJECT_ROOT defaults to \$PWD." >&2
    exit 1
  fi
  start_vllm_background
  trap cleanup_vllm EXIT
  wait_for_vllm
  run_pipeline
  echo "Pipeline completed successfully."
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
