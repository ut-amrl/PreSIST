#!/usr/bin/env bash
# Shared logic for starting TRT-LLM and running the LLM labeling pipeline on Stampede3.
# Source this from sbatch_h100_trtllm_pipeline.sh, or run standalone from an idev session.
#
# Mirrors run_vllm_and_pipeline.sh but uses trtllm-serve instead of vllm serve.
#
# Two images are required (unlike the vLLM variant which uses one):
#   TRTLLM_IMAGE  - Apptainer SIF of the TRT-LLM Tritonserver container
#                   Build with: apptainer pull docker://nvcr.io/nvidia/tritonserver:25.07-trtllm-python-py3
#   APPTAINER_IMAGE - Apptainer SIF of the SPF project container (same as vLLM variant)
#
# Where code and data live (mirrors docker_start_spf.sh volumes):
#   Host PROJECT_ROOT     -> container /workspace/code/semantic-persistence-filter  (repo code)
#   Host SPF_DATA         -> container /workspace/data                               (datasets; optional)
#   Host HF_CACHE         -> container /opt/hf                                      (HuggingFace cache)
#
# Required env: TRTLLM_IMAGE, APPTAINER_IMAGE, PROJECT_ROOT.
# Optional env: SPF_DATA, HF_CACHE, CONFIG_NAME, IMAGE_DATASET, TRTLLM_*, RUN_LOG_DIR, PIPELINE_EXTRA_ARGS.
#
# Example (idev, after allocation):
#   export TRTLLM_IMAGE=$SCRATCH/tritonserver_25.07-trtllm-python-py3.sif
#   export APPTAINER_IMAGE=$SCRATCH/spf.sif
#   export PROJECT_ROOT=$WORK/semantic-persistence-filter
#   export SPF_DATA=$SCRATCH/data/semantic-persistence-filter
#   export HF_CACHE=$SCRATCH/hf_cache
#   export IMAGE_DATASET=openloris_location_all
#   ./scripts/stampede3/run_trtllm_and_pipeline.sh
#
# Example (override config/dataset/model):
#   CONFIG_NAME=my_config IMAGE_DATASET=other_dataset TRTLLM_PORT=9002 ./scripts/stampede3/run_trtllm_and_pipeline.sh

set -euo pipefail

# -----------------------------------------------------------------------------
# Defaults and derived vars (overridden by caller when sourcing)
# -----------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
SPF_DATA="${SPF_DATA:-}"
CONFIG_NAME="${CONFIG_NAME:-init_dataset_gen_qwen3_30b_instruct_semantic_sam_minArea0p001}"
IMAGE_DATASET="${IMAGE_DATASET:-openloris_location_all}"

TRTLLM_MODEL="${TRTLLM_MODEL:-Qwen/Qwen3-30B-A3B-Thinking-2507}"
TRTLLM_PORT="${TRTLLM_PORT:-9001}"
KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTION:-0.90}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"

HF_CACHE="${HF_CACHE:-${SCRATCH:-/tmp}/hf_cache}"
RUN_LOG_DIR="${RUN_LOG_DIR:-}"
PIPELINE_EXTRA_ARGS="${PIPELINE_EXTRA_ARGS:-}"

# Tensor parallel: use all GPUs on the node when in Slurm, else 1
GPU_COUNT="${SLURM_GPUS_ON_NODE:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${GPU_COUNT}}"

# Auto-select backend:
#   VLMs and MoE models require --backend pytorch.
#   Dense text-only models use the default C++ backend (omit the flag).
# Pattern: -vl / vision / vlm  -> VLM;  -a<N>b / moe -> MoE
TRTLLM_MODEL_LC="$(echo "${TRTLLM_MODEL}" | tr '[:upper:]' '[:lower:]')"
if [[ "${TRTLLM_MODEL_LC}" == *"-vl"* || "${TRTLLM_MODEL_LC}" == *"vision"* || "${TRTLLM_MODEL_LC}" == *"vlm"* ]]; then
  TRTLLM_BACKEND="pytorch"
elif [[ "${TRTLLM_MODEL_LC}" =~ -a[0-9]+b || "${TRTLLM_MODEL_LC}" == *"moe"* ]]; then
  TRTLLM_BACKEND="pytorch"
else
  TRTLLM_BACKEND=""
fi

CONTAINER_BIND="-B ${PROJECT_ROOT}:/workspace/code/semantic-persistence-filter -B ${HF_CACHE}:/opt/hf"
[[ -n "${SPF_DATA:-}" ]] && [[ -d "${SPF_DATA}" ]] && CONTAINER_BIND="${CONTAINER_BIND} -B ${SPF_DATA}:/workspace/data"
CONTAINER_PWD="/workspace/code/semantic-persistence-filter"

# PID of background TRT-LLM process (set by start_trtllm_background)
trtllm_pid=""

# -----------------------------------------------------------------------------
# Functions (used by sbatch and optionally by idev)
# -----------------------------------------------------------------------------

start_trtllm_background() {
  if [[ -z "${TRTLLM_IMAGE:-}" ]] || [[ ! -f "${TRTLLM_IMAGE}" ]]; then
    echo "ERROR: TRTLLM_IMAGE must be set and point to an existing SIF." >&2
    echo "       Build with: apptainer pull docker://nvcr.io/nvidia/tritonserver:25.07-trtllm-python-py3" >&2
    return 1
  fi

  # Build trtllm-serve arg string
  local serve_args="${TRTLLM_MODEL} \
    --host 0.0.0.0 \
    --port ${TRTLLM_PORT} \
    --tp_size ${TENSOR_PARALLEL_SIZE} \
    --kv_cache_free_gpu_memory_fraction ${KV_CACHE_FREE_GPU_MEM_FRACTION} \
    --max_seq_len ${MAX_SEQ_LEN} \
    --trust_remote_code"

  if [[ -n "${TRTLLM_BACKEND}" ]]; then
    serve_args="${serve_args} --backend ${TRTLLM_BACKEND}"
  fi

  echo "Launching TRT-LLM server..."
  echo "  Model:           ${TRTLLM_MODEL}"
  echo "  Port:            ${TRTLLM_PORT}"
  echo "  TP size:         ${TENSOR_PARALLEL_SIZE}"
  echo "  Backend:         ${TRTLLM_BACKEND:-<default (cpp)>}"
  echo "  KV cache mem:    ${KV_CACHE_FREE_GPU_MEM_FRACTION}"
  echo "  Max seq len:     ${MAX_SEQ_LEN}"

  if [[ -n "${RUN_LOG_DIR:-}" ]]; then
    apptainer exec --nv \
      -B "${HF_CACHE}:/root/.cache/huggingface" \
      "${TRTLLM_IMAGE}" \
      bash -lc "trtllm-serve ${serve_args}" \
      > "${RUN_LOG_DIR}/trtllm.log" 2>&1 &
  else
    apptainer exec --nv \
      -B "${HF_CACHE}:/root/.cache/huggingface" \
      "${TRTLLM_IMAGE}" \
      bash -lc "trtllm-serve ${serve_args}" &
  fi
  trtllm_pid=$!
}

wait_for_trtllm() {
  # TRT-LLM takes longer than vLLM to start: JIT kernel compilation on first
  # run adds several minutes on top of model load time.
  local max_wait_s="${1:-1800}"
  local elapsed=0
  echo "Waiting for TRT-LLM readiness on /v1/models (max ${max_wait_s}s) ..."
  until curl -sf "http://127.0.0.1:${TRTLLM_PORT}/v1/models" >/dev/null 2>&1; do
    # Bail out early if the server process has already died
    if ! kill -0 "${trtllm_pid}" 2>/dev/null; then
      echo "ERROR: TRT-LLM process (pid=${trtllm_pid}) exited before becoming ready." >&2
      [[ -n "${RUN_LOG_DIR:-}" ]] && echo "See ${RUN_LOG_DIR}/trtllm.log" >&2
      return 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    if (( elapsed >= max_wait_s )); then
      echo "ERROR: TRT-LLM did not become ready within ${max_wait_s}s" >&2
      [[ -n "${RUN_LOG_DIR:-}" ]] && echo "See ${RUN_LOG_DIR}/trtllm.log" >&2
      return 1
    fi
  done
  echo "TRT-LLM is ready."
}

run_pipeline() {
  if [[ -z "${APPTAINER_IMAGE:-}" ]] || [[ ! -f "${APPTAINER_IMAGE}" ]]; then
    echo "ERROR: APPTAINER_IMAGE must be set and point to an existing SIF." >&2
    return 1
  fi
  echo "Starting pipeline..."
  # The querier's _resolve_served_model_name() auto-detects whether the server
  # uses a derived name (vLLM) or the raw HF path (TRT-LLM), so no
  # served_model_name override is needed here.
  local pipeline_cmd="HYDRA_FULL_ERROR=1 python3 run_llm_labeling_pipeline.py \
    --config-name '${CONFIG_NAME}' \
    +image_dataset='${IMAGE_DATASET}' \
    ++dataset_generator.llmPersistenceQuerier.llmQuerier.use_vllm_server=True \
    ++dataset_generator.llmPersistenceQuerier.llmQuerier.vllm_url_root=http://localhost \
    ++dataset_generator.llmPersistenceQuerier.llmQuerier.vllm_port=${TRTLLM_PORT} \
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

cleanup_trtllm() {
  if [[ -n "${trtllm_pid:-}" ]] && kill -0 "${trtllm_pid}" 2>/dev/null; then
    echo "Stopping TRT-LLM server (pid=${trtllm_pid})"
    kill "${trtllm_pid}" || true
    wait "${trtllm_pid}" || true
  fi
}

# -----------------------------------------------------------------------------
# Standalone mode: run full flow (for idev: ./run_trtllm_and_pipeline.sh)
# -----------------------------------------------------------------------------
main() {
  if [[ -z "${TRTLLM_IMAGE:-}" ]] || [[ ! -f "${TRTLLM_IMAGE}" ]]; then
    echo "ERROR: Set TRTLLM_IMAGE to your TRT-LLM SIF path." >&2
    echo "       apptainer pull docker://nvcr.io/nvidia/tritonserver:25.07-trtllm-python-py3" >&2
    exit 1
  fi
  if [[ -z "${APPTAINER_IMAGE:-}" ]] || [[ ! -f "${APPTAINER_IMAGE}" ]]; then
    echo "ERROR: Set APPTAINER_IMAGE to your SPF project SIF path." >&2
    exit 1
  fi
  start_trtllm_background
  trap cleanup_trtllm EXIT
  wait_for_trtllm
  run_pipeline
  echo "Pipeline completed successfully."
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
