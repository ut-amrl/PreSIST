#!/usr/bin/env bash
#SBATCH -J spf-trtllm-pipeline
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o spf-trtllm-pipeline.%j.out
#SBATCH -e spf-trtllm-pipeline.%j.err

set -euo pipefail

# TRT-LLM equivalent of sbatch_h100_vllm_pipeline.sh.
#
# Two Apptainer SIFs are required (unlike the vLLM variant which uses one):
#   TRTLLM_IMAGE    - TRT-LLM Tritonserver SIF. Pull once with:
#                       apptainer pull docker://nvcr.io/nvidia/tritonserver:26.03-trtllm-python-py3
#                     This creates tritonserver_26.03-trtllm-python-py3.sif in the current dir.
#   APPTAINER_IMAGE - SPF project SIF (same as the vLLM variant).
#
# Where code and data live (mirrors docker_start_spf.sh volumes):
#   Host PROJECT_ROOT  -> container /workspace/code/semantic-persistence-filter  (repo code)
#   Host SPF_DATA      -> container /workspace/data                               (datasets; optional)
#   Host HF_CACHE      -> container /opt/hf                                       (HuggingFace cache)
#
# Setup steps:
#   1. Have a Stampede3 allocation (request via ACCESS / TACC Portal).
#   2. Clone/copy the repo to $WORK or $SCRATCH (PROJECT_ROOT).
#   3. Pull the TRT-LLM SIF (see above). Set TRTLLM_IMAGE.
#   4. Build or transfer the SPF SIF. Set APPTAINER_IMAGE.
#   5. Create HF cache dir (e.g. $SCRATCH/hf_cache). Optionally set SPF_DATA.
#   6. Submit from the project root, or set PROJECT_ROOT when submitting.
#
# Example usage:
#
#   cd $WORK/semantic-persistence-filter
#   export TRTLLM_IMAGE=$SCRATCH/tritonserver_26.03-trtllm-python-py3.sif
#   export APPTAINER_IMAGE=$SCRATCH/spf.sif
#   export PROJECT_ROOT=$WORK/semantic-persistence-filter
#   export HF_CACHE=$SCRATCH/hf_cache
#   export SPF_DATA=$SCRATCH/data/semantic-persistence-filter
#   export CONFIG_NAME=init_dataset_gen_qwen3_30b_instruct_semantic_sam_minArea0p001
#   export IMAGE_DATASET=openloris_location_all
#   export TRTLLM_MODEL=Qwen/Qwen3-30B-A3B-Thinking-2507
#   export TRTLLM_PORT=9001
#   sbatch scripts/stampede3/sbatch_h100_trtllm_pipeline.sh
#
# Notes:
#   - TRT-LLM takes longer to start than vLLM (JIT kernel compilation on first
#     run). The wait timeout in run_trtllm_and_pipeline.sh is 1800s (30 min).
#     Subsequent runs reuse the compiled kernels from the HF cache.
#   - trtllm-serve has no --served-model-name flag; it uses the raw HF model
#     path. The querier's _resolve_served_model_name() auto-detects this.
#   - VLMs and MoE models (e.g. Qwen3-30B-A3B) automatically get
#     --backend pytorch. Dense text-only models use the faster C++ backend.

if [[ -n "${PROJECT:-}" ]]; then
  echo "Using allocation project: ${PROJECT}"
fi

if [[ -z "${TRTLLM_IMAGE:-}" ]]; then
  echo "ERROR: TRTLLM_IMAGE is not set."
  echo "       apptainer pull docker://nvcr.io/nvidia/tritonserver:26.03-trtllm-python-py3"
  exit 1
fi
if [[ ! -f "${TRTLLM_IMAGE}" ]]; then
  echo "ERROR: TRTLLM_IMAGE does not exist: ${TRTLLM_IMAGE}"
  exit 1
fi

if [[ -z "${APPTAINER_IMAGE:-}" ]]; then
  echo "ERROR: APPTAINER_IMAGE is not set (SPF project SIF)."
  exit 1
fi
if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
  echo "ERROR: APPTAINER_IMAGE does not exist: ${APPTAINER_IMAGE}"
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
CONFIG_NAME="${CONFIG_NAME:-init_dataset_gen_qwen3_30b_instruct_semantic_sam_minArea0p001}"
IMAGE_DATASET="${IMAGE_DATASET:-openloris_location_all}"

TRTLLM_MODEL="${TRTLLM_MODEL:-Qwen/Qwen3-30B-A3B-Thinking-2507}"
TRTLLM_PORT="${TRTLLM_PORT:-9001}"
KV_CACHE_FREE_GPU_MEM_FRACTION="${KV_CACHE_FREE_GPU_MEM_FRACTION:-0.90}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"

HF_CACHE="${HF_CACHE:-${SCRATCH}/hf_cache}"
SPF_DATA="${SPF_DATA:-}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${SCRATCH}/spf_runs/${SLURM_JOB_ID}}"
PIPELINE_EXTRA_ARGS="${PIPELINE_EXTRA_ARGS:-}"

mkdir -p "${HF_CACHE}" "${RUN_LOG_DIR}"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "ERROR: PROJECT_ROOT not found: ${PROJECT_ROOT}"
  exit 1
fi

cd "${PROJECT_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_trtllm_and_pipeline.sh"

echo "==== SPF Stampede3 TRT-LLM Batch Run ===="
echo "JOB_ID:               ${SLURM_JOB_ID}"
echo "NODELIST:             ${SLURM_JOB_NODELIST}"
echo "PROJECT_ROOT:         ${PROJECT_ROOT}"
echo "TRTLLM_IMAGE:         ${TRTLLM_IMAGE}"
echo "APPTAINER_IMAGE:      ${APPTAINER_IMAGE}"
echo "RUN_LOG_DIR:          ${RUN_LOG_DIR}"
echo "TRTLLM_MODEL:         ${TRTLLM_MODEL}"
echo "TRTLLM_PORT:          ${TRTLLM_PORT}"
echo "TP_SIZE:              ${TENSOR_PARALLEL_SIZE}"
echo "BACKEND:              ${TRTLLM_BACKEND:-<default (cpp)>}"
echo "REASONING_PARSER:     ${TRTLLM_REASONING_PARSER:-<none>}"
echo "=========================================="

trap cleanup_trtllm EXIT
start_trtllm_background
wait_for_trtllm
run_pipeline

echo "Pipeline completed successfully."
echo "Logs:"
echo "  ${RUN_LOG_DIR}/trtllm.log"
echo "  ${RUN_LOG_DIR}/pipeline.log"
