#!/usr/bin/env bash
#SBATCH -J spf-vllm-pipeline
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o spf-vllm-pipeline.%j.out
#SBATCH -e spf-vllm-pipeline.%j.err

set -euo pipefail

# vLLM port and use_vllm_server are set in this script (CLI overrides), not in config YAML.
# YAML configs define model/sampling/behavior; runtime server/port stay in scripts.
#
# Where code and data live (mirrors docker_start_spf.sh volumes):
#   Host PROJECT_ROOT  -> container /workspace/code/semantic-persistence-filter  (repo code)
#   Host SPF_DATA      -> container /workspace/data                               (datasets; optional)
#   Host HF_CACHE      -> container /opt/hf                                       (HuggingFace cache)
# Store code on $WORK or $SCRATCH; store data and HF cache on $SCRATCH (or $WORK). Use SPF_DATA
# so the container sees /workspace/data like in Docker; your config/registry paths should point
# there or under PROJECT_ROOT as appropriate.
#
# Setup steps required for this to work:
#   1. Have a Stampede3 allocation (request via ACCESS / TACC Portal).
#   2. Clone/copy the repo to $WORK or $SCRATCH (PROJECT_ROOT). Copy datasets to $SCRATCH (or $WORK).
#   3. Build an Apptainer SIF from the project Dockerfile, or transfer a SIF. Set APPTAINER_IMAGE.
#   4. Create HF cache dir (e.g. $SCRATCH/hf_cache). Optionally set SPF_DATA to your data dir.
#   5. Ensure config model_name matches vLLM --model and served-model-name (script uses basename__dtype).
#   6. Submit from the project root, or set PROJECT_ROOT when submitting.
#
# Example usage (set variables then submit; optional vars can be omitted to use defaults):
#
#   cd $WORK/semantic-persistence-filter
#   export APPTAINER_IMAGE=$SCRATCH/spf.sif
#   export PROJECT=$WORK  # or your allocation id if you have multiple
#   export PROJECT_ROOT=$WORK/semantic-persistence-filter
#   export HF_CACHE=$SCRATCH/hf_cache
#   export SPF_DATA=$SCRATCH/data/semantic-persistence-filter
#   export CONFIG_NAME=init_dataset_gen_qwen3_30b_instruct_semantic_sam_minArea0p001
#   export IMAGE_DATASET=openloris_location_all
#   export VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
#   export VLLM_PORT=9001
#   sbatch scripts/stampede3/sbatch_h100_vllm_pipeline.sh
#
# Example with overrides (different config, dataset, extra Hydra args):
#
#   export CONFIG_NAME=other_config
#   export IMAGE_DATASET=other_dataset
#   export PIPELINE_EXTRA_ARGS='++dataloader.num_workers=4'
#   sbatch scripts/stampede3/sbatch_h100_vllm_pipeline.sh

if [[ -n "${PROJECT:-}" ]]; then
  echo "Using allocation project: ${PROJECT}"
fi

if [[ -z "${APPTAINER_IMAGE:-}" ]]; then
  echo "ERROR: APPTAINER_IMAGE is not set."
  echo "Set it to a SIF that already contains your project runtime."
  exit 1
fi

if [[ ! -f "${APPTAINER_IMAGE}" ]]; then
  echo "ERROR: APPTAINER_IMAGE does not exist: ${APPTAINER_IMAGE}"
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
CONFIG_NAME="${CONFIG_NAME:-init_dataset_gen_qwen3_30b_instruct_semantic_sam_minArea0p001}"
IMAGE_DATASET="${IMAGE_DATASET:-openloris_location_all}"

VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_PORT="${VLLM_PORT:-9001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

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
source "${SCRIPT_DIR}/run_vllm_and_pipeline.sh"

echo "==== SPF Stampede3 Batch Run ===="
echo "JOB_ID:               ${SLURM_JOB_ID}"
echo "NODELIST:             ${SLURM_JOB_NODELIST}"
echo "PROJECT_ROOT:         ${PROJECT_ROOT}"
echo "APPTAINER_IMAGE:      ${APPTAINER_IMAGE}"
echo "RUN_LOG_DIR:          ${RUN_LOG_DIR}"
echo "VLLM_MODEL:           ${VLLM_MODEL}"
echo "SERVED_MODEL_NAME:    ${SERVED_MODEL_NAME}"
echo "VLLM_PORT:            ${VLLM_PORT}"
echo "TP_SIZE:              ${TENSOR_PARALLEL_SIZE}"
echo "==============================="

trap cleanup_vllm EXIT
start_vllm_background
wait_for_vllm
run_pipeline

echo "Pipeline completed successfully."
echo "Logs:"
echo "  ${RUN_LOG_DIR}/vllm.log"
echo "  ${RUN_LOG_DIR}/pipeline.log"
