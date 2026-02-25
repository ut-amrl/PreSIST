#!/usr/bin/env bash
set -euo pipefail

# Thin helper around idev for Stampede3 H100 bring-up.
#
# Usage (request interactive node; pass allocation or time if needed):
#   PROJECT=<allocation> ./scripts/stampede3/idev_h100_spf.sh
#   ./scripts/stampede3/idev_h100_spf.sh -A <allocation> -t 01:00:00
#
# Where code and data live (mirrors docker_start_spf.sh volumes):
#   Host PROJECT_ROOT  -> container /workspace/code/semantic-persistence-filter  (repo code)
#   Host SPF_DATA      -> container /workspace/data                               (datasets; optional)
#   Host HF_CACHE      -> container /opt/hf                                       (HuggingFace cache)
# Put code on $WORK or $SCRATCH; put data and HF cache on $SCRATCH (or $WORK).
#
# After you get a node, set variables then run the shared script (same as sbatch):
#
#   export APPTAINER_IMAGE=$SCRATCH/spf.sif
#   export PROJECT_ROOT=$PWD
#   export HF_CACHE=$SCRATCH/hf_cache
#   export SPF_DATA=$SCRATCH/data/semantic-persistence-filter
#   ./scripts/stampede3/run_vllm_and_pipeline.sh
#
# Example with overrides (config, dataset, vLLM port/model):
#
#   export CONFIG_NAME=init_dataset_gen_qwen3_30b_instruct_semantic_sam_minArea0p001
#   export IMAGE_DATASET=openloris_location_all
#   export VLLM_PORT=9001
#   export VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
#   ./scripts/stampede3/run_vllm_and_pipeline.sh
#
# Step-by-step (e.g. vLLM in one terminal, pipeline in another):
#   source ./scripts/stampede3/run_vllm_and_pipeline.sh
#   start_vllm_background
#   # In another shell (or after): wait_for_vllm && run_pipeline

PARTITION="${PARTITION:-h100}"
NODES="${NODES:-1}"
TASKS="${TASKS:-1}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
PROJECT="${PROJECT:-}"

if ! command -v idev >/dev/null 2>&1; then
  echo "ERROR: idev is not available in PATH. Run this on a Stampede3 login node."
  exit 1
fi

cmd=(idev -p "$PARTITION" -N "$NODES" -n "$TASKS" -t "$TIME_LIMIT")
if [[ -n "$PROJECT" ]]; then
  cmd+=(-A "$PROJECT")
fi

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

echo "Requesting interactive allocation with:"
printf '  %q' "${cmd[@]}"
echo
exec "${cmd[@]}"
