#!/usr/bin/env bash
#SBATCH -J spf-seg-desc                           # Job name
#SBATCH -o logs/spf/spf-seg-desc.%j.out          # Stdout
#SBATCH -e logs/spf/spf-seg-desc.%j.err          # Stderr
#SBATCH -p amd-rtx                               # Partition / queue
#SBATCH -N 1                                     # Number of nodes
#SBATCH -n 1                                     # Number of tasks
#SBATCH -t 12:00:00                               # Wall time
#SBATCH -A IRI26004                              # Allocation

set -eo pipefail

echo "=== Seg+Desc batch wrapper ==="
echo "Job ID: $SLURM_JOB_ID"
echo

# Override any defaults from tacc_run_spf_trtllm_dam_seg_desc.sh here:
# export TRTLLM_MODEL="Qwen/Qwen3-VL-8B-Thinking"
# export TRTLLM_PORT=9011
# export TRTLLM_TP=4
# export DAM_PORT=9014
# export DATASET_NAME=train_mix_0
# export TRTLLM_GPUS="0,1,2,3"
# export DAM_GPUS="4"
# export PIPELINE_GPUS="5,6,7"

CODE_DIR="${CODE_DIR:-$WORK/workspaces/semantic-persistence-filter}"
cd "$CODE_DIR"

./tacc_run_spf_trtllm_dam_seg_desc.sh
