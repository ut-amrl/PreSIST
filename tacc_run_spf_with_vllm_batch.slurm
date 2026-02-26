#!/usr/bin/env bash
#SBATCH -J spf_job                        # Job name
#SBATCH -o logs/spf/spf_job.%j.out       # Stdout
#SBATCH -e logs/spf/spf_job.%j.err       # Stderr
#SBATCH -p amd-rtx                       # Partition / queue
#SBATCH -N 1                             # Number of nodes
#SBATCH -n 1                             # Number of tasks
#SBATCH -t 12:00:00                       # Wall time
#SBATCH -A IRI26004                      # Allocation

set -eo pipefail

echo "=== Batch wrapper ==="
echo "Job ID: $SLURM_JOB_ID"
echo

# You can override things here if you want:
# export VLLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
# export VLLM_PORT=9006
# export VLLM_TP=4

CODE_DIR="${CODE_DIR:-$WORK/workspaces/semantic-persistence-filter}"
cd "$CODE_DIR"

./tacc_run_spf_with_vllm.sh