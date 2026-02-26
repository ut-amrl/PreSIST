#!/usr/bin/env bash
#SBATCH -J spf_job               # Job name
#SBATCH -o $SCRATCH/logs/spf/spf_job.%j.out        # Stdout (%j = job ID)
#SBATCH -e $SCRATCH/logs/spf/spf_job.%j.err        # Stderr
#SBATCH -p amd-rtx               # Partition / queue
#SBATCH -N 1                     # Number of nodes
#SBATCH -n 1                     # Number of tasks
#SBATCH -t 12:00:00              # Wall time (hh:mm:ss)
#SBATCH -A IRI26004       # <-- replace with your TACC project/allocation

#set -euo pipefail
set -eo pipefail

echo "=== Job info ==="
echo "  Date:   $(date)"
echo "  Host:   $(hostname)"
echo "  Job ID: $SLURM_JOB_ID"
echo

# 1) Load Apptainer (needed by tacc_run_spf_container.sh)
module load tacc-apptainer

# 2) Paths: override defaults if needed (optional)
export IMAGE="${IMAGE:-$SCRATCH/containers/spf_latest.sif}"
export CODE_DIR="${CODE_DIR:-$WORK/workspaces/semantic-persistence-filter}"
export DATA_DIR="${DATA_DIR:-$SCRATCH/data/semantic-persistence-filter}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$SCRATCH/.cache/huggingface}"

cd "$CODE_DIR"

echo "Using:"
echo "  Image:    $IMAGE"
echo "  Code dir: $CODE_DIR"
echo "  Data dir: $DATA_DIR"
echo "  HF cache: $HF_CACHE_DIR"
echo

# 3) Ensure helper script is executable
chmod +x "$CODE_DIR/tacc_run_spf_container.sh"

# 4) Command to run *inside* the container
#    - cd into the mounted code dir
#    - HYDRA_FULL_ERROR=1
#    - your pipeline command
CONTAINER_CMD='cd /workspace/code/semantic-persistence-filter && HYDRA_FULL_ERROR=1 python3 run_llm_labeling_pipeline.py --config-name qwen3_vl_30b_thinking_v2_b_semantic_sam_minArea0p001 +image_dataset=openloris_location_all'

echo "Running container command:"
echo "  $CONTAINER_CMD"
echo

# 5) Call helper with that string argument
IMAGE="$IMAGE" CODE_DIR="$CODE_DIR" DATA_DIR="$DATA_DIR" HF_CACHE_DIR="$HF_CACHE_DIR" \
  "$CODE_DIR/tacc_run_spf_container.sh" "$CONTAINER_CMD"

EXIT_CODE=$?

echo
echo "Job finished with exit code: $EXIT_CODE"
echo "End time: $(date)"

exit $EXIT_CODE