#!/usr/bin/env bash
# Single-node driver: apptainer-execs run_ablations.py on the 4 local
# GPUs with whatever config list the outer wrapper supplied.
#
# Inputs (env, set by outer wrapper):
#   CONFIGS_STR        space-separated Hydra config names to train
#
# Optional (env, with defaults):
#   IMAGE              apptainer .sif (default $SCRATCH/containers/spf_latest.sif)
#   CODE_DIR           project root ($WORK/workspaces/semantic-persistence-filter)
#   DATA_DIR           dataset root ($SCRATCH/data/semantic-persistence-filter)
#   HF_CACHE_DIR       HF cache root ($SCRATCH/.cache/huggingface)
#   JOB_ROOT           per-job working dir ($SCRATCH/jobs/spf-ablations/$SLURM_JOB_ID)
#   ABLATION_LOG_ROOT  where .done markers + per-run logs land ($JOB_ROOT/ablation_logs)
#   GPUS               comma-separated GPU list to pass to run_ablations.py (default 0,1,2,3)
#
# Args ($@):
#   Forwarded verbatim to run_ablations.py.

set -eo pipefail

############################################
# Sanity
############################################
if [ -z "${CONFIGS_STR:-}" ]; then
    echo "ERROR: CONFIGS_STR must be set (use an outer wrapper)." >&2
    exit 1
fi

############################################
# Paths
############################################
export IMAGE="${IMAGE:-$SCRATCH/containers/spf_latest.sif}"
export CODE_DIR="${CODE_DIR:-$WORK/workspaces/semantic-persistence-filter}"
export DATA_DIR="${DATA_DIR:-$SCRATCH/data/semantic-persistence-filter}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$SCRATCH/.cache/huggingface}"
export JOB_ROOT="${JOB_ROOT:-$SCRATCH/jobs/spf-ablations/${SLURM_JOB_ID:-interactive}}"
export ABLATION_LOG_ROOT="${ABLATION_LOG_ROOT:-$JOB_ROOT/ablation_logs}"
export GPUS="${GPUS:-0,1,2,3}"
mkdir -p "$JOB_ROOT" "$ABLATION_LOG_ROOT"

for p in "$IMAGE" "$CODE_DIR" "$DATA_DIR"; do
    if [ ! -e "$p" ]; then
        echo "ERROR: required path missing: $p" >&2
        exit 1
    fi
done

module load tacc-apptainer

############################################
# Tee subsequent stdout/stderr to a persistent driver log so apptainer /
# launcher events survive terminal disconnect.
############################################
read -ra CONFIGS <<< "$CONFIGS_STR"
DRIVER_LOG="${ABLATION_LOG_ROOT}/driver.$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$DRIVER_LOG") 2>&1

echo "=== TACC ablation driver ==="
echo "  Job:      ${SLURM_JOB_ID:-interactive}  host $(hostname)"
echo "  Image:    $IMAGE"
echo "  Code:     $CODE_DIR"
echo "  Data:     $DATA_DIR"
echo "  Log dir:  $ABLATION_LOG_ROOT"
echo "  GPUs:     $GPUS"
echo "  Configs:"
printf '    %s\n' "${CONFIGS[@]}"
echo "  Extra:    $*"
echo
nvidia-smi -L
echo

if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "no configs supplied; nothing to do."
    exit 0
fi

############################################
# Launch
############################################
apptainer exec --nv \
    --bind "${CODE_DIR}:${CODE_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${HF_CACHE_DIR}:${HF_CACHE_DIR}" \
    --bind "${ABLATION_LOG_ROOT}:${ABLATION_LOG_ROOT}" \
    --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    --env "HF_HOME=${HF_CACHE_DIR}" \
    --pwd "${CODE_DIR}" \
    "${IMAGE}" \
    python run_ablations.py \
        --gpus "$GPUS" \
        --log-dir "$ABLATION_LOG_ROOT" \
        --configs "${CONFIGS[@]}" \
        "$@"

echo "finished $(date)"
