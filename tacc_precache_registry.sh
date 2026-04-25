#!/usr/bin/env bash
set -eo pipefail

# Pre-populate the SQLite image registry for all datasets before launching
# multi-node labeling jobs. Run this once (single process, sequentially) so
# that all labeling workers open the registry read-only and avoid cross-node
# WAL locking issues on Lustre.
#
# Usage:
#   bash tacc_precache_registry.sh
#
# Override any variable via env:
#   CONFIG_NAME=my_config DATASETS="train_mix_0 train_mix_1" bash tacc_precache_registry.sh

echo "=== SPF registry pre-cache ==="
echo "  Date: $(date)"
echo "  Host: $(hostname)"
echo

############################################
# Modules
############################################

module load tacc-apptainer

############################################
# Paths (mirror main run script)
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

CONFIG_NAME="${CONFIG_NAME:-q3_30bt_lang_q3T_pU3_sam3b_maxOb_30_mConf_0p35_mA0p001_dd0p9_capQ38bTproc2_v3_iqr_only_behav}"
DATASETS="${DATASETS:-train_mix_0 train_mix_1 train_mix_2 train_mix_3 train_mix_4 train_mix_5 train_mix_6 train_mix_7 train_mix_8 train_mix_9 train_mix_10}"

mkdir -p \
  "${HF_CACHE_DIR}" \
  "${LOG_DIR}" \
  "${JOB_RUN_DIR}" \
  "${JOB_CACHE_DIR}/pycache" \
  "${JOB_TMP_DIR}"

export PYTHONPYCACHEPREFIX="${JOB_CACHE_DIR}/pycache"
export TMPDIR="${JOB_TMP_DIR}"
export TEMP="${JOB_TMP_DIR}"
export TMP="${JOB_TMP_DIR}"
export APPTAINERENV_PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}"
export APPTAINERENV_TMPDIR="${TMPDIR}"
export APPTAINERENV_TEMP="${TEMP}"
export APPTAINERENV_TMP="${TMP}"

unset TRANSFORMERS_CACHE

chmod +x "$CODE_DIR/tacc_run_spf_container.sh"

echo "Config:   $CONFIG_NAME"
echo "Datasets: $DATASETS"
echo "Logs:     $LOG_DIR"
echo

############################################
# Pre-cache each dataset sequentially
############################################

read -ra DATASET_ARR <<< "$DATASETS"
OVERALL_EXIT=0

for dataset in "${DATASET_ARR[@]}"; do
    log="${LOG_DIR}/precache_${dataset}_${JOB_ID_TAG}.out"
    echo "Pre-caching $dataset  →  $log"

    container_cmd="cd /workspace/code/semantic-persistence-filter && \
HYDRA_FULL_ERROR=1 python3 precache_registry.py \
  --config-name ${CONFIG_NAME} \
  +image_dataset=${dataset}"

    CUDA_VISIBLE_DEVICES="" \
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

    exit_code=$?
    if [ "$exit_code" -ne 0 ]; then
        echo "  ERROR: $dataset failed (exit $exit_code) — check $log"
        OVERALL_EXIT=1
    else
        echo "  OK: $dataset"
    fi
done

echo
if [ "$OVERALL_EXIT" -eq 0 ]; then
    echo "All datasets pre-cached. Registry is ready for read-only labeling workers."
else
    echo "Some datasets failed. Check logs before launching labeling jobs."
fi
echo "Done: $(date)"
exit $OVERALL_EXIT
