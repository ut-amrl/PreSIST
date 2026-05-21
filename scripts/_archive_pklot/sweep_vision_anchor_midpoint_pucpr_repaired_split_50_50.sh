#!/usr/bin/env bash
# Sweep vision_anchor_midpoint training across (fraction, seed) cells against
# the PUCPR_repaired_split 50/50 train manifest. Vision eval is wired separately
# from the trainer (see comment in
# persistence_eval/methods/training/vision_anchor_midpoint.py) so this script
# trains only — analogous to the Perpetua sweep_train_eval.sh but without the
# eval/CSV step.
#
# Parallelism:
#   - First positional arg is GPU_IDS (comma-separated, like start_vllm_server.sh).
#   - --jobs N runs up to N training jobs concurrently, each pinned to one
#     GPU from the pool via CUDA_VISIBLE_DEVICES (round-robin assignment).
#
# Usage:
#   scripts/sweep_vision_anchor_midpoint_pucpr_repaired_split_50_50.sh <GPU_IDS> \
#       [--jobs N] [--config CFG] [--fractions "..."] [--seeds "..."] [--force]
#
# Examples:
#   # one GPU, sequential
#   scripts/sweep_vision_anchor_midpoint_pucpr_repaired_split_50_50.sh 0
#
#   # four GPUs, two concurrent jobs (each gets one GPU; spare GPUs idle until
#   # a slot frees)
#   scripts/sweep_vision_anchor_midpoint_pucpr_repaired_split_50_50.sh 0,1,2,3 --jobs 2
#
#   # custom fraction sweep, single seed
#   scripts/sweep_vision_anchor_midpoint_pucpr_repaired_split_50_50.sh 0,1 \
#       --jobs 2 --fractions "0.1 0.5 1.0" --seeds "0"

set -euo pipefail

############################################
# Defaults
############################################

TRAIN_CFG="train_config_vision_anchor_midpoint_pucpr_repaired_split_50_50"
FRACTIONS="0.1 1.0 0.5 0.3 0.7 0.2 0.4 0.6 0.8 0.9"
SEEDS="0"
JOBS=1
FORCE_TRAIN=""

############################################
# Arg parsing
############################################

usage() {
  cat <<EOF
Usage: $0 <GPU_IDS> [--jobs N] [--config CFG] [--fractions "..."] [--seeds "..."] [--force]

  GPU_IDS      Comma-separated GPU index list, e.g. 0 or 0,1,2,3.
  --jobs N     Max concurrent training jobs (default: 1). Each job pins to one
               GPU from GPU_IDS via CUDA_VISIBLE_DEVICES.
  --config     Hydra --config-name to launch
               (default: $TRAIN_CFG).
  --fractions  Space-separated subsample_fraction list
               (default: "$FRACTIONS").
  --seeds      Space-separated subsample_seed list (default: "$SEEDS").
  --force      Retrain even if a model_dir already has training_metadata.json.
EOF
}

if [ "$#" -lt 1 ]; then usage; exit 1; fi

GPU_IDS="$1"; shift
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Error: GPU_IDS must be comma-separated integers (e.g., 0 or 0,1,2)" >&2
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --config) TRAIN_CFG="$2"; shift 2 ;;
    --fractions) FRACTIONS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --force) FORCE_TRAIN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --jobs must be a positive integer." >&2
  exit 1
fi

# Build GPU pool as bash array.
IFS=',' read -r -a GPU_POOL <<< "$GPU_IDS"
GPU_COUNT=${#GPU_POOL[@]}

if (( JOBS > GPU_COUNT )); then
  echo "WARNING: --jobs=$JOBS > #GPUs=$GPU_COUNT; multiple jobs would share a GPU." >&2
  echo "         Capping JOBS to $GPU_COUNT." >&2
  JOBS=$GPU_COUNT
fi

############################################
# Per-cell train function
############################################

# Reads experiment.name from the chosen Hydra config so MODEL_DIR matches what
# train_persistence_models.py will write to. Saves us from hardcoding the name.
EXP_NAME=$(python -c "
from hydra import compose, initialize_config_dir
import os
with initialize_config_dir(config_dir=os.path.abspath('configs/train_confs'), version_base=None):
    cfg = compose(config_name='$TRAIN_CFG')
print(cfg.experiment.name)
")
MODEL_ROOT="/workspace/data/output/perpetua"

run_cell() {
  local gpu="$1" fraction="$2" seed="$3"
  local model_dir="$MODEL_ROOT/$EXP_NAME/frac${fraction}_seed${seed}"

  if [[ -f "$model_dir/training_metadata.json" && -z "$FORCE_TRAIN" ]]; then
    echo "[skip-train][gpu=$gpu] $model_dir already trained; pass --force to retrain."
    return 0
  fi

  echo "[$(date +%H:%M:%S)][gpu=$gpu] train $TRAIN_CFG frac=$fraction seed=$seed"

  CUDA_VISIBLE_DEVICES="$gpu" \
    python train_persistence_models.py \
      --config-name="$TRAIN_CFG" \
      experiment.subsample_fraction="$fraction" \
      experiment.subsample_seed="$seed"
}

export -f run_cell
export TRAIN_CFG EXP_NAME MODEL_ROOT FORCE_TRAIN

############################################
# Build cell list and dispatch with bounded parallelism
############################################

# Generate the (fraction, seed) cell list, then round-robin a GPU from the pool
# to each cell. xargs -P bounds concurrency to $JOBS.
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

idx=0
for fraction in $FRACTIONS; do
  for seed in $SEEDS; do
    gpu=${GPU_POOL[$(( idx % GPU_COUNT ))]}
    printf '%s\t%s\t%s\n' "$gpu" "$fraction" "$seed" >> "$TMP"
    idx=$(( idx + 1 ))
  done
done

echo "Sweep plan: $idx cells, jobs=$JOBS, GPUs=${GPU_IDS}"
echo "  config:    $TRAIN_CFG"
echo "  exp.name:  $EXP_NAME"
echo "  fractions: $FRACTIONS"
echo "  seeds:     $SEEDS"
echo

# xargs -P $JOBS executes run_cell on each line in parallel.
< "$TMP" xargs -P "$JOBS" -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r gpu fraction seed <<<"$1"; run_cell "$gpu" "$fraction" "$seed"' _ {}

echo
echo "Vision sweep complete."
