#!/usr/bin/env bash
# Train vision_anchor_midpoint with the per-track balanced WeightedRandomSampler
# enabled (see configs/train_confs/trainer/vision_anchor_midpoint_balanced.yaml).
# Equivalent to sweep_vision_anchor_midpoint_pucpr_repaired_split_50_50.sh but:
#   - points at train_config_vision_anchor_midpoint_pucpr_split_50_50_balanced_train
#     (distinct experiment.name → distinct model_dir, no checkpoint collision)
#   - same arg/usage contract as the non-balanced script.
#
# Companion eval sweep:
#   scripts/sweep_eval_vision_anchor_midpoint_pucpr_split_50_50_balanced_train_balanced_eval.sh
#
# Usage:
#   scripts/sweep_vision_anchor_midpoint_pucpr_split_50_50_balanced_train.sh <GPU_IDS> \
#       [--jobs N] [--fractions "..."] [--seeds "..."] [--force]

set -euo pipefail

TRAIN_CFG="train_config_vision_anchor_midpoint_pucpr_split_50_50_balanced_train"
FRACTIONS="0.1 1.0 0.5 0.3 0.7 0.2 0.4 0.6 0.8 0.9"
SEEDS="0"
JOBS=1
FORCE_TRAIN=""

usage() {
  cat <<EOF
Usage: $0 <GPU_IDS> [--jobs N] [--config CFG] [--fractions "..."] [--seeds "..."] [--force]

  GPU_IDS      Comma-separated GPU index list, e.g. 0 or 0,1,2,3.
  --jobs N     Max concurrent training jobs (default: 1).
  --config     Hydra --config-name (default: $TRAIN_CFG).
  --fractions  Space-separated subsample_fraction list (default: "$FRACTIONS").
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
  echo "Error: --jobs must be a positive integer." >&2; exit 1
fi

IFS=',' read -r -a GPU_POOL <<< "$GPU_IDS"
GPU_COUNT=${#GPU_POOL[@]}
if (( JOBS > GPU_COUNT )); then
  echo "WARNING: --jobs=$JOBS > #GPUs=$GPU_COUNT; capping JOBS to $GPU_COUNT." >&2
  JOBS=$GPU_COUNT
fi

# Resolve experiment.name from the chosen config so the skip-train check
# points at the right model_dir.
SCRIPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_YAML="$SCRIPT_REPO_ROOT/configs/train_confs/${TRAIN_CFG}.yaml"
EXP_NAME=$(awk '/^  name:[[:space:]]*/ {sub(/^  name:[[:space:]]*/, ""); print; exit}' "$TRAIN_YAML")
if [[ -z "$EXP_NAME" ]]; then
  echo "ERROR: could not parse experiment.name from $TRAIN_YAML" >&2; exit 1
fi
MODEL_ROOT="/workspace/data/output/perpetua"

run_train_cell() {
  local gpu="$1" fraction="$2" seed="$3"
  local model_dir="$MODEL_ROOT/$EXP_NAME/frac${fraction}_seed${seed}"

  if [[ -z "$FORCE_TRAIN" && -f "$model_dir/training_metadata.json" ]]; then
    echo "[skip-train][gpu=$gpu] $model_dir already trained; pass --force to retrain."
    return 0
  fi

  echo "[$(date +%H:%M:%S)][gpu=$gpu] train frac=$fraction seed=$seed"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python train_persistence_models.py \
      --config-name="$TRAIN_CFG" \
      experiment.subsample_fraction="$fraction" \
      experiment.subsample_seed="$seed"
}

export -f run_train_cell
export TRAIN_CFG EXP_NAME MODEL_ROOT FORCE_TRAIN

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

echo "Vision train sweep plan (balanced-train, flanks-fixed): $idx cells, jobs=$JOBS, GPUs=${GPU_IDS}"
echo "  config:    $TRAIN_CFG"
echo "  exp.name:  $EXP_NAME"
echo "  fractions: $FRACTIONS"
echo "  seeds:     $SEEDS"
echo

< "$TMP" xargs -P "$JOBS" -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r gpu fraction seed <<<"$1"; run_train_cell "$gpu" "$fraction" "$seed"' _ {}

echo
echo "Vision train sweep (balanced-train, flanks-fixed) complete."
