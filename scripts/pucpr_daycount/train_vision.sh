#!/usr/bin/env bash
# Day-count train sweep for vision_anchor_midpoint on the 50/20/30 split.
# Sibling of sweep_vision_anchor_midpoint_pucpr_split_50_50_balanced_train.sh
# (fraction sweep).
#
# Day-count subsetting must be materialized first via
# generate_daycount_train_manifests.py.
#
# Companion eval sweep:
#   scripts/sweep_eval_vision_anchor_midpoint_pucpr_split_50_20_30_daycount_balanced_train_balanced_eval.sh
#
# Usage:
#   scripts/sweep_vision_anchor_midpoint_pucpr_split_50_20_30_daycount_balanced_train.sh <GPU_IDS> \
#       [--jobs N] [--days "..."] [--seeds "..."] [--force]

set -euo pipefail

SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30
DAYCOUNT_ROOT="${DAYCOUNT_ROOT:-$SPLIT_DIR/daycount}"
SPLIT_METADATA="$SPLIT_DIR/split_metadata.json"

TRAIN_CFG="pucpr_train_config_vision_anchor_midpoint_split_50_20_30_daycount_balanced_train"
SEEDS="0"
JOBS=1
FORCE_TRAIN=""

# Default DAYS grid (denser low end, capped at the split's train_days).
# Always includes the fractional 0.5-day point (median image timestamp of
# day 0); integer points are kept iff strictly less than train_days, plus
# train_days itself as the rightmost point.
DEFAULT_DAYS="0.5"
if [[ -f "$SPLIT_METADATA" ]]; then
    TRAIN_DAYS=$(python3 -c "import json; print(json.load(open('$SPLIT_METADATA'))['train_days'])")
    GRID="1 2 3 4 5 7 10 14 21 28"
    for n in $GRID; do (( n < TRAIN_DAYS )) && DEFAULT_DAYS="$DEFAULT_DAYS $n"; done
    DEFAULT_DAYS="$(echo "$DEFAULT_DAYS $TRAIN_DAYS" | xargs)"
fi
DAYS="${DAYS:-$DEFAULT_DAYS}"

usage() {
  cat <<EOF
Usage: $0 <GPU_IDS> [--jobs N] [--config CFG] [--days "..."] [--seeds "..."] [--force]

  GPU_IDS      Comma-separated GPU index list, e.g. 0 or 0,1,2,3.
  --jobs N     Max concurrent training jobs (default: 1).
  --config     Hydra --config-name (default: $TRAIN_CFG).
  --days       Space-separated day-count list (default: "$DAYS").
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
    --days) DAYS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --force) FORCE_TRAIN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --jobs must be a positive integer." >&2; exit 1
fi
if [[ -z "$DAYS" ]]; then
  echo "Error: --days is empty (split_metadata.json missing? pass --days explicitly)." >&2; exit 1
fi

IFS=',' read -r -a GPU_POOL <<< "$GPU_IDS"
GPU_COUNT=${#GPU_POOL[@]}
if (( JOBS > GPU_COUNT )); then
  echo "WARNING: --jobs=$JOBS > #GPUs=$GPU_COUNT; capping JOBS to $GPU_COUNT." >&2
  JOBS=$GPU_COUNT
fi

SCRIPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_YAML="$SCRIPT_REPO_ROOT/configs/train_confs/${TRAIN_CFG}.yaml"
EXP_NAME=$(awk '/^  name:[[:space:]]*/ {sub(/^  name:[[:space:]]*/, ""); print; exit}' "$TRAIN_YAML")
if [[ -z "$EXP_NAME" ]]; then
  echo "ERROR: could not parse experiment.name from $TRAIN_YAML" >&2; exit 1
fi
MODEL_ROOT="/workspace/data/output/perpetua"

# Day-token slug: integer N stays unpadded for model_dir (matching pre-existing
# layout like `days1_seed0`) and zero-padded for the daycount split subdir
# (matching the generator's integer-day layout). Fractional N uses the dot-free
# `0p5`-style slug from generate_daycount_train_manifests.py so both forms
# remain filesystem-safe.
day_slug() {
  local n="$1"
  if [[ "$n" == *.* ]]; then
    echo "${n//./p}"
  else
    echo "$n"
  fi
}
day_dir_slug() {
  local n="$1"
  if [[ "$n" == *.* ]]; then
    echo "${n//./p}"
  else
    printf '%04d' "$n"
  fi
}

resolve_train_manifest() {
    local n="$1"
    local subdir="$DAYCOUNT_ROOT/days_$(day_dir_slug "$n")"
    local hits=( "$subdir"/train_first_*.json )
    if (( ${#hits[@]} != 1 )) || [[ ! -f "${hits[0]}" ]]; then
        echo "ERROR: expected exactly one train_first_*.json in $subdir; got ${#hits[@]}." >&2
        return 2
    fi
    echo "${hits[0]}"
}

run_train_cell() {
  local gpu="$1" n="$2" seed="$3"
  local n_slug
  n_slug=$(day_slug "$n")
  local model_dir="$MODEL_ROOT/$EXP_NAME/days${n_slug}_seed${seed}"

  if [[ -z "$FORCE_TRAIN" && -f "$model_dir/training_metadata.json" ]]; then
    echo "[skip-train][gpu=$gpu] $model_dir already trained; pass --force to retrain."
    return 0
  fi

  local train_manifest
  train_manifest=$(resolve_train_manifest "$n") || return $?

  echo "[$(date +%H:%M:%S)][gpu=$gpu] train days=$n seed=$seed manifest=$train_manifest"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python train_persistence_models.py \
      --config-name="$TRAIN_CFG" \
      train_dataset.manifest_path="$train_manifest" \
      experiment.day_count="$n" \
      experiment.subsample_fraction=1.0 \
      experiment.subsample_seed="$seed"
}

export -f run_train_cell resolve_train_manifest day_slug day_dir_slug
export TRAIN_CFG EXP_NAME MODEL_ROOT FORCE_TRAIN DAYCOUNT_ROOT

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

idx=0
for n in $DAYS; do
  for seed in $SEEDS; do
    gpu=${GPU_POOL[$(( idx % GPU_COUNT ))]}
    printf '%s\t%s\t%s\n' "$gpu" "$n" "$seed" >> "$TMP"
    idx=$(( idx + 1 ))
  done
done

echo "Vision daycount train sweep (50/20/30 balanced): $idx cells, jobs=$JOBS, GPUs=${GPU_IDS}"
echo "  config:    $TRAIN_CFG"
echo "  exp.name:  $EXP_NAME"
echo "  days:      $DAYS"
echo "  seeds:     $SEEDS"
echo

< "$TMP" xargs -P "$JOBS" -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r gpu n seed <<<"$1"; run_train_cell "$gpu" "$n" "$seed"' _ {}

echo
echo "Vision daycount train sweep (50/20/30 balanced) complete."
