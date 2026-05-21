#!/usr/bin/env bash
# Evaluate VisionAnchorMidpoint checkpoints produced by
# scripts/sweep_vision_anchor_midpoint_pucpr_repaired_split_50_50.sh.
#
# For each (fraction, seed) cell, points the new
#   eval_config_vision_anchor_midpoint_pucpr_repaired_split_50_50
# config at the cell's `best_model.pt` and runs eval_persistence.py against the
# PUCPR_repaired_split 50/50 test split (same target as
# sweep_train_eval_pucpr_repaired_split_50_50.sh).
#
# Eval pipeline: VisionAnchorMidpointInferencer emits t50 -> single_exp survival
# (lam = t50 / ln(2)) -> survival-based predictor.
#
# Parallelism mirrors the trainer sweep:
#   - First positional arg is GPU_IDS (comma-separated).
#   - --jobs N runs up to N evals concurrently, each pinned to one GPU via
#     CUDA_VISIBLE_DEVICES (round-robin).
#
# Usage:
#   scripts/sweep_eval_vision_anchor_midpoint_pucpr_repaired_split_50_50.sh <GPU_IDS> \
#       [--jobs N] [--fractions "..."] [--seeds "..."] [--force]
#
# Examples:
#   # one GPU, sequential, default fractions, seed=0
#   scripts/sweep_eval_vision_anchor_midpoint_pucpr_repaired_split_50_50.sh 2 --seeds "0"
#
#   # four GPUs, two concurrent evals
#   scripts/sweep_eval_vision_anchor_midpoint_pucpr_repaired_split_50_50.sh 0,1,2,3 --jobs 2 --seeds "0"

set -euo pipefail

############################################
# Defaults — kept in sync with the trainer sweep so we evaluate exactly the
# cells that script trains.
############################################

EVAL_CFG="eval_config_vision_anchor_midpoint_pucpr_repaired_split_50_50"
EXP_NAME="vision_anchor_midpoint_pucpr_repaired_split_50_50"
MODEL_ROOT="/workspace/data/output/perpetua"

FRACTIONS="0.1 1.0 0.5 0.3 0.7 0.2 0.4 0.6 0.8 0.9"
SEEDS="0"
JOBS=1
FORCE_EVAL=""

# Test-side sidecar required by per-location inferencer configs. The vision
# inferencer doesn't read it, but eval_scenario / dataset wiring may. Mirrors
# sweep_train_eval_pucpr_repaired_split_50_50.sh.
SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split/splits/50_50
TEST_SIDECAR="${TEST_SIDECAR:-$SPLIT_DIR/test_from_2012-10-16.track_id_to_location_cell128.json}"

OUT_CSV="${OUT_CSV:-/workspace/data/output/persistence_eval_outputs/sweep_eval_metrics_vision_anchor_midpoint_pucpr_repaired_split_50_50.csv}"

############################################
# Arg parsing
############################################

usage() {
  cat <<EOF
Usage: $0 <GPU_IDS> [--jobs N] [--fractions "..."] [--seeds "..."] [--force]

  GPU_IDS      Comma-separated GPU index list, e.g. 2 or 0,1,2,3.
  --jobs N     Max concurrent eval jobs (default: 1).
  --fractions  Space-separated fraction list (default: "$FRACTIONS").
  --seeds      Space-separated seed list (default: "$SEEDS").
  --force      Re-run eval even if metric_results.json already exists for the
               cell.
EOF
}

if [ "$#" -lt 1 ]; then usage; exit 1; fi

GPU_IDS="$1"; shift
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Error: GPU_IDS must be comma-separated integers (e.g., 2 or 0,1,2)" >&2
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --fractions) FRACTIONS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --force) FORCE_EVAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --jobs must be a positive integer." >&2
  exit 1
fi

IFS=',' read -r -a GPU_POOL <<< "$GPU_IDS"
GPU_COUNT=${#GPU_POOL[@]}
if (( JOBS > GPU_COUNT )); then
  echo "WARNING: --jobs=$JOBS > #GPUs=$GPU_COUNT; capping JOBS to $GPU_COUNT." >&2
  JOBS=$GPU_COUNT
fi

mkdir -p "$(dirname "$OUT_CSV")"
if [[ ! -f "$OUT_CSV" ]]; then
  echo "fraction,seed,accuracy@0.5,balanced_accuracy@0.5,f1@0.5,precision@0.5,recall@0.5,tnr@0.5,existence_mae,eval_dir" > "$OUT_CSV"
fi

############################################
# Per-cell eval function
############################################

run_eval_cell() {
  local gpu="$1" fraction="$2" seed="$3"
  local model_dir="$MODEL_ROOT/$EXP_NAME/frac${fraction}_seed${seed}"
  local ckpt="$model_dir/best_model.pt"

  if [[ ! -f "$ckpt" ]]; then
    echo "[skip-eval][gpu=$gpu] $ckpt missing — train this cell first."
    echo "$fraction,$seed,nan,nan,nan,nan,nan,nan,nan,MISSING_CKPT" >> "$OUT_CSV"
    return 0
  fi

  # Skip if a metric_results.json already exists for this cell and is newer
  # than the checkpoint, unless --force.
  local existing_eval=""
  if [[ -z "$FORCE_EVAL" ]]; then
    existing_eval=$(find /workspace/data/output/persistence_eval_outputs \
      -path "*/${EXP_NAME}_frac${fraction}_seed${seed}/*/metric_results.json" \
      -newer "$ckpt" 2>/dev/null \
      | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)
  fi

  if [[ -n "$existing_eval" ]]; then
    echo "[skip-eval][gpu=$gpu] reusing $existing_eval; pass --force to re-eval."
  else
    echo "[$(date +%H:%M:%S)][gpu=$gpu] eval frac=$fraction seed=$seed"
    CUDA_VISIBLE_DEVICES="$gpu" \
      python eval_persistence.py \
        --config-name="$EVAL_CFG" \
        experiment.subsample_fraction="$fraction" \
        experiment.subsample_seed="$seed"
  fi

  local latest_eval
  latest_eval=$(find /workspace/data/output/persistence_eval_outputs \
    -path "*/${EXP_NAME}_frac${fraction}_seed${seed}/*/metric_results.json" \
    -newer "$ckpt" 2>/dev/null \
    | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)

  if [[ -n "$latest_eval" && -f "$latest_eval" ]]; then
    python -c "
import json
with open('$latest_eval') as f:
    metrics = {x['name']: x['value'] for x in json.load(f)}
fields = ['accuracy@0.5','balanced_accuracy@0.5','f1@0.5','precision@0.5','recall@0.5','tnr@0.5','existence_mae']
row = ['$fraction', '$seed'] + [f\"{metrics.get(k, float('nan')):.6f}\" for k in fields] + ['$latest_eval']
print(','.join(row))
" >> "$OUT_CSV"
  else
    echo "WARNING: no metric_results.json found newer than $ckpt"
    echo "$fraction,$seed,nan,nan,nan,nan,nan,nan,nan,MISSING" >> "$OUT_CSV"
  fi
}

export -f run_eval_cell
export EVAL_CFG EXP_NAME MODEL_ROOT FORCE_EVAL OUT_CSV

############################################
# Build cell list and dispatch
############################################

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

echo "Eval sweep plan: $idx cells, jobs=$JOBS, GPUs=${GPU_IDS}"
echo "  config:    $EVAL_CFG"
echo "  exp.name:  $EXP_NAME"
echo "  fractions: $FRACTIONS"
echo "  seeds:     $SEEDS"
echo "  out_csv:   $OUT_CSV"
echo

< "$TMP" xargs -P "$JOBS" -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r gpu fraction seed <<<"$1"; run_eval_cell "$gpu" "$fraction" "$seed"' _ {}

echo
echo "Vision eval sweep complete. Results: $OUT_CSV"
column -ts, "$OUT_CSV" | head -20
