#!/usr/bin/env bash
# Eval the day-count-trained vision_anchor_midpoint checkpoints on the
# 50/20/30 balanced test split. Sibling of
# sweep_eval_vision_anchor_midpoint_pucpr_split_50_50_balanced_train_balanced_eval.sh.

set -euo pipefail

EVAL_CFG="eval_config_vision_anchor_midpoint_pucpr_split_50_20_30_daycount_balanced_train_balanced_eval"
EXP_NAME="vision_anchor_midpoint_pucpr_split_50_20_30_daycount_balanced_train"
MODEL_ROOT="/workspace/data/output/perpetua"

SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30
SPLIT_METADATA="$SPLIT_DIR/split_metadata.json"
TEST_SIDECAR="${TEST_SIDECAR:-$SPLIT_DIR/test_from_2012-10-30.track_id_to_location_cell128.json}"

DEFAULT_DAYS=""
if [[ -f "$SPLIT_METADATA" ]]; then
    TRAIN_DAYS=$(python3 -c "import json; print(json.load(open('$SPLIT_METADATA'))['train_days'])")
    GRID="1 2 3 4 5 7 10 14 21 28"
    for n in $GRID; do (( n < TRAIN_DAYS )) && DEFAULT_DAYS="$DEFAULT_DAYS $n"; done
    DEFAULT_DAYS="$(echo "$DEFAULT_DAYS $TRAIN_DAYS" | xargs)"
fi
DAYS="${DAYS:-$DEFAULT_DAYS}"

SEEDS="0"
JOBS=1
FORCE_EVAL=""

OUT_CSV="${OUT_CSV:-/workspace/data/output/persistence_eval_outputs/sweep_eval_metrics_vision_anchor_midpoint_pucpr_split_50_20_30_daycount_balanced.csv}"

usage() {
  cat <<EOF
Usage: $0 <GPU_IDS> [--jobs N] [--days "..."] [--seeds "..."] [--force]

  GPU_IDS      Comma-separated GPU index list, e.g. 2 or 0,1,2,3.
  --jobs N     Max concurrent eval jobs (default: 1).
  --days       Space-separated day-count list (default: "$DAYS").
  --seeds      Space-separated seed list (default: "$SEEDS").
  --force      Re-run eval even if metric_results.json already exists for the cell.
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
    --days) DAYS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --force) FORCE_EVAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --jobs must be a positive integer." >&2; exit 1
fi
if [[ -z "$DAYS" ]]; then
  echo "Error: --days is empty." >&2; exit 1
fi

IFS=',' read -r -a GPU_POOL <<< "$GPU_IDS"
GPU_COUNT=${#GPU_POOL[@]}
if (( JOBS > GPU_COUNT )); then
  echo "WARNING: --jobs=$JOBS > #GPUs=$GPU_COUNT; capping JOBS to $GPU_COUNT." >&2
  JOBS=$GPU_COUNT
fi

mkdir -p "$(dirname "$OUT_CSV")"
if [[ ! -f "$OUT_CSV" ]]; then
  echo "days,seed,accuracy@0.5,balanced_accuracy@0.5,f1@0.5,precision@0.5,recall@0.5,tnr@0.5,existence_mae,eval_dir" > "$OUT_CSV"
fi

day_slug() {
  local n="$1"
  if [[ "$n" == *.* ]]; then echo "${n//./p}"; else echo "$n"; fi
}

run_eval_cell() {
  local gpu="$1" n="$2" seed="$3"
  local n_slug
  n_slug=$(day_slug "$n")
  # Hydra writes literal experiment.day_count into the model_dir name, so
  # fractional N may live under `days0.5_seed0` rather than `days0p5_seed0`.
  # Prefer the slug form; fall back to the literal form if it exists instead.
  if [[ ! -f "$MODEL_ROOT/$EXP_NAME/days${n_slug}_seed${seed}/best_model.pt" \
        && -f "$MODEL_ROOT/$EXP_NAME/days${n}_seed${seed}/best_model.pt" ]]; then
    n_slug="$n"
  fi
  local model_dir="$MODEL_ROOT/$EXP_NAME/days${n_slug}_seed${seed}"
  local ckpt="$model_dir/best_model.pt"

  if [[ ! -f "$ckpt" ]]; then
    echo "[skip-eval][gpu=$gpu] $ckpt missing — train this cell first."
    echo "$n,$seed,nan,nan,nan,nan,nan,nan,nan,MISSING_CKPT" >> "$OUT_CSV"
    return 0
  fi

  local existing_eval=""
  if [[ -z "$FORCE_EVAL" ]]; then
    existing_eval=$(find /workspace/data/output/persistence_eval_outputs \
      -path "*/${EXP_NAME}_days${n_slug}_seed${seed}/*/metric_results.json" \
      -newer "$ckpt" 2>/dev/null \
      | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)
  fi

  if [[ -n "$existing_eval" ]]; then
    echo "[skip-eval][gpu=$gpu] reusing $existing_eval; pass --force to re-eval."
  else
    echo "[$(date +%H:%M:%S)][gpu=$gpu] eval days=$n seed=$seed"
    CUDA_VISIBLE_DEVICES="$gpu" \
      python eval_persistence.py \
        --config-name="$EVAL_CFG" \
        experiment.day_count="$n" \
        experiment.subsample_fraction=1.0 \
        experiment.subsample_seed="$seed"
  fi

  local latest_eval
  latest_eval=$(find /workspace/data/output/persistence_eval_outputs \
    -path "*/${EXP_NAME}_days${n_slug}_seed${seed}/*/metric_results.json" \
    -newer "$ckpt" 2>/dev/null \
    | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)

  if [[ -n "$latest_eval" && -f "$latest_eval" ]]; then
    python -c "
import json
with open('$latest_eval') as f:
    metrics = {x['name']: x['value'] for x in json.load(f)}
fields = ['accuracy@0.5','balanced_accuracy@0.5','f1@0.5','precision@0.5','recall@0.5','tnr@0.5','existence_mae']
row = ['$n', '$seed'] + [f\"{metrics.get(k, float('nan')):.6f}\" for k in fields] + ['$latest_eval']
print(','.join(row))
" >> "$OUT_CSV"
  else
    echo "WARNING: no metric_results.json found newer than $ckpt"
    echo "$n,$seed,nan,nan,nan,nan,nan,nan,nan,MISSING" >> "$OUT_CSV"
  fi
}

export -f run_eval_cell day_slug
export EVAL_CFG EXP_NAME MODEL_ROOT FORCE_EVAL OUT_CSV

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

echo "Vision daycount eval sweep (50/20/30 balanced): $idx cells, jobs=$JOBS, GPUs=${GPU_IDS}"
echo "  config:    $EVAL_CFG"
echo "  exp.name:  $EXP_NAME"
echo "  days:      $DAYS"
echo "  seeds:     $SEEDS"
echo "  out_csv:   $OUT_CSV"
echo

< "$TMP" xargs -P "$JOBS" -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r gpu n seed <<<"$1"; run_eval_cell "$gpu" "$n" "$seed"' _ {}

echo
echo "Vision daycount eval sweep (50/20/30 balanced) complete. Results: $OUT_CSV"
if command -v column >/dev/null 2>&1; then
  column -ts, "$OUT_CSV" | head -20
else
  head -20 "$OUT_CSV"
fi
