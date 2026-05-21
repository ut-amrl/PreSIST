#!/bin/bash
# One-shot: run the 4 single-component variants at N=2 with fit_method=em_k1
# under a parallel experiment.name suffix (`_em_k1`) so naive-MLE results
# stay intact for comparison. Intended to be run inside spf_perpetua_container.
set -euo pipefail
SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30
TRAIN_MANIFEST=$SPLIT_DIR/daycount/days_0002/train_first_2d_through_2012-09-12.json
TEST_MANIFEST=$SPLIT_DIR/test_from_2012-10-30.json
TEST_SIDECAR=$SPLIT_DIR/test_from_2012-10-30.track_id_to_location_cell128.json
N=2
SEED=0
OUT_CSV=/workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_split_50_20_30_daycount_balanced_em_k1.csv
mkdir -p "$(dirname "$OUT_CSV")"
[[ -f "$OUT_CSV" ]] || echo "variant,grouping,days,seed,fit_method,accuracy@0.5,balanced_accuracy@0.5,f1@0.5,precision@0.5,recall@0.5,tnr@0.5,existence_mae,eval_dir" > "$OUT_CSV"

for VG in single_exp:class single_exp:location single_lognormal:class single_lognormal:location; do
  DIST="${VG%:*}"
  GROUPING="${VG#*:}"
  TRAIN_CFG="train_config_perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30_daycount"
  EVAL_CFG="eval_config_perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30_daycount_balanced"
  EXP_NAME="perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30_daycount_em_k1"
  MODEL_DIR="/workspace/data/output/perpetua/${EXP_NAME}/days${N}_seed${SEED}"

  echo "============================================================"
  echo "[$(date +%H:%M:%S)] $DIST × $GROUPING  EM K=1  days=$N seed=$SEED"
  echo "============================================================"

  if [[ -f "$MODEL_DIR/persistence_params.json" ]]; then
    echo "[skip-train] $MODEL_DIR already trained."
  else
    python train_persistence_models.py \
      --config-name="$TRAIN_CFG" \
      grouping="$GROUPING" \
      train_dataset.cell_size_px=128 \
      train_dataset.manifest_path="$TRAIN_MANIFEST" \
      experiment.name="$EXP_NAME" \
      experiment.day_count="$N" \
      experiment.subsample_fraction=1.0 \
      experiment.subsample_seed="$SEED" \
      +trainer.fit_method=em_k1
  fi

  SIDECAR_OVERRIDE=""
  if [[ "$GROUPING" == "location" ]]; then
    SIDECAR_OVERRIDE="inferencer.track_id_to_location_file=$TEST_SIDECAR"
  fi

  # shellcheck disable=SC2086
  python eval_persistence.py \
    --config-name="$EVAL_CFG" \
    experiment.name="$EXP_NAME" \
    experiment.day_count="$N" \
    experiment.subsample_fraction=1.0 \
    experiment.subsample_seed="$SEED" \
    benchmark_dataset.manifest_path="$TEST_MANIFEST" \
    $SIDECAR_OVERRIDE

  LATEST_EVAL=$(find /workspace/data/output/persistence_eval_outputs \
    -path "*/${EXP_NAME}_days${N}_seed${SEED}/*/metric_results.json" \
    -newer "$MODEL_DIR/persistence_params.json" 2>/dev/null \
    | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)

  if [[ -n "$LATEST_EVAL" && -f "$LATEST_EVAL" ]]; then
    python -c "
import json
with open('$LATEST_EVAL') as f:
    metrics = {x['name']: x['value'] for x in json.load(f)}
fields = ['accuracy@0.5','balanced_accuracy@0.5','f1@0.5','precision@0.5','recall@0.5','tnr@0.5','existence_mae']
row = ['$DIST', '$GROUPING', '$N', '$SEED', 'em_k1'] + [f\"{metrics.get(k, float('nan')):.6f}\" for k in fields] + ['$LATEST_EVAL']
print(','.join(row))
" >> "$OUT_CSV"
  else
    echo "$DIST,$GROUPING,$N,$SEED,em_k1,nan,nan,nan,nan,nan,nan,nan,MISSING" >> "$OUT_CSV"
  fi
done

echo "Done. Results: $OUT_CSV"
