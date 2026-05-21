#!/bin/bash
# Headline cells: D=35 (max for 50/20/30 split), location grouping,
# min_samples=5, p_m=p_f=0.01, percentile_horizons.
#
# 4 cells:
#   - mixture_exp:location      (full select-and-refine: K∈[2,3,4], refine=5)
#   - mixture_lognormal:location (full select-and-refine)
#   - single_exp:location        (fit_method=em_k1)
#   - single_lognormal:location  (fit_method=em_k1)
#
# Intended for spf_perpetua_container. Full select-and-refine mixtures are
# slow (~70 min/cell on 7333 tracks); singles are fast.
set -euo pipefail
SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30
TRAIN_MANIFEST=$SPLIT_DIR/daycount/days_0035/train_first_35d_through_2012-10-15.json
TEST_MANIFEST=$SPLIT_DIR/test_from_2012-10-30.json
TEST_SIDECAR=$SPLIT_DIR/test_from_2012-10-30.track_id_to_location_cell128.json
N=35
SEED=0
PMPF=0.01
NSLUG=0p01
MINSAMPLES=5
OUT_CSV=/workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_split_50_20_30_d35_location_mins${MINSAMPLES}_p${NSLUG}.csv
mkdir -p "$(dirname "$OUT_CSV")"
[[ -f "$OUT_CSV" ]] || echo "trainer_kind,variant,grouping,days,p_m,p_f,min_samples,seed,K_post,n_pruned,accuracy@0.5,balanced_accuracy@0.5,f1@0.5,precision@0.5,recall@0.5,tnr@0.5,existence_mae,eval_dir" > "$OUT_CSV"

for VG in mixture_exp:location mixture_lognormal:location single_exp:location single_lognormal:location; do
  DIST="${VG%:*}"
  GROUPING="${VG#*:}"
  TRAIN_CFG="train_config_perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30_daycount"
  EVAL_CFG="eval_config_perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30_daycount_balanced"

  if [[ "$DIST" == mixture_* ]]; then
    KIND="mixture"
    EXP_NAME="perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30_d35_fullK_mins${MINSAMPLES}_p${NSLUG}"
    TRAIN_FLAGS="trainer.params_range=[2,3,4] trainer.num_retrain_attempts=5 trainer.min_mixture_samples=${MINSAMPLES} trainer.p_m=${PMPF} trainer.p_f=${PMPF}"
  else
    KIND="single"
    EXP_NAME="perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30_d35_emk1_mins${MINSAMPLES}_p${NSLUG}"
    TRAIN_FLAGS="+trainer.fit_method=em_k1 +trainer.min_em_samples=${MINSAMPLES} +trainer.p_m=${PMPF} +trainer.p_f=${PMPF}"
  fi
  MODEL_DIR="/workspace/data/output/perpetua/${EXP_NAME}/days${N}_seed${SEED}"

  echo "================================================================================================"
  echo "[$(date +%H:%M:%S)] $KIND  $DIST × $GROUPING  D=$N  p=$PMPF  min_samples=$MINSAMPLES"
  echo "================================================================================================"

  if [[ -f "$MODEL_DIR/persistence_params.json" ]]; then
    echo "[skip-train] $MODEL_DIR already trained."
  else
    # shellcheck disable=SC2086
    python train_persistence_models.py \
      --config-name="$TRAIN_CFG" \
      grouping="$GROUPING" \
      train_dataset.cell_size_px=128 \
      train_dataset.manifest_path="$TRAIN_MANIFEST" \
      experiment.name="$EXP_NAME" \
      experiment.day_count="$N" \
      experiment.subsample_fraction=1.0 \
      experiment.subsample_seed="$SEED" \
      $TRAIN_FLAGS
  fi

  K_POST=$(python3 -c "import json; p=json.load(open('$MODEL_DIR/persistence_params.json')); print(p['__global__']['metadata']['n_components_fitted'])")
  N_PRUNED=$(python3 -c "import json; p=json.load(open('$MODEL_DIR/persistence_params.json')); print(p['__global__']['metadata']['n_pruned_components'])")

  # shellcheck disable=SC2086
  python eval_persistence.py \
    --config-name="$EVAL_CFG" \
    experiment.name="$EXP_NAME" \
    experiment.day_count="$N" \
    experiment.subsample_fraction=1.0 \
    experiment.subsample_seed="$SEED" \
    benchmark_dataset.manifest_path="$TEST_MANIFEST" \
    inferencer.track_id_to_location_file="$TEST_SIDECAR"

  LATEST_EVAL=$(find /workspace/data/output/persistence_eval_outputs \
    -path "*/${EXP_NAME}_days${N}_seed${SEED}/*evaluation_runs/survival_based__percentile_horizons/*/metric_results.json" \
    -newer "$MODEL_DIR/persistence_params.json" 2>/dev/null \
    | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)

  if [[ -n "$LATEST_EVAL" && -f "$LATEST_EVAL" ]]; then
    python -c "
import json
with open('$LATEST_EVAL') as f:
    metrics = {x['name']: x['value'] for x in json.load(f)}
fields = ['accuracy@0.5','balanced_accuracy@0.5','f1@0.5','precision@0.5','recall@0.5','tnr@0.5','existence_mae']
row = ['$KIND', '$DIST', '$GROUPING', '$N', '$PMPF', '$PMPF', '$MINSAMPLES', '$SEED', '$K_POST', '$N_PRUNED'] + [f\"{metrics.get(k, float('nan')):.6f}\" for k in fields] + ['$LATEST_EVAL']
print(','.join(row))
" >> "$OUT_CSV"
  else
    echo "$KIND,$DIST,$GROUPING,$N,$PMPF,$PMPF,$MINSAMPLES,$SEED,$K_POST,$N_PRUNED,nan,nan,nan,nan,nan,nan,nan,MISSING" >> "$OUT_CSV"
  fi
done

echo "Done. Results: $OUT_CSV"
