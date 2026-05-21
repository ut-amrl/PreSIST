#!/bin/bash
# Sweep over absolute number of training tracks for the 4 SINGLE variants
# using fit_method=em_k1 (upstream-faithful K=1 EM, censoring-aware). 2 noise
# levels (p_m=p_f in {0.10, 0.05}) × 3 track counts (100, 200, 300) = 24 cells.
# Eval scenario: percentile_horizons (cfg default). Intended for
# spf_perpetua_container (EM K=1 imports jax via mixture trainer).
#
# Uses the same fraction trick as the mixture sweep:
# fraction = (n_tracks + 0.5) / 7333.
set -euo pipefail
SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30
TRAIN_MANIFEST=$SPLIT_DIR/train_through_2012-10-15.json
TOTAL_TRACKS=7333
TEST_MANIFEST=$SPLIT_DIR/test_from_2012-10-30.json
TEST_SIDECAR=$SPLIT_DIR/test_from_2012-10-30.track_id_to_location_cell128.json
SEED=0
OUT_CSV=/workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_split_50_20_30_track_sweep_single_emk1.csv
mkdir -p "$(dirname "$OUT_CSV")"
[[ -f "$OUT_CSV" ]] || echo "variant,grouping,n_tracks,p_m,p_f,seed,accuracy@0.5,balanced_accuracy@0.5,f1@0.5,precision@0.5,recall@0.5,tnr@0.5,existence_mae,eval_dir" > "$OUT_CSV"

NOISES=("0.10 0p10" "0.05 0p05")
TRACK_COUNTS=(100 200 300)
VARIANTS="single_exp:class single_exp:location single_lognormal:class single_lognormal:location"

for entry in "${NOISES[@]}"; do
  read -r PMPF NSLUG <<< "$entry"
  for N_TRACKS in "${TRACK_COUNTS[@]}"; do
    FRAC=$(python3 -c "print((${N_TRACKS} + 0.5) / ${TOTAL_TRACKS})")

    for VG in $VARIANTS; do
      DIST="${VG%:*}"
      GROUPING="${VG#*:}"
      TRAIN_CFG="train_config_perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30_daycount"
      EVAL_CFG="eval_config_perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30_daycount_balanced"
      EXP_NAME="perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30_tracks_emk1_p${NSLUG}"
      MODEL_DIR="/workspace/data/output/perpetua/${EXP_NAME}/n${N_TRACKS}_seed${SEED}"

      echo "==================================================================================="
      echo "[$(date +%H:%M:%S)] $DIST × $GROUPING  EM K=1  n_tracks=$N_TRACKS  p_m=p_f=$PMPF"
      echo "  frac=$FRAC  manifest=$TRAIN_MANIFEST"
      echo "==================================================================================="

      if [[ -f "$MODEL_DIR/persistence_params.json" ]]; then
        echo "[skip-train] $MODEL_DIR already trained."
      else
        python train_persistence_models.py \
          --config-name="$TRAIN_CFG" \
          grouping="$GROUPING" \
          train_dataset.cell_size_px=128 \
          train_dataset.manifest_path="$TRAIN_MANIFEST" \
          experiment.name="$EXP_NAME" \
          experiment.day_count=0 \
          experiment.subsample_fraction="$FRAC" \
          experiment.subsample_seed="$SEED" \
          paths.model_dir="$MODEL_DIR" \
          +trainer.fit_method=em_k1 \
          +trainer.p_m="$PMPF" \
          +trainer.p_f="$PMPF"
      fi

      SIDECAR_OVERRIDE=""
      if [[ "$GROUPING" == "location" ]]; then
        SIDECAR_OVERRIDE="inferencer.track_id_to_location_file=$TEST_SIDECAR"
      fi

      # shellcheck disable=SC2086
      python eval_persistence.py \
        --config-name="$EVAL_CFG" \
        experiment.name="$EXP_NAME" \
        experiment.day_count=0 \
        experiment.subsample_fraction="$FRAC" \
        experiment.subsample_seed="$SEED" \
        paths.model_dir="$MODEL_DIR" \
        inferencer.name="${EXP_NAME}_n${N_TRACKS}_seed${SEED}" \
        benchmark_dataset.manifest_path="$TEST_MANIFEST" \
        $SIDECAR_OVERRIDE

      LATEST_EVAL=$(find /workspace/data/output/persistence_eval_outputs \
        -path "*/${EXP_NAME}_n${N_TRACKS}_seed${SEED}/*/metric_results.json" \
        -newer "$MODEL_DIR/persistence_params.json" 2>/dev/null \
        | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)

      if [[ -n "$LATEST_EVAL" && -f "$LATEST_EVAL" ]]; then
        python -c "
import json
with open('$LATEST_EVAL') as f:
    metrics = {x['name']: x['value'] for x in json.load(f)}
fields = ['accuracy@0.5','balanced_accuracy@0.5','f1@0.5','precision@0.5','recall@0.5','tnr@0.5','existence_mae']
row = ['$DIST', '$GROUPING', '$N_TRACKS', '$PMPF', '$PMPF', '$SEED'] + [f\"{metrics.get(k, float('nan')):.6f}\" for k in fields] + ['$LATEST_EVAL']
print(','.join(row))
" >> "$OUT_CSV"
      else
        echo "$DIST,$GROUPING,$N_TRACKS,$PMPF,$PMPF,$SEED,nan,nan,nan,nan,nan,nan,nan,MISSING" >> "$OUT_CSV"
      fi
    done
  done
done
echo "Done. Results: $OUT_CSV"
