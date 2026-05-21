#!/bin/bash
# Fill out the (min_samples × p_m/p_f) grid for location-grouped variants at
# D ∈ {1, 2}. Eval = percentile_horizons. MIXTURE_FAST mode (params_range=[2],
# num_retrain_attempts=0). Reuses existing checkpoints (same exp.name
# convention as earlier sweeps), so already-trained cells skip training.
#
# Grid:
#   variant   ∈ {mixture_exp, mixture_lognormal, single_exp, single_lognormal} : location
#   D         ∈ {1, 2}
#   min_samples ∈ {default(=20), 5, 2(mixture) / 1(single)}
#   p_m = p_f ∈ {0.01, 0.05, 0.10}
#
# Intended for spf_perpetua_container.
set -euo pipefail
SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30
TEST_MANIFEST=$SPLIT_DIR/test_from_2012-10-30.json
TEST_SIDECAR=$SPLIT_DIR/test_from_2012-10-30.track_id_to_location_cell128.json
SEED=0
OUT_CSV=/workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_split_50_20_30_min_noise_grid_location.csv
mkdir -p "$(dirname "$OUT_CSV")"
[[ -f "$OUT_CSV" ]] || echo "trainer_kind,variant,grouping,days,p_m,p_f,min_samples,seed,accuracy@0.5,balanced_accuracy@0.5,f1@0.5,precision@0.5,recall@0.5,tnr@0.5,existence_mae,eval_dir" > "$OUT_CSV"

DAYS_LIST=(1 2)
NOISES=("0.01 0p01" "0.05 0p05" "0.10 0p10")
# (label, mixture_min, single_min)
MIN_CONFIGS=("default 20 20" "5 5 5" "low 2 1")

MIXTURE_VARIANTS="mixture_exp:location mixture_lognormal:location"
SINGLE_VARIANTS="single_exp:location single_lognormal:location"

for N in "${DAYS_LIST[@]}"; do
  TRAIN_MANIFEST=$(ls "$SPLIT_DIR/daycount/days_$(printf '%04d' "$N")"/train_first_*.json | head -1)
  for noise_entry in "${NOISES[@]}"; do
    read -r PMPF NSLUG <<< "$noise_entry"
    for min_entry in "${MIN_CONFIGS[@]}"; do
      read -r MIN_LABEL MIX_MIN SINGLE_MIN <<< "$min_entry"

      for KIND in mixture single; do
        if [[ "$KIND" == "mixture" ]]; then
          VARIANTS="$MIXTURE_VARIANTS"
          MINSAMPLES=$MIX_MIN
          # Match historical exp.name conventions to enable skip-train reuse:
          if [[ "$MIN_LABEL" == "default" ]]; then
            EXP_SUFFIX="_daycount_noise${NSLUG}"
            TRAIN_FLAGS="trainer.params_range=[2] trainer.num_retrain_attempts=0 trainer.p_m=${PMPF} trainer.p_f=${PMPF}"
          else
            EXP_SUFFIX="_daycount_noise${NSLUG}_mins${MINSAMPLES}"
            TRAIN_FLAGS="trainer.params_range=[2] trainer.num_retrain_attempts=0 trainer.min_mixture_samples=${MINSAMPLES} trainer.p_m=${PMPF} trainer.p_f=${PMPF}"
          fi
        else
          VARIANTS="$SINGLE_VARIANTS"
          MINSAMPLES=$SINGLE_MIN
          if [[ "$MIN_LABEL" == "default" ]]; then
            EXP_SUFFIX="_daycount_emk1_noise${NSLUG}"
            TRAIN_FLAGS="+trainer.fit_method=em_k1 +trainer.p_m=${PMPF} +trainer.p_f=${PMPF}"
          else
            EXP_SUFFIX="_daycount_emk1_noise${NSLUG}_mins${MINSAMPLES}"
            TRAIN_FLAGS="+trainer.fit_method=em_k1 +trainer.min_em_samples=${MINSAMPLES} +trainer.p_m=${PMPF} +trainer.p_f=${PMPF}"
          fi
        fi

        for VG in $VARIANTS; do
          DIST="${VG%:*}"
          GROUPING="${VG#*:}"
          TRAIN_CFG="train_config_perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30_daycount"
          EVAL_CFG="eval_config_perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30_daycount_balanced"
          EXP_NAME="perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30${EXP_SUFFIX}"
          MODEL_DIR="/workspace/data/output/perpetua/${EXP_NAME}/days${N}_seed${SEED}"

          echo "================================================================================================"
          echo "[$(date +%H:%M:%S)] $KIND  $DIST × $GROUPING  D=$N  p=$PMPF  min=$MINSAMPLES  exp=$EXP_NAME"
          echo "================================================================================================"

          if [[ -f "$MODEL_DIR/persistence_params.json" ]]; then
            echo "[skip-train] already trained."
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

          EXISTING_EVAL=$(find /workspace/data/output/persistence_eval_outputs \
            -path "*/${EXP_NAME}_days${N}_seed${SEED}/*evaluation_runs/survival_based__percentile_horizons/*/metric_results.json" \
            -newer "$MODEL_DIR/persistence_params.json" 2>/dev/null \
            | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)
          if [[ -n "$EXISTING_EVAL" ]]; then
            echo "[skip-eval] reusing $EXISTING_EVAL"
            LATEST_EVAL="$EXISTING_EVAL"
          else
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
          fi

          if [[ -n "$LATEST_EVAL" && -f "$LATEST_EVAL" ]]; then
            python -c "
import json
with open('$LATEST_EVAL') as f:
    metrics = {x['name']: x['value'] for x in json.load(f)}
fields = ['accuracy@0.5','balanced_accuracy@0.5','f1@0.5','precision@0.5','recall@0.5','tnr@0.5','existence_mae']
row = ['$KIND', '$DIST', '$GROUPING', '$N', '$PMPF', '$PMPF', '$MINSAMPLES', '$SEED'] + [f\"{metrics.get(k, float('nan')):.6f}\" for k in fields] + ['$LATEST_EVAL']
print(','.join(row))
" >> "$OUT_CSV"
          else
            echo "$KIND,$DIST,$GROUPING,$N,$PMPF,$PMPF,$MINSAMPLES,$SEED,nan,nan,nan,nan,nan,nan,nan,MISSING" >> "$OUT_CSV"
          fi
        done
      done
    done
  done
done
echo "Done. Results: $OUT_CSV"
