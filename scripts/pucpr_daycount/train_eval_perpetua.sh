#!/bin/bash
# Day-count sweep at p_m=p_f=0.01, min_samples=5, percentile_horizons, for
# all 8 perpetua variants (4 mixture + 4 single em_k1) × {class, location}.
# Mixtures use full select-and-refine (params_range=[2,3,4], num_retrain=5);
# singles use fit_method=em_k1.
#
# Naming chosen to reuse existing checkpoints where available:
#   - singles match `_daycount_emk1_noise0p01_mins5` (covers D=1, D=2 from
#     the prior lowmin sweep — will skip-train + skip-eval those).
#   - mixtures use a fresh `_daycount_fullK_mins5_p0p01` suffix (no prior
#     full-K runs at this name).
#
# DAYS list: standard day-count grid (1..35).
# Intended for spf_perpetua_container.
#
# Env overrides for sub-sweeps:
#   DAYS="2 7 35"           — restrict to these day-counts
#   KINDS="single"          — only single (em_k1), skip mixtures
#   KINDS="mixture"         — only mixtures, skip singles
#   FORCE_TRAIN=1           — retrain even if persistence_params.json exists
#   FORCE_EVAL=1            — re-eval even if a fresh metric_results.json exists
#   EXP_TAG=v2              — append "_v2" to experiment.name and model_dir so a
#                             second training run (e.g., after a trainer bugfix)
#                             lands beside the original instead of overwriting.
#                             Default OUT_CSV also gets the tag suffix so the
#                             two sweeps' metric rows live in separate CSVs.
#   JOBS=4                  — run up to N cells in parallel (default 1, fully
#                             serial like the original). Each parallel cell
#                             redirects stdout/stderr to a per-cell log under
#                             $LOG_DIR; failures echo a [FAIL] banner to the
#                             driver's stderr. JOBS=1 keeps the original
#                             behavior bit-for-bit (cell output goes straight
#                             to the driver's stdout).
#   LOG_DIR=<path>          — destination for per-cell logs when JOBS>1.
#                             Default: $(dirname OUT_CSV)/_oneshot_logs/
set -euo pipefail
SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30
TEST_MANIFEST=$SPLIT_DIR/test_from_2012-10-30.json
TEST_SIDECAR=$SPLIT_DIR/test_from_2012-10-30.track_id_to_location_cell128.json
SEED=0
PMPF=0.01
NSLUG=0p01
MINSAMPLES=5
# Optional version tag (e.g. EXP_TAG=v2) appended to experiment.name and the
# default OUT_CSV so two trainings of the same hyperparameter set don't collide.
EXP_TAG="${EXP_TAG:-}"
TAG_SUFFIX="${EXP_TAG:+_${EXP_TAG}}"
OUT_CSV="${OUT_CSV:-/workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_split_50_20_30_daycount_p${NSLUG}_mins${MINSAMPLES}_all8${TAG_SUFFIX}.csv}"
mkdir -p "$(dirname "$OUT_CSV")"
[[ -f "$OUT_CSV" ]] || echo "trainer_kind,variant,grouping,days,p_m,p_f,min_samples,seed,K_post,n_pruned,accuracy@0.5,balanced_accuracy@0.5,f1@0.5,precision@0.5,recall@0.5,tnr@0.5,existence_mae,eval_dir" > "$OUT_CSV"

# Parallelism. JOBS=1 (default) runs cells serially and writes their output
# directly to the driver's stdout, matching the original behavior. JOBS>1
# backgrounds each cell, throttled by `wait -n`, and redirects its output to
# $LOG_DIR/<cell>.log so the driver's stdout stays readable.
JOBS="${JOBS:-1}"
LOG_DIR="${LOG_DIR:-$(dirname "$OUT_CSV")/_oneshot_logs}"
mkdir -p "$LOG_DIR"

# Env-driven overrides; default to the full sweep.
DEFAULT_DAYS="0.5 1 2 3 4 5 7 10 14 21 28 35"
DAYS_LIST=(${DAYS:-$DEFAULT_DAYS})
DEFAULT_KINDS="mixture single"
KINDS_LIST=(${KINDS:-$DEFAULT_KINDS})

MIXTURE_VARIANTS="mixture_exp:class mixture_exp:location mixture_lognormal:class mixture_lognormal:location"
SINGLE_VARIANTS="single_exp:class single_exp:location single_lognormal:class single_lognormal:location"

# Train + eval + CSV-append for one (KIND, VG, N) cell. Each call is fully
# self-contained: every path is derived from its three args. Idempotency is
# preserved because the skip-train / skip-eval guards key off MODEL_DIR (a
# function of EXP_NAME, hence of KIND/VG/N) — concurrent cells touch disjoint
# files. FORCE_TRAIN / FORCE_EVAL propagate via standard env inheritance.
run_cell() {
  local KIND="$1" VG="$2" N="$3"
  local DIST="${VG%:*}"
  local GROUPING="${VG#*:}"

  local EXP_SUFFIX TRAIN_FLAGS
  if [[ "$KIND" == "mixture" ]]; then
    EXP_SUFFIX="_daycount_fullK_mins${MINSAMPLES}_p${NSLUG}"
    TRAIN_FLAGS="trainer.params_range=[2,3,4] trainer.num_retrain_attempts=5 trainer.min_mixture_samples=${MINSAMPLES} trainer.p_m=${PMPF} trainer.p_f=${PMPF}"
  else
    EXP_SUFFIX="_daycount_emk1_noise${NSLUG}_mins${MINSAMPLES}"
    TRAIN_FLAGS="+trainer.fit_method=em_k1 +trainer.min_em_samples=${MINSAMPLES} +trainer.p_m=${PMPF} +trainer.p_f=${PMPF}"
  fi

  local TRAIN_CFG="pucpr_train_config_perpetua_${DIST}_${GROUPING}_split_50_20_30_daycount"
  local EVAL_CFG="pucpr_eval_config_perpetua_${DIST}_${GROUPING}_split_50_20_30_daycount_balanced"
  local EXP_NAME="perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30${EXP_SUFFIX}${TAG_SUFFIX}"
  # Day-token slug: integer N keeps the legacy zero-padded form (days_0001,
  # days0001) so previously-trained cells still match; fractional N (e.g. 0.5)
  # uses the dot-free form (days_0p5, days0p5) used by the generator's
  # fractional-day output.
  local N_SLUG
  if [[ "$N" == *.* ]]; then
    N_SLUG="${N//./p}"
  else
    N_SLUG="$N"
  fi
  local N_DIR
  if [[ "$N" == *.* ]]; then
    N_DIR="$N_SLUG"
  else
    N_DIR=$(printf '%04d' "$N")
  fi
  local MODEL_DIR="/workspace/data/output/perpetua/${EXP_NAME}/days${N_SLUG}_seed${SEED}"
  local TRAIN_MANIFEST
  TRAIN_MANIFEST=$(ls "$SPLIT_DIR/daycount/days_${N_DIR}"/train_first_*.json | head -1)

  echo "================================================================================================"
  echo "[$(date +%H:%M:%S)] $KIND  $DIST × $GROUPING  D=$N  p=$PMPF  min=$MINSAMPLES"
  echo "================================================================================================"

  if [[ -f "$MODEL_DIR/persistence_params.json" && -z "${FORCE_TRAIN:-}" ]]; then
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

  local K_POST N_PRUNED
  K_POST=$(python3 -c "import json; p=json.load(open('$MODEL_DIR/persistence_params.json')); print(p['__global__']['metadata']['n_components_fitted'])")
  N_PRUNED=$(python3 -c "import json; p=json.load(open('$MODEL_DIR/persistence_params.json')); print(p['__global__']['metadata']['n_pruned_components'])")

  local EXISTING_EVAL=""
  if [[ -z "${FORCE_EVAL:-}" && -z "${FORCE_TRAIN:-}" ]]; then
    EXISTING_EVAL=$(find /workspace/data/output/persistence_eval_outputs \
      -path "*/${EXP_NAME}_days${N_SLUG}_seed${SEED}/*evaluation_runs/survival_based__percentile_horizons/*/metric_results.json" \
      -newer "$MODEL_DIR/persistence_params.json" 2>/dev/null \
      | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)
  fi
  local LATEST_EVAL
  if [[ -n "$EXISTING_EVAL" ]]; then
    echo "[skip-eval] reusing $EXISTING_EVAL"
    LATEST_EVAL="$EXISTING_EVAL"
  else
    local SIDECAR_OVERRIDE=""
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
      -path "*/${EXP_NAME}_days${N_SLUG}_seed${SEED}/*evaluation_runs/survival_based__percentile_horizons/*/metric_results.json" \
      -newer "$MODEL_DIR/persistence_params.json" 2>/dev/null \
      | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)
  fi

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
}

# Throttle to at most $JOBS concurrent background jobs. `wait -n` returns the
# exit code of whichever job finished; we swallow it so a cell failure doesn't
# abort the sweep — the [FAIL] banner from the launcher subshell flags it for
# the user, and the (empty) CSV row makes the missing cell obvious downstream.
_throttle() {
  while (( $(jobs -rp | wc -l) >= JOBS )); do
    wait -n 2>/dev/null || true
  done
}

n_launched=0
for N in "${DAYS_LIST[@]}"; do
  for KIND in "${KINDS_LIST[@]}"; do
    if [[ "$KIND" == "mixture" ]]; then
      VARIANTS="$MIXTURE_VARIANTS"
    else
      VARIANTS="$SINGLE_VARIANTS"
    fi
    for VG in $VARIANTS; do
      if (( JOBS > 1 )); then
        DIST="${VG%:*}"
        GROUPING="${VG#*:}"
        LOGFILE="$LOG_DIR/${KIND}_${DIST}_${GROUPING}_days${N}.log"
        _throttle
        (
          if ! run_cell "$KIND" "$VG" "$N" > "$LOGFILE" 2>&1; then
            echo "[FAIL] $KIND $DIST × $GROUPING D=$N (see $LOGFILE)" >&2
          fi
        ) &
        n_launched=$((n_launched + 1))
      else
        run_cell "$KIND" "$VG" "$N"
      fi
    done
  done
done

if (( JOBS > 1 )); then
  wait
  echo
  echo "Done. Launched $n_launched cells across JOBS=$JOBS workers."
  echo "Per-cell logs: $LOG_DIR/"
  if grep -lE "^\[FAIL\]|Error|Traceback" "$LOG_DIR"/*.log 2>/dev/null | grep -q .; then
    echo "WARNING: failure markers found in:" >&2
    grep -lE "^\[FAIL\]|Error|Traceback" "$LOG_DIR"/*.log 2>/dev/null >&2
  fi
fi
echo "Done. Results: $OUT_CSV"
