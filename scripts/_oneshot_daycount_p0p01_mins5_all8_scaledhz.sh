#!/bin/bash
# Eval-only companion to _oneshot_daycount_p0p01_mins5_all8.sh.
#
# Runs eval_persistence.py against the SAME 8 perpetua model checkpoints
# (4 mixture + 4 single em_k1) × {class, location} × {D = 1..35} that the
# percentile-horizons training+eval sweep produces. Defaults to the
# percentile_scaled_horizons eval_scenario (percentile-derived horizons +
# scaled-horizon causal masking with recency_cap=1800 s); switch to plain
# scaled_horizon or any other scenario via EVAL_SCENARIO=. This script never
# trains — it only evaluates.
#
# Semantics:
#   - For each cell, looks for an already-trained persistence_params.json
#     under the SAME model_dir naming the percentile-horizons sweep uses.
#     If the model is missing → log "[skip-no-model]" and continue (cell
#     contributes no CSV row).
#   - Looks for an existing metric_results.json for the chosen eval_scenario
#     that is newer than the params file. If found → log "[skip-eval]" unless
#     FORCE_REEVAL=1.
#   - Otherwise runs eval and appends a row to the per-scenario sweep CSV.
#
# Env overrides:
#   EVAL_SCENARIO=<name>    — eval_scenario YAML name under
#                             configs/train_confs/eval_scenario/. Default:
#                             "percentile_scaled_horizons". The script keys
#                             all sentinel paths (CSV, LOG_DIR, eval-dir
#                             filter) off this name so multiple scenarios can
#                             coexist without colliding.
#   DAYS="2 7 35"           — restrict to these day-counts
#   KINDS="single"          — only single (em_k1), skip mixtures
#   KINDS="mixture"         — only mixtures, skip singles
#   FORCE_REEVAL=1          — re-run eval even if a fresh metric_results.json
#                             already exists (newer than persistence_params.json)
#   EXP_TAG=v2              — append "_v2" to experiment.name so this script
#                             targets the matching v2 namespace produced by the
#                             percentile-horizons sweep. Default OUT_CSV also
#                             gets the tag suffix.
#   JOBS=4                  — parallel cell workers (default 1, fully serial).
#                             Per-cell logs go to $LOG_DIR when JOBS>1.
#   LOG_DIR=<path>          — destination for per-cell logs (default:
#                             $(dirname OUT_CSV)/_oneshot_${EVAL_SCENARIO}_logs/).
#   OUT_CSV=<path>          — override aggregate sweep CSV.
#   EVAL_NOISE_PM=0.01      — override the persistence_filter.missed_detection_prob
#   EVAL_NOISE_PF=0.01      — override the persistence_filter.false_detection_prob
#                             Must be supplied together (or neither). When set,
#                             outputs land under a noise-tagged CSV / log dir so
#                             default-noise (config) results are not overwritten.
#                             Idempotency switches to a CSV-row-presence check
#                             (which doesn't confuse noise-varied eval timestamps
#                             with default ones under the same scenario subdir).
#   FILTER_OVERRIDE=name    — override the persistence_filter Hydra config
#                             (must match a YAML name under
#                             configs/train_confs/persistence_filter/). Useful
#                             for ablation runs that swap the filter wrapper
#                             without touching the eval_config defaults — e.g.
#                             FILTER_OVERRIDE=mixture_prior_pick_filter for
#                             "observations select component but no
#                             observation-conditioned likelihood correction".
#                             Outputs go to a filter-tagged CSV / log dir so
#                             default-filter results are preserved.
set -euo pipefail
SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30
TEST_MANIFEST=$SPLIT_DIR/test_from_2012-10-30.json
TEST_SIDECAR=$SPLIT_DIR/test_from_2012-10-30.track_id_to_location_cell128.json
SEED=0
PMPF=0.01
NSLUG=0p01
MINSAMPLES=5
EXP_TAG="${EXP_TAG:-}"
TAG_SUFFIX="${EXP_TAG:+_${EXP_TAG}}"

# Eval scenario — must match a YAML under configs/train_confs/eval_scenario/
# AND the `name:` field inside that YAML (the framework uses `name:` to build
# the on-disk path `evaluation_runs/survival_based__<name>/...`).
EVAL_SCENARIO="${EVAL_SCENARIO:-percentile_scaled_horizons}"

# Optional eval-time noise overrides. When set, both must be set together;
# outputs land under a noise-tagged namespace so default-config results are
# preserved untouched.
EVAL_NOISE_PM="${EVAL_NOISE_PM:-}"
EVAL_NOISE_PF="${EVAL_NOISE_PF:-}"
if [[ -n "$EVAL_NOISE_PM$EVAL_NOISE_PF" ]]; then
  if [[ -z "$EVAL_NOISE_PM" || -z "$EVAL_NOISE_PF" ]]; then
    echo "EVAL_NOISE_PM and EVAL_NOISE_PF must be set together" >&2
    exit 2
  fi
  NOISE_TAG="_pm$(printf "%s" "$EVAL_NOISE_PM" | tr '.' 'p')_pf$(printf "%s" "$EVAL_NOISE_PF" | tr '.' 'p')"
  NOISE_OVERRIDES="++persistence_filter.missed_detection_prob=${EVAL_NOISE_PM} ++persistence_filter.false_detection_prob=${EVAL_NOISE_PF}"
else
  NOISE_TAG=""
  NOISE_OVERRIDES=""
fi

# Optional persistence_filter override (e.g. mixture_prior_pick_filter for the
# ablation). When set, the eval CLI appends `persistence_filter=$FILTER_OVERRIDE`
# and outputs go to a filter-tagged namespace so the default-filter results
# aren't touched.
FILTER_OVERRIDE="${FILTER_OVERRIDE:-}"
if [[ -n "$FILTER_OVERRIDE" ]]; then
  FILTER_TAG="_filter${FILTER_OVERRIDE}"
  FILTER_OVERRIDES="persistence_filter=${FILTER_OVERRIDE}"
else
  FILTER_TAG=""
  FILTER_OVERRIDES=""
fi

# Per-scenario CSV so multiple eval_scenarios written by the same model
# checkpoint set live side-by-side under the same parent dir.
OUT_CSV="${OUT_CSV:-/workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_split_50_20_30_daycount_p${NSLUG}_mins${MINSAMPLES}_all8_${EVAL_SCENARIO}${TAG_SUFFIX}${NOISE_TAG}${FILTER_TAG}.csv}"
mkdir -p "$(dirname "$OUT_CSV")"
[[ -f "$OUT_CSV" ]] || echo "trainer_kind,variant,grouping,days,p_m,p_f,min_samples,seed,K_post,n_pruned,accuracy@0.5,balanced_accuracy@0.5,f1@0.5,precision@0.5,recall@0.5,tnr@0.5,existence_mae,eval_dir" > "$OUT_CSV"

JOBS="${JOBS:-1}"
LOG_DIR="${LOG_DIR:-$(dirname "$OUT_CSV")/_oneshot_${EVAL_SCENARIO}${NOISE_TAG}${FILTER_TAG}_logs}"
mkdir -p "$LOG_DIR"

DEFAULT_DAYS="1 2 3 4 5 7 10 14 21 28 35"
DAYS_LIST=(${DAYS:-$DEFAULT_DAYS})
DEFAULT_KINDS="mixture single"
KINDS_LIST=(${KINDS:-$DEFAULT_KINDS})

MIXTURE_VARIANTS="mixture_exp:class mixture_exp:location mixture_lognormal:class mixture_lognormal:location"
SINGLE_VARIANTS="single_exp:class single_exp:location single_lognormal:class single_lognormal:location"

# Eval-only worker for one cell. Returns:
#   0 — eval ran (or was skipped because a fresh result exists), CSV row appended.
#   0 + no CSV row — model not yet trained; logged as [skip-no-model].
#   1 — eval was attempted but failed (the parallel wrapper prints [FAIL]).
run_cell() {
  local KIND="$1" VG="$2" N="$3"
  local DIST="${VG%:*}"
  local GROUPING="${VG#*:}"
  # Slug rule matches the training script: fractional N becomes "0p5" so paths
  # line up with the trainer's days0p5_seed0 layout. If the slug-form model_dir
  # is missing but the literal dot-form exists (e.g. trained outside the
  # all8 script which writes the literal Hydra value), fall back to that.
  local N_SLUG
  if [[ "$N" == *.* ]]; then N_SLUG="${N//./p}"; else N_SLUG="$N"; fi
  local _EXP_SUFFIX_TMP
  if [[ "$KIND" == "mixture" ]]; then
    _EXP_SUFFIX_TMP="_daycount_fullK_mins${MINSAMPLES}_p${NSLUG}"
  else
    _EXP_SUFFIX_TMP="_daycount_emk1_noise${NSLUG}_mins${MINSAMPLES}"
  fi
  local _EXP_NAME_TMP="perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30${_EXP_SUFFIX_TMP}${TAG_SUFFIX}"
  if [[ ! -f "/workspace/data/output/perpetua/${_EXP_NAME_TMP}/days${N_SLUG}_seed${SEED}/persistence_params.json" \
        && -f "/workspace/data/output/perpetua/${_EXP_NAME_TMP}/days${N}_seed${SEED}/persistence_params.json" ]]; then
    N_SLUG="$N"
  fi

  local EXP_SUFFIX
  if [[ "$KIND" == "mixture" ]]; then
    EXP_SUFFIX="_daycount_fullK_mins${MINSAMPLES}_p${NSLUG}"
  else
    EXP_SUFFIX="_daycount_emk1_noise${NSLUG}_mins${MINSAMPLES}"
  fi

  local EVAL_CFG="eval_config_perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30_daycount_balanced"
  local EXP_NAME="perpetua_${DIST}_${GROUPING}_pucpr_split_50_20_30${EXP_SUFFIX}${TAG_SUFFIX}"
  local MODEL_DIR="/workspace/data/output/perpetua/${EXP_NAME}/days${N_SLUG}_seed${SEED}"
  local PARAMS="$MODEL_DIR/persistence_params.json"

  echo "================================================================================================"
  echo "[$(date +%H:%M:%S)] scaledhz  $KIND  $DIST × $GROUPING  D=$N"
  echo "================================================================================================"

  # Gate 1 — model must already be trained.
  if [[ ! -f "$PARAMS" ]]; then
    echo "[skip-no-model] $PARAMS does not exist; train via the percentile-horizons sweep first."
    return 0
  fi

  # K_POST / N_PRUNED from the existing model metadata (best-effort; if the
  # extraction fails we still proceed to eval but log nan for these columns).
  local K_POST N_PRUNED
  K_POST=$(python3 -c "import json; p=json.load(open('$PARAMS')); print(p['__global__']['metadata']['n_components_fitted'])" 2>/dev/null || echo "")
  N_PRUNED=$(python3 -c "import json; p=json.load(open('$PARAMS')); print(p['__global__']['metadata']['n_pruned_components'])" 2>/dev/null || echo "")
  if [[ -z "$K_POST" || -z "$N_PRUNED" ]]; then
    echo "[warn] couldn't read K_post / n_pruned from $PARAMS"
  fi

  # Gate 2 — idempotency. With noise or filter overrides, `find -newer` can't
  # distinguish ablation evals from default ones under the same scenario
  # subdir, so switch to checking the (suffix-tagged) OUT_CSV for a row.
  local EXISTING_EVAL=""
  if [[ -z "${FORCE_REEVAL:-}" ]]; then
    if [[ -n "$NOISE_TAG" || -n "$FILTER_TAG" ]]; then
      if awk -F',' -v k="$KIND" -v v="$DIST" -v g="$GROUPING" -v n="$N" \
          'NR>1 && $1==k && $2==v && $3==g && $4==n {found=1; exit} END {exit !found}' \
          "$OUT_CSV"; then
        EXISTING_EVAL="csv_row"
      fi
    else
      EXISTING_EVAL=$(find /workspace/data/output/persistence_eval_outputs \
        -path "*/${EXP_NAME}_days${N_SLUG}_seed${SEED}/*evaluation_runs/survival_based__${EVAL_SCENARIO}/*/metric_results.json" \
        -newer "$PARAMS" 2>/dev/null \
        | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)
    fi
  fi
  local LATEST_EVAL
  if [[ -n "$EXISTING_EVAL" && "$EXISTING_EVAL" != "csv_row" ]]; then
    echo "[skip-eval] reusing $EXISTING_EVAL"
    LATEST_EVAL="$EXISTING_EVAL"
  elif [[ "$EXISTING_EVAL" == "csv_row" ]]; then
    echo "[skip-eval] CSV already has a row for this cell"
    return 0
  else
    local SIDECAR_OVERRIDE=""
    if [[ "$GROUPING" == "location" ]]; then
      SIDECAR_OVERRIDE="inferencer.track_id_to_location_file=$TEST_SIDECAR"
    fi
    # shellcheck disable=SC2086
    if ! python eval_persistence.py \
        --config-name="$EVAL_CFG" \
        eval_scenario="$EVAL_SCENARIO" \
        experiment.name="$EXP_NAME" \
        experiment.day_count="$N" \
        experiment.subsample_fraction=1.0 \
        experiment.subsample_seed="$SEED" \
        benchmark_dataset.manifest_path="$TEST_MANIFEST" \
        $SIDECAR_OVERRIDE \
        $NOISE_OVERRIDES \
        $FILTER_OVERRIDES; then
      echo "[FAIL] eval crashed for $KIND $DIST $GROUPING D=$N"
      return 1
    fi
    LATEST_EVAL=$(find /workspace/data/output/persistence_eval_outputs \
      -path "*/${EXP_NAME}_days${N_SLUG}_seed${SEED}/*evaluation_runs/survival_based__${EVAL_SCENARIO}/*/metric_results.json" \
      -newer "$PARAMS" 2>/dev/null \
      | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)
  fi

  if [[ -n "$LATEST_EVAL" && -f "$LATEST_EVAL" ]]; then
    python -c "
import json
with open('$LATEST_EVAL') as f:
    metrics = {x['name']: x['value'] for x in json.load(f) if not x.get('group')}
fields = ['accuracy@0.5','balanced_accuracy@0.5','f1@0.5','precision@0.5','recall@0.5','tnr@0.5','existence_mae']
row = ['$KIND', '$DIST', '$GROUPING', '$N', '$PMPF', '$PMPF', '$MINSAMPLES', '$SEED', '$K_POST', '$N_PRUNED'] + [f\"{metrics.get(k, float('nan')):.6f}\" for k in fields] + ['$LATEST_EVAL']
print(','.join(row))
" >> "$OUT_CSV"
  else
    echo "[FAIL] eval finished but no metric_results.json found newer than $PARAMS"
    return 1
  fi
}

# Throttle to at most $JOBS concurrent background jobs.
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
        LABEL="$KIND $DIST × $GROUPING D=$N"
        _throttle
        n_launched=$((n_launched + 1))
        echo "[$(date +%H:%M:%S)] launched ($n_launched): $LABEL  -> $(basename "$LOGFILE")"
        (
          _start=$SECONDS
          if ! run_cell "$KIND" "$VG" "$N" > "$LOGFILE" 2>&1; then
            echo "[$(date +%H:%M:%S)] FAILED  $LABEL  ($((SECONDS - _start))s, see $LOGFILE)" >&2
          else
            echo "[$(date +%H:%M:%S)] done    $LABEL  ($((SECONDS - _start))s)"
          fi
        ) &
      else
        run_cell "$KIND" "$VG" "$N" || true
      fi
    done
  done
done

if (( JOBS > 1 )); then
  wait
  echo
  echo "All $n_launched cells completed (JOBS=$JOBS)."
  echo "Per-cell logs: $LOG_DIR/"
  if grep -lE "^\[FAIL\]|Error|Traceback" "$LOG_DIR"/*.log 2>/dev/null | grep -q .; then
    echo "WARNING: failure markers found in:" >&2
    grep -lE "^\[FAIL\]|Error|Traceback" "$LOG_DIR"/*.log 2>/dev/null >&2
  fi
fi
echo "Done. Results: $OUT_CSV"
