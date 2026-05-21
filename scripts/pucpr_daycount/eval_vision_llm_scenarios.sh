#!/bin/bash
# Eval-only sweep for the vision_anchor + LLM (gemma) baselines under any
# chosen eval_scenario (defaults to percentile_scaled_horizons). Companion to
# _oneshot_daycount_p0p01_mins5_all8_scaledhz.sh, which handles the 8 perpetua
# variants. This script never re-runs inference — it relies on the inference
# artifacts produced by the original training+inference passes, located under
# the existing `inference_runs/<timestamp>/` trees.
#
# Coverage:
#   - vision_anchor:class × {D = 1, 2, 3, 4, 5, 7, 10, 14, 21, 28, 35}  → 11 cells
#   - LLM (gemma):global × {single eval, no D axis}                     → 1 cell
#
# Semantics:
#   - For each cell, locates the latest inference_runs/<timestamp>/ dir for
#     that (method, day) pair. If none exists → log "[skip-no-inference]"
#     and continue (cell contributes no CSV row).
#   - Looks for an existing metric_results.json under the chosen scenario's
#     evaluation_runs subtree that is newer than the inference dir. If found
#     → log "[skip-eval]" unless FORCE_REEVAL=1.
#   - Otherwise re-runs eval_persistence.py with mode=evaluate against the
#     existing inference artifacts (`++paths.explicit_inference_run_dir=...`)
#     and appends a row to the sweep CSV.
#
# Env overrides (mirror the perpetua scaledhz script):
#   EVAL_SCENARIO=<name>    — eval_scenario YAML name (default
#                             "percentile_scaled_horizons").
#   DAYS="2 7 35"           — restrict vision_anchor to these day-counts.
#                             Ignored for LLM (no day axis).
#   METHODS="vision"        — only vision_anchor
#   METHODS="llm"           — only LLM (gemma)
#   FORCE_REEVAL=1          — re-run eval even if a fresh metric_results.json
#                             already exists.
#   MAX_ANCHOR_OBS=50       — value of benchmark_dataset.max_anchor_observations.
#   JOBS=4                  — parallel workers (default 1).
#   LOG_DIR=<path>          — per-cell logs when JOBS>1.
#   OUT_CSV=<path>          — override aggregate sweep CSV.
#   EVAL_NOISE_PM=0.01      — override persistence_filter.missed_detection_prob
#   EVAL_NOISE_PF=0.01      — override persistence_filter.false_detection_prob
#                             Must be supplied together. When set, outputs go
#                             under a noise-tagged CSV / log dir; existing
#                             default-noise results are not touched. Skip-eval
#                             switches to CSV-row-presence.
set -euo pipefail
SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30
TEST_MANIFEST=$SPLIT_DIR/test_from_2012-10-30.json
TEST_SIDECAR=$SPLIT_DIR/test_from_2012-10-30.track_id_to_location_cell128.json
SEED=0
PMPF=0.01
NSLUG=0p01
MINSAMPLES=5
MAX_ANCHOR_OBS="${MAX_ANCHOR_OBS:-50}"

EVAL_SCENARIO="${EVAL_SCENARIO:-percentile_horizons}"

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
  # Default = matched noise from the eval YAML (p_m = p_f = 0.01). Tag the
  # CSV / log paths so they line up with the historical `_pm0p01_pf0p01`
  # naming the renderer expects.
  NOISE_TAG="_pm0p01_pf0p01"
  NOISE_OVERRIDES=""
fi

# Output roots & dataset namespace (match the perpetua scaledhz script).
OUTPUT_ROOT=/workspace/data/output/persistence_eval_outputs
DATASET_NAME=pucpr_split_new_pklot_PUCPR_test_50_20_30_balanced

# Method-specific identifiers.
VISION_EXP_BASE=vision_anchor_midpoint_pucpr_split_50_20_30_daycount_balanced_train
VISION_EVAL_CFG=eval_config_vision_anchor_midpoint_pucpr_split_50_20_30_daycount_balanced_train_balanced_eval
LLM_EVAL_CFG=eval_config_llm_pucpr_split_50_20_30_gemma_v3iqrBehav4_mO20_percentile_balanced
LLM_INFERENCER_DIR=gemma_cap_proc_long7__gemmaVLMDesc_blur_bg_context_crop_mask_t3_cmd1_no_holes_longPrompt__ph_v3IqrBehav4__gemma4_31b_default

OUT_CSV="${OUT_CSV:-${OUTPUT_ROOT}/sweep_metrics_pucpr_split_50_20_30_daycount_p${NSLUG}_mins${MINSAMPLES}_vision_llm_${EVAL_SCENARIO}${NOISE_TAG}.csv}"
mkdir -p "$(dirname "$OUT_CSV")"
[[ -f "$OUT_CSV" ]] || echo "trainer_kind,variant,grouping,days,p_m,p_f,min_samples,seed,K_post,n_pruned,accuracy@0.5,balanced_accuracy@0.5,f1@0.5,precision@0.5,recall@0.5,tnr@0.5,existence_mae,eval_dir" > "$OUT_CSV"

JOBS="${JOBS:-1}"
LOG_DIR="${LOG_DIR:-$(dirname "$OUT_CSV")/_oneshot_${EVAL_SCENARIO}${NOISE_TAG}_vision_llm_logs}"
mkdir -p "$LOG_DIR"

DEFAULT_DAYS="0.5 1 2 3 4 5 7 10 14 21 28 35"
DAYS_LIST=(${DAYS:-$DEFAULT_DAYS})
DEFAULT_METHODS="vision llm"
METHODS_LIST=(${METHODS:-$DEFAULT_METHODS})

# Append a CSV row by parsing $1=eval_dir/metric_results.json for aggregate
# metrics. Empty K_post/n_pruned (these are perpetua concepts only).
_append_row() {
  local METRIC_PATH="$1"
  local KIND="$2"
  local VARIANT="$3"
  local GROUPING="$4"
  local DAYS_VAL="$5"
  python -c "
import json
with open('$METRIC_PATH') as f:
    metrics = {x['name']: x['value'] for x in json.load(f) if not x.get('group')}
fields = ['accuracy@0.5','balanced_accuracy@0.5','f1@0.5','precision@0.5','recall@0.5','tnr@0.5','existence_mae']
row = ['$KIND', '$VARIANT', '$GROUPING', '$DAYS_VAL', '$PMPF', '$PMPF', '$MINSAMPLES', '$SEED', '', ''] + [f\"{metrics.get(k, float('nan')):.6f}\" for k in fields] + ['$METRIC_PATH']
print(','.join(row))
" >> "$OUT_CSV"
}

# vision_anchor cell at a given day_count.
run_vision_cell() {
  local N="$1"
  local EXP_NAME="$VISION_EXP_BASE"
  local INF_PARENT=$OUTPUT_ROOT/$DATASET_NAME/${EXP_NAME}_days${N}_seed${SEED}/pucpr_grouped_images/inference_runs

  echo "================================================================================================"
  echo "[$(date +%H:%M:%S)] $EVAL_SCENARIO  vision_anchor × class  D=$N"
  echo "================================================================================================"

  # Gate 1 — inference artifact must already exist.
  if [[ ! -d "$INF_PARENT" ]] || ! compgen -G "$INF_PARENT/*" >/dev/null; then
    echo "[skip-no-inference] no inference_runs under $INF_PARENT"
    return 0
  fi
  local INF_DIR
  INF_DIR=$(ls -td "$INF_PARENT"/*/ 2>/dev/null | head -1 | sed 's:/*$::')

  # Gate 2 — idempotency. Noise-tagged runs key off the noise-tagged CSV
  # (avoids confusing noise-varied eval timestamps with default-noise ones
  # under the same scenario subdir).
  local EXISTING_EVAL=""
  if [[ -z "${FORCE_REEVAL:-}" ]]; then
    if [[ -n "$NOISE_TAG" ]]; then
      if awk -F',' -v v="vision_anchor_midpoint" -v g="class" -v n="$N" \
          'NR>1 && $1=="vision" && $2==v && $3==g && $4==n {found=1; exit} END {exit !found}' \
          "$OUT_CSV"; then
        echo "[skip-eval] CSV already has a row for vision D=$N"
        return 0
      fi
    else
      EXISTING_EVAL=$(find "$INF_DIR/evaluation_runs/survival_based__${EVAL_SCENARIO}" \
        -name metric_results.json -newer "$INF_DIR" 2>/dev/null \
        | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)
    fi
  fi
  local LATEST_EVAL
  if [[ -n "$EXISTING_EVAL" ]]; then
    echo "[skip-eval] reusing $EXISTING_EVAL"
    LATEST_EVAL="$EXISTING_EVAL"
  else
    # vision_anchor uses class_keyed_lookup (the inferencer config has no
    # track_id_to_location_file field), so we do NOT pass the sidecar override.
    # shellcheck disable=SC2086
    if ! python eval_persistence.py \
        --config-name="$VISION_EVAL_CFG" \
        mode=evaluate \
        experiment.name="$EXP_NAME" \
        experiment.day_count="$N" \
        experiment.subsample_fraction=1.0 \
        experiment.subsample_seed="$SEED" \
        benchmark_dataset.manifest_path="$TEST_MANIFEST" \
        ++benchmark_dataset.max_anchor_observations="$MAX_ANCHOR_OBS" \
        eval_scenario="$EVAL_SCENARIO" \
        ++paths.explicit_inference_run_dir="$INF_DIR" \
        $NOISE_OVERRIDES; then
      echo "[FAIL] vision_anchor D=$N eval crashed"
      return 1
    fi
    LATEST_EVAL=$(find "$INF_DIR/evaluation_runs/survival_based__${EVAL_SCENARIO}" \
      -name metric_results.json -newer "$INF_DIR" 2>/dev/null \
      | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)
  fi

  if [[ -n "$LATEST_EVAL" && -f "$LATEST_EVAL" ]]; then
    _append_row "$LATEST_EVAL" "vision" "vision_anchor_midpoint" "class" "$N"
  else
    echo "[FAIL] vision_anchor D=$N: no metric_results.json found after eval"
    return 1
  fi
}

# LLM (gemma) cell — single eval, no day_count axis.
run_llm_cell() {
  local INF_PARENT=$OUTPUT_ROOT/$DATASET_NAME/$LLM_INFERENCER_DIR/pucpr_grouped_images/inference_runs

  echo "================================================================================================"
  echo "[$(date +%H:%M:%S)] $EVAL_SCENARIO  LLM (gemma) — single eval"
  echo "================================================================================================"

  if [[ ! -d "$INF_PARENT" ]] || ! compgen -G "$INF_PARENT/*" >/dev/null; then
    echo "[skip-no-inference] no inference_runs under $INF_PARENT"
    return 0
  fi
  local INF_DIR
  INF_DIR=$(ls -td "$INF_PARENT"/*/ 2>/dev/null | head -1 | sed 's:/*$::')

  local EXISTING_EVAL=""
  if [[ -z "${FORCE_REEVAL:-}" ]]; then
    if [[ -n "$NOISE_TAG" ]]; then
      if awk -F',' 'NR>1 && $1=="llm" {found=1; exit} END {exit !found}' "$OUT_CSV"; then
        echo "[skip-eval] CSV already has the LLM row"
        return 0
      fi
    else
      EXISTING_EVAL=$(find "$INF_DIR/evaluation_runs/survival_based__${EVAL_SCENARIO}" \
        -name metric_results.json -newer "$INF_DIR" 2>/dev/null \
        | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)
    fi
  fi
  local LATEST_EVAL
  if [[ -n "$EXISTING_EVAL" ]]; then
    echo "[skip-eval] reusing $EXISTING_EVAL"
    LATEST_EVAL="$EXISTING_EVAL"
  else
    # LLM eval config doesn't take experiment.name / day_count overrides —
    # those identifiers are baked into the inference_runs directory which we
    # pass explicitly via paths.explicit_inference_run_dir.
    # shellcheck disable=SC2086
    if ! python eval_persistence.py \
        --config-name="$LLM_EVAL_CFG" \
        mode=evaluate \
        experiment.subsample_fraction=1.0 \
        experiment.subsample_seed="$SEED" \
        benchmark_dataset.manifest_path="$TEST_MANIFEST" \
        ++benchmark_dataset.max_anchor_observations="$MAX_ANCHOR_OBS" \
        eval_scenario="$EVAL_SCENARIO" \
        ++paths.explicit_inference_run_dir="$INF_DIR" \
        $NOISE_OVERRIDES; then
      echo "[FAIL] LLM eval crashed"
      return 1
    fi
    LATEST_EVAL=$(find "$INF_DIR/evaluation_runs/survival_based__${EVAL_SCENARIO}" \
      -name metric_results.json -newer "$INF_DIR" 2>/dev/null \
      | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)
  fi

  if [[ -n "$LATEST_EVAL" && -f "$LATEST_EVAL" ]]; then
    # days=0 sentinel (LLM has no day axis); grouping=global mirrors the
    # render_bacc_table convention used for the daycount manifests.
    _append_row "$LATEST_EVAL" "llm" "gemma" "global" "0"
  else
    echo "[FAIL] LLM: no metric_results.json found after eval"
    return 1
  fi
}

# Throttle to at most $JOBS concurrent background jobs.
_throttle() {
  while (( $(jobs -rp | wc -l) >= JOBS )); do
    wait -n 2>/dev/null || true
  done
}

# Dispatch: vision per-D × |DAYS_LIST| cells, LLM as a single cell.
N_LAUNCHED=0
_launch_cell() {
  # Args: METHOD [N]
  # METHOD ∈ {vision, llm}; N supplied only for vision.
  local METHOD="$1"
  if (( JOBS > 1 )); then
    local LOGFILE LABEL
    if [[ "$METHOD" == "vision" ]]; then
      LOGFILE="$LOG_DIR/vision_anchor_class_days${2}.log"
      LABEL="vision_anchor × class D=$2"
    else
      LOGFILE="$LOG_DIR/llm_gemma.log"
      LABEL="LLM (gemma)"
    fi
    _throttle
    N_LAUNCHED=$((N_LAUNCHED + 1))
    echo "[$(date +%H:%M:%S)] launched ($N_LAUNCHED): $LABEL  -> $(basename "$LOGFILE")"
    (
      _start=$SECONDS
      if [[ "$METHOD" == "vision" ]]; then
        if ! run_vision_cell "$2" > "$LOGFILE" 2>&1; then
          echo "[$(date +%H:%M:%S)] FAILED  $LABEL  ($((SECONDS - _start))s, see $LOGFILE)" >&2
        else
          echo "[$(date +%H:%M:%S)] done    $LABEL  ($((SECONDS - _start))s)"
        fi
      else
        if ! run_llm_cell > "$LOGFILE" 2>&1; then
          echo "[$(date +%H:%M:%S)] FAILED  $LABEL  ($((SECONDS - _start))s, see $LOGFILE)" >&2
        else
          echo "[$(date +%H:%M:%S)] done    $LABEL  ($((SECONDS - _start))s)"
        fi
      fi
    ) &
  else
    if [[ "$METHOD" == "vision" ]]; then
      run_vision_cell "$2" || true
    else
      run_llm_cell || true
    fi
  fi
}

for METHOD in "${METHODS_LIST[@]}"; do
  case "$METHOD" in
    vision)
      for N in "${DAYS_LIST[@]}"; do
        _launch_cell vision "$N"
      done
      ;;
    llm)
      _launch_cell llm
      ;;
    *)
      echo "[warn] unknown METHOD '$METHOD' — expected vision|llm" >&2
      ;;
  esac
done

if (( JOBS > 1 )); then
  wait
  echo
  echo "All $N_LAUNCHED cells completed (JOBS=$JOBS)."
  echo "Per-cell logs: $LOG_DIR/"
  if grep -lE "^\[FAIL\]|Error|Traceback" "$LOG_DIR"/*.log 2>/dev/null | grep -q .; then
    echo "WARNING: failure markers found in:" >&2
    grep -lE "^\[FAIL\]|Error|Traceback" "$LOG_DIR"/*.log 2>/dev/null >&2
  fi
fi
echo "Done. Results: $OUT_CSV"
