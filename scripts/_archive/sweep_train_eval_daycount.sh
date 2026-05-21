#!/bin/bash
# Day-count parallel of sweep_train_eval.sh: train+eval Perpetua variants
# across a DAYS sweep instead of a FRACTIONS sweep. The day-count subsetting
# is materialized on disk by generate_daycount_train_manifests.py; this
# driver overrides TRAIN_MANIFEST per cell to point at the right pre-built
# manifest, and forces experiment.subsample_fraction=1.0.
#
# Required env:
#   DAYCOUNT_ROOT   — directory holding days_NNNN/ subdirs from
#                     generate_daycount_train_manifests.py.
# Required Hydra group/template overrides (set by the wrapper):
#   TRAIN_PRESET    — Hydra train_dataset group choice (e.g.
#                     annotation_tool_pklot_split_50_20_30_train).
#   BENCH_PRESET    — Hydra benchmark_dataset group choice.
#   TRAIN_CFG_TEMPLATE / EVAL_CFG_TEMPLATE — {DIST}/{GROUPING}-templated cfg
#                     names for the daycount variants (e.g.
#                     train_config_perpetua_{DIST}_{GROUPING}_pucpr_split_50_20_30_daycount).
#
# Optional env:
#   DAYS            — space-separated day-count list. Default derived from the
#                     <DAYCOUNT_ROOT>/../split_metadata.json's train_days field
#                     (denser low end, capped at train_days):
#                       "1 2 3 4 5 7 10 14 21 28 <train_days>".
#   SEEDS           — space-separated seeds (default: 0).
#   CELL_SIZE       — cell_size_px override (default: 128).
#   VARIANTS        — same syntax as sweep_train_eval.sh.
#   EVAL_SCENARIO   — Hydra eval_scenario group (default: percentile_horizons).
#   MIXTURE_FAST    — if non-empty, mixture trainers run in sanity mode.
#   OUT_CSV         — aggregate metrics CSV path.
#   FORCE_TRAIN / FORCE_EVAL — same semantics as sweep_train_eval.sh.
#   EXP_SUFFIX      — appended to experiment.name's variant key for cell paths
#                     (defaults to empty; daycount cfgs already encode the
#                     suffix in their experiment.name).
#   SINGLE_EXP_SUFFIX — additional suffix appended on top of EXP_SUFFIX for
#                     single_* variants only (lets you split single/mixture
#                     output dirs while reusing existing mixture checkpoints).
#   SINGLE_TRAIN_EXTRA — extra Hydra train-side overrides for single_* variants
#                     only. Useful for `+trainer.fit_method=em_k1`.
#   TEST_MANIFEST   — explicit test manifest path (defaults to wrapper-set).
#   TEST_SIDECAR    — track_id_to_location sidecar for per-location eval.
#
# This script is invoked by the dataset-specific wrappers, e.g.
# sweep_train_eval_pucpr_split_50_20_30_daycount_balanced.sh.

set -euo pipefail

if [[ -z "${DAYCOUNT_ROOT:-}" ]]; then
    echo "ERROR: DAYCOUNT_ROOT must be set (e.g. /workspace/.../splits/50_20_30/daycount)." >&2
    exit 2
fi
if [[ ! -d "$DAYCOUNT_ROOT" ]]; then
    echo "ERROR: DAYCOUNT_ROOT does not exist: $DAYCOUNT_ROOT" >&2
    exit 2
fi

CELL_SIZE="${CELL_SIZE:-128}"

# Default DAYS: derive grid from split_metadata.json sitting next to the
# daycount subdir (i.e. its parent split dir).
SPLIT_METADATA="$DAYCOUNT_ROOT/../split_metadata.json"
TRAIN_DAYS=""
if [[ -f "$SPLIT_METADATA" ]]; then
    TRAIN_DAYS=$(python3 -c "import json,sys; print(json.load(open('$SPLIT_METADATA'))['train_days'])" 2>/dev/null || true)
fi
if [[ -z "${DAYS:-}" ]]; then
    if [[ -z "$TRAIN_DAYS" ]]; then
        echo "ERROR: DAYS unset and could not read train_days from $SPLIT_METADATA." >&2
        exit 2
    fi
    # Standard low-end-dense grid, with intermediates dropped if they exceed
    # train_days; train_days is always appended last.
    DEFAULT_GRID="1 2 3 4 5 7 10 14 21 28"
    DAYS=""
    for n in $DEFAULT_GRID; do
        if (( n < TRAIN_DAYS )); then
            DAYS="$DAYS $n"
        fi
    done
    DAYS="$DAYS $TRAIN_DAYS"
    DAYS="$(echo "$DAYS" | xargs)"  # collapse whitespace
fi

SEEDS="${SEEDS:-0}"
EVAL_SCENARIO="${EVAL_SCENARIO:-percentile_horizons}"
VARIANTS="${VARIANTS:-single_exp:location single_lognormal:location single_exp:class single_lognormal:class mixture_exp:location mixture_lognormal:location mixture_exp:class mixture_lognormal:class}"
OUT_CSV="${OUT_CSV:-/workspace/data/output/persistence_eval_outputs/sweep_metrics_daycount.csv}"

if [[ -z "${TRAIN_CFG_TEMPLATE:-}" || -z "${EVAL_CFG_TEMPLATE:-}" ]]; then
    echo "ERROR: TRAIN_CFG_TEMPLATE and EVAL_CFG_TEMPLATE must be set by the wrapper." >&2
    exit 2
fi

TEST_MANIFEST="${TEST_MANIFEST:-}"
TEST_SIDECAR_DEFAULT=""
if [[ -n "$TEST_MANIFEST" ]]; then
    TEST_SIDECAR_DEFAULT="$(dirname "$TEST_MANIFEST")/$(basename "$TEST_MANIFEST" .json).track_id_to_location_cell${CELL_SIZE}.json"
fi
TEST_SIDECAR="${TEST_SIDECAR:-$TEST_SIDECAR_DEFAULT}"

# Train manifest override is set per-cell (different N → different manifest);
# unlike sweep_train_eval.sh we don't honor a global TRAIN_MANIFEST env.
if [[ -n "${TRAIN_MANIFEST:-}" ]]; then
    echo "WARNING: TRAIN_MANIFEST is ignored by the daycount driver (it iterates DAYS)." >&2
fi

TRAIN_OVERRIDE_BASE=""
if [[ -n "${TRAIN_PRESET:-}" ]]; then
    TRAIN_OVERRIDE_BASE="train_dataset=$TRAIN_PRESET"
fi
TEST_OVERRIDE=""
if [[ -n "${BENCH_PRESET:-}" ]]; then
    TEST_OVERRIDE="benchmark_dataset=$BENCH_PRESET"
fi
if [[ -n "$TEST_MANIFEST" ]]; then
    TEST_OVERRIDE="$TEST_OVERRIDE benchmark_dataset.manifest_path=$TEST_MANIFEST"
fi

if [[ -n "$TEST_SIDECAR" && ! -f "$TEST_SIDECAR" ]]; then
    echo "ERROR: test sidecar $TEST_SIDECAR not found." >&2
    echo "Build it with:" >&2
    echo "    python build_track_location_index.py --manifest $TEST_MANIFEST --cell-size-px $CELL_SIZE" >&2
    exit 1
fi

# Resolve benchmark_dataset.name for scoped eval-result lookups (same logic
# as sweep_train_eval.sh).
SCRIPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DATASET_NAME=""
if [[ -n "${BENCH_PRESET:-}" ]]; then
    BENCH_YAML="$SCRIPT_REPO_ROOT/configs/train_confs/benchmark_dataset/${BENCH_PRESET}.yaml"
    if [[ -f "$BENCH_YAML" ]]; then
        BENCH_DATASET_NAME=$(awk '/^name:[[:space:]]*/ {sub(/^name:[[:space:]]*/, ""); print; exit}' "$BENCH_YAML")
    fi
fi
if [[ -n "$BENCH_DATASET_NAME" ]]; then
    EVAL_PATH_PREFIX="/workspace/data/output/persistence_eval_outputs/$BENCH_DATASET_NAME"
    echo "[scope] eval-result lookups scoped to dataset subdir: $BENCH_DATASET_NAME"
else
    EVAL_PATH_PREFIX="/workspace/data/output/persistence_eval_outputs"
    echo "[scope] BENCH_PRESET not set or YAML name not parseable; eval-result lookups are UNSCOPED."
fi

MIXTURE_FAST_OVERRIDES=""
if [[ -n "${MIXTURE_FAST:-}" ]]; then
    MIXTURE_FAST_OVERRIDES="trainer.params_range=[2] trainer.num_retrain_attempts=0"
fi

mkdir -p "$(dirname "$OUT_CSV")"
if [[ ! -f "$OUT_CSV" ]]; then
    echo "variant,grouping,days,seed,n_buckets,accuracy@0.5,balanced_accuracy@0.5,f1@0.5,precision@0.5,recall@0.5,tnr@0.5,existence_mae,eval_dir" > "$OUT_CSV"
fi

# Resolve train manifest for one N (single glob hit, error otherwise).
resolve_train_manifest() {
    local n="$1"
    local subdir
    subdir="$DAYCOUNT_ROOT/days_$(printf '%04d' "$n")"
    local hits=( "$subdir"/train_first_*.json )
    if (( ${#hits[@]} != 1 )) || [[ ! -f "${hits[0]}" ]]; then
        echo "ERROR: expected exactly one train_first_*.json in $subdir; got ${#hits[@]}." >&2
        return 2
    fi
    echo "${hits[0]}"
}

EXP_SUFFIX="${EXP_SUFFIX:-}"

echo "Daycount sweep plan: DAYS=[$DAYS]  SEEDS=[$SEEDS]  VARIANTS=[$VARIANTS]"
echo "  daycount_root: $DAYCOUNT_ROOT"
echo "  bench_preset:  ${BENCH_PRESET:-<none>}"
echo "  test_manifest: $TEST_MANIFEST"
echo "  out_csv:       $OUT_CSV"
echo

for variant_grouping in $VARIANTS; do
    DIST="${variant_grouping%:*}"
    GROUPING="${variant_grouping#*:}"
    TRAIN_CFG="${TRAIN_CFG_TEMPLATE//\{DIST\}/$DIST}"
    TRAIN_CFG="${TRAIN_CFG//\{GROUPING\}/$GROUPING}"
    EVAL_CFG="${EVAL_CFG_TEMPLATE//\{DIST\}/$DIST}"
    EVAL_CFG="${EVAL_CFG//\{GROUPING\}/$GROUPING}"
    # Single variants can carry an additional suffix + extra Hydra overrides
    # so single fits (e.g. fit_method=em_k1) land in a parallel exp dir
    # without disturbing existing mixture checkpoints.
    PER_VARIANT_SUFFIX=""
    PER_VARIANT_TRAIN_EXTRA=""
    if [[ "$DIST" == single_* ]]; then
        PER_VARIANT_SUFFIX="${SINGLE_EXP_SUFFIX:-}"
        PER_VARIANT_TRAIN_EXTRA="${SINGLE_TRAIN_EXTRA:-}"
    fi
    EXP_NAME="perpetua_${DIST}_${GROUPING}${EXP_SUFFIX}${PER_VARIANT_SUFFIX}"

    for N in $DAYS; do
        TRAIN_MANIFEST_N=$(resolve_train_manifest "$N")
        TRAIN_OVERRIDE="$TRAIN_OVERRIDE_BASE train_dataset.manifest_path=$TRAIN_MANIFEST_N"
        DAYS_SLUG="days${N}"

        for SEED in $SEEDS; do
            echo "===================================================================="
            echo "[$(date +%H:%M:%S)] $DIST × $GROUPING @ days=$N  seed=$SEED"
            echo "===================================================================="

            MODEL_DIR="/workspace/data/output/perpetua/${EXP_NAME}/${DAYS_SLUG}_seed${SEED}"

            TRAINED=0
            if [[ -f "$MODEL_DIR/persistence_params.json" && -f "$MODEL_DIR/training_metadata.json" && -z "${FORCE_TRAIN:-}" ]]; then
                echo "[skip-train] $MODEL_DIR already trained; pass FORCE_TRAIN=1 to retrain."
            else
                TRAINED=1
                TRAIN_MIXTURE_OVERRIDES=""
                if [[ "$DIST" == mixture_* && -n "$MIXTURE_FAST_OVERRIDES" ]]; then
                    TRAIN_MIXTURE_OVERRIDES="$MIXTURE_FAST_OVERRIDES"
                fi

                # shellcheck disable=SC2086
                python train_persistence_models.py \
                    --config-name="$TRAIN_CFG" \
                    grouping="$GROUPING" \
                    train_dataset.cell_size_px="$CELL_SIZE" \
                    experiment.name="$EXP_NAME" \
                    experiment.day_count="$N" \
                    experiment.subsample_fraction=1.0 \
                    experiment.subsample_seed="$SEED" \
                    $TRAIN_OVERRIDE \
                    $TRAIN_MIXTURE_OVERRIDES \
                    $PER_VARIANT_TRAIN_EXTRA
            fi

            EXISTING_EVAL=""
            if [[ "$TRAINED" -eq 0 && -z "${FORCE_EVAL:-}" && -z "${FORCE_TRAIN:-}" ]]; then
                EXISTING_EVAL=$(find "$EVAL_PATH_PREFIX" \
                    -path "*/${EXP_NAME}_${DAYS_SLUG}_seed${SEED}/*/metric_results.json" \
                    -newer "$MODEL_DIR/persistence_params.json" 2>/dev/null \
                    | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)
            fi

            if [[ -n "$EXISTING_EVAL" ]]; then
                echo "[skip-eval] reusing $EXISTING_EVAL; pass FORCE_EVAL=1 to re-eval."
            else
                SIDECAR_OVERRIDE=""
                if [[ "$GROUPING" == "location" && -n "$TEST_SIDECAR" ]]; then
                    SIDECAR_OVERRIDE="inferencer.track_id_to_location_file=$TEST_SIDECAR"
                fi

                # shellcheck disable=SC2086
                python eval_persistence.py \
                    --config-name="$EVAL_CFG" \
                    eval_scenario="$EVAL_SCENARIO" \
                    experiment.name="$EXP_NAME" \
                    experiment.day_count="$N" \
                    experiment.subsample_fraction=1.0 \
                    experiment.subsample_seed="$SEED" \
                    $SIDECAR_OVERRIDE \
                    $TEST_OVERRIDE
            fi

            if ! python scripts/inspect_trained_model.py "$MODEL_DIR"; then
                echo "WARNING: $MODEL_DIR failed inspect_trained_model checks." >&2
                echo "  Re-run with --verbose for details:" >&2
                echo "    python scripts/inspect_trained_model.py --verbose $MODEL_DIR" >&2
            fi

            LATEST_EVAL=$(find "$EVAL_PATH_PREFIX" \
                -path "*/${EXP_NAME}_${DAYS_SLUG}_seed${SEED}/*/metric_results.json" \
                -newer "$MODEL_DIR/persistence_params.json" 2>/dev/null \
                | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)

            N_BUCKETS=$(python -c "
import json
with open('$MODEL_DIR/training_metadata.json') as f: m = json.load(f)
print(m['n_buckets'])
")

            if [[ -n "${LATEST_EVAL:-}" && -f "$LATEST_EVAL" ]]; then
                python -c "
import json
with open('$LATEST_EVAL') as f:
    metrics = {x['name']: x['value'] for x in json.load(f)}
fields = ['accuracy@0.5','balanced_accuracy@0.5','f1@0.5','precision@0.5','recall@0.5','tnr@0.5','existence_mae']
row = ['$DIST', '$GROUPING', '$N', '$SEED', '$N_BUCKETS'] + [f\"{metrics.get(k, float('nan')):.6f}\" for k in fields] + ['$LATEST_EVAL']
print(','.join(row))
" >> "$OUT_CSV"
            else
                echo "WARNING: no metric_results.json found newer than $MODEL_DIR/persistence_params.json"
                echo "$DIST,$GROUPING,$N,$SEED,$N_BUCKETS,nan,nan,nan,nan,nan,nan,nan,MISSING" >> "$OUT_CSV"
            fi
        done
    done
done

echo
echo "Daycount sweep complete. Results: $OUT_CSV"
if command -v column >/dev/null 2>&1; then
    column -ts, "$OUT_CSV" | head -20
else
    head -20 "$OUT_CSV"
fi
