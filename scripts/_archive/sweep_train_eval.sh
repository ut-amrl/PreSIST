#!/bin/bash
# Sweep distribution × grouping across training-data fractions in spread order,
# train+eval each cell, and log a CSV of the headline metrics.
#
# By default the sweep runs the four single-component variants first
# (cheap, ~10s/cell) and then the four mixture variants (expensive,
# ~70min/cell with select-and-refine — order intentional so partial sweeps
# leave you with the single-component baseline before mixtures finish).
#
# Default targets: the *repaired* PUCPR train/test split via the
# `annotation_tool_pklot_repaired_train` and `annotation_tool_pucpr_repaired_test`
# Hydra presets. Override the following env vars to point elsewhere:
#
#   TRAIN_PRESET    — Hydra train_dataset group choice to use in place of the
#                     train config's default. Sets the preset's `name:` field,
#                     which is what shows up in logs / experiment metadata.
#   BENCH_PRESET    — Hydra benchmark_dataset group choice; its `name:` becomes
#                     the top-level dataset subdir under the eval output root.
#   TRAIN_MANIFEST  — path to the train manifest JSON   (default: preset)
#   TEST_MANIFEST   — path to the test manifest JSON    (default: preset)
#   TEST_SIDECAR    — path to test track_id_to_location sidecar (REQUIRED for
#                     per-location variants; defaults to the cell128 sidecar
#                     next to the repaired test manifest)
#   FRACTIONS       — space-separated fraction list (default: spread order)
#   SEED            — experiment.subsample_seed (default: 0)
#   CELL_SIZE       — cell_size_px (default: 128)
#   VARIANTS        — space-separated "<dist>:<grouping>" list. Default is
#                     all 8 (4 single + 4 mixture). Recognised <dist> values:
#                     single_exp, single_lognormal, mixture_exp,
#                     mixture_lognormal.
#   EVAL_SCENARIO   — Hydra eval_scenario group choice (default:
#                     percentile_horizons). Override to dynamic_anchor_only
#                     for the previous default behaviour.
#   MIXTURE_FAST    — if non-empty, mixture trainers run in sanity mode
#                     (params_range=[2], num_retrain_attempts=0). Knocks
#                     ~8x off mixture cell wall-clock. Default: full
#                     select-and-refine.
#   OUT_CSV         — aggregate metrics CSV (test-side results; lives under the
#                     eval-outputs root, not the perpetua model-artifact root)
#                     (default: /workspace/data/output/persistence_eval_outputs/sweep_metrics.csv)
#   FORCE_TRAIN     — if non-empty, retrain (and re-eval) every cell even if
#                     a trained model already exists. Implies FORCE_EVAL.
#   FORCE_EVAL      — if non-empty, re-run eval even when a metric_results.json
#                     newer than the trained params is already on disk. Has
#                     no effect on the train skip check.
#
# Usage (inside the perpetua container):
#   bash scripts/sweep_single_components.sh

set -euo pipefail

CELL_SIZE="${CELL_SIZE:-128}"

DEFAULT_TEST_MANIFEST=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_50/test_from_2012-10-16.json
TEST_MANIFEST="${TEST_MANIFEST:-$DEFAULT_TEST_MANIFEST}"
TEST_SIDECAR_DEFAULT="$(dirname "$TEST_MANIFEST")/$(basename "$TEST_MANIFEST" .json).track_id_to_location_cell${CELL_SIZE}.json"
TEST_SIDECAR="${TEST_SIDECAR:-$TEST_SIDECAR_DEFAULT}"

# Dataset overrides. Two layers:
#   - TRAIN_PRESET / BENCH_PRESET swap the Hydra group choice wholesale
#     (i.e. `train_dataset=<preset>` / `benchmark_dataset=<preset>`). Use this
#     when you want the preset's `name:` to drive output subdirs — eval output
#     dataset folder, train_dataset.name in logs, etc.
#   - TRAIN_MANIFEST / TEST_MANIFEST override only `manifest_path` on whatever
#     preset is active. Useful when the preset name is fine and you just want
#     to point it at a different file.
# Preset overrides win on the name fields; manifest overrides win on the path.
TRAIN_OVERRIDE=""
if [[ -n "${TRAIN_PRESET:-}" ]]; then
    TRAIN_OVERRIDE="train_dataset=$TRAIN_PRESET"
fi
if [[ -n "${TRAIN_MANIFEST:-}" ]]; then
    TRAIN_OVERRIDE="$TRAIN_OVERRIDE train_dataset.manifest_path=$TRAIN_MANIFEST"
fi
TEST_OVERRIDE=""
if [[ -n "${BENCH_PRESET:-}" ]]; then
    TEST_OVERRIDE="benchmark_dataset=$BENCH_PRESET"
fi
if [[ "$TEST_MANIFEST" != "$DEFAULT_TEST_MANIFEST" ]]; then
    TEST_OVERRIDE="$TEST_OVERRIDE benchmark_dataset.manifest_path=$TEST_MANIFEST"
fi

if [[ ! -f "$TEST_SIDECAR" ]]; then
    echo "ERROR: test sidecar $TEST_SIDECAR not found." >&2
    echo "Build it with:" >&2
    echo "    python build_track_location_index.py --manifest $TEST_MANIFEST --cell-size-px $CELL_SIZE" >&2
    exit 1
fi

# Resolve the benchmark_dataset.name so eval-result lookups can be scoped to
# the active dataset subdir. Without this scoping, the LATEST_EVAL/EXISTING_EVAL
# find commands match metric_results.json from ANY dataset that shares this
# EXP_NAME (train vs test, or older split variants), so concurrent or prior
# sweeps cross-contaminate the per-cell CSV row.
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
    echo "[scope] BENCH_PRESET not set or YAML name not parseable; eval-result lookups are UNSCOPED across datasets."
fi

FRACTIONS="${FRACTIONS:-0.1 1.0 0.5 0.3 0.7 0.2 0.4 0.6 0.8 0.9}"
SEED="${SEED:-0}"
EVAL_SCENARIO="${EVAL_SCENARIO:-percentile_horizons}"
# Default order: all single-component variants first (cheap), then mixtures.
VARIANTS="${VARIANTS:-single_exp:location single_lognormal:location single_exp:class single_lognormal:class mixture_exp:location mixture_lognormal:location mixture_exp:class mixture_lognormal:class}"
OUT_CSV="${OUT_CSV:-/workspace/data/output/persistence_eval_outputs/sweep_metrics.csv}"

# Sanity-mode overrides for mixture trainers — disables the K-sweep and
# perturbation refinement loop that drive the long mixture wall-clock.
MIXTURE_FAST_OVERRIDES=""
if [[ -n "${MIXTURE_FAST:-}" ]]; then
    MIXTURE_FAST_OVERRIDES="trainer.params_range=[2] trainer.num_retrain_attempts=0"
fi

mkdir -p "$(dirname "$OUT_CSV")"
if [[ ! -f "$OUT_CSV" ]]; then
    echo "variant,grouping,fraction,seed,n_buckets,accuracy@0.5,balanced_accuracy@0.5,f1@0.5,precision@0.5,recall@0.5,tnr@0.5,existence_mae,eval_dir" > "$OUT_CSV"
fi

# EXP_SUFFIX is appended to EXP_NAME so callers can keep model_dir / eval
# output paths from colliding with prior runs against different data splits.
# Used by wrapper scripts like sweep_train_eval_pucpr_repaired_split_50_50.sh.
EXP_SUFFIX="${EXP_SUFFIX:-}"

for variant_grouping in $VARIANTS; do
    DIST="${variant_grouping%:*}"
    GROUPING="${variant_grouping#*:}"
    # TRAIN_CFG_TEMPLATE / EVAL_CFG_TEMPLATE: callers can override the cfg
    # naming pattern. {DIST} and {GROUPING} are interpolated at expansion
    # time. Defaults preserve the historical layout where the train cfg
    # picks grouping at runtime and the eval cfg encodes grouping in the
    # filename. Wrappers that materialize per-(dist, grouping) train/eval
    # configs (e.g. _pucpr_split_50_50 variants) override the templates to
    # match.
    if [[ -z "${TRAIN_CFG_TEMPLATE:-}" ]]; then
        TRAIN_CFG_TEMPLATE='train_config_perpetua_{DIST}'
    fi
    if [[ -z "${EVAL_CFG_TEMPLATE:-}" ]]; then
        EVAL_CFG_TEMPLATE='eval_config_perpetua_{DIST}_{GROUPING}'
    fi
    TRAIN_CFG="${TRAIN_CFG_TEMPLATE//\{DIST\}/$DIST}"
    TRAIN_CFG="${TRAIN_CFG//\{GROUPING\}/$GROUPING}"
    EVAL_CFG="${EVAL_CFG_TEMPLATE//\{DIST\}/$DIST}"
    EVAL_CFG="${EVAL_CFG//\{GROUPING\}/$GROUPING}"
    EXP_NAME="perpetua_${DIST}_${GROUPING}${EXP_SUFFIX}"

    for FRACTION in $FRACTIONS; do
        echo "===================================================================="
        echo "[$(date +%H:%M:%S)] $DIST × $GROUPING @ frac=$FRACTION  seed=$SEED"
        echo "===================================================================="

        FRACTION_SLUG="${FRACTION//./p}"
        MODEL_DIR="/workspace/data/output/perpetua/${EXP_NAME}/frac${FRACTION_SLUG}_seed${SEED}"

        TRAINED=0
        if [[ -f "$MODEL_DIR/persistence_params.json" && -f "$MODEL_DIR/training_metadata.json" && -z "${FORCE_TRAIN:-}" ]]; then
            echo "[skip-train] $MODEL_DIR already trained; pass FORCE_TRAIN=1 to retrain."
        else
            TRAINED=1
            # Mixture trainers respect $MIXTURE_FAST; single trainers ignore the
            # vars (they don't accept those keys, so we skip the override).
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
                experiment.subsample_fraction="$FRACTION" \
                experiment.subsample_seed="$SEED" \
                $TRAIN_OVERRIDE \
                $TRAIN_MIXTURE_OVERRIDES
        fi

        # Skip eval if a fresh metric_results.json already exists (newer than
        # the trained params) and we didn't just retrain. FORCE_EVAL/FORCE_TRAIN
        # override the skip.
        # Scope the eval-skip search to this exact cell (benchmark/exp+frac+seed
        # subdir). The previous unscoped `find` matched any newer
        # metric_results.json anywhere under persistence_eval_outputs, including
        # results from unrelated variants/cells.
        EXISTING_EVAL=""
        if [[ "$TRAINED" -eq 0 && -z "${FORCE_EVAL:-}" && -z "${FORCE_TRAIN:-}" ]]; then
            EXISTING_EVAL=$(find "$EVAL_PATH_PREFIX" \
                -path "*/${EXP_NAME}_frac${FRACTION_SLUG}_seed${SEED}/*/metric_results.json" \
                -newer "$MODEL_DIR/persistence_params.json" 2>/dev/null \
                | xargs -I{} ls -td {} 2>/dev/null | head -1 || true)
        fi

        if [[ -n "$EXISTING_EVAL" ]]; then
            echo "[skip-eval] reusing $EXISTING_EVAL; pass FORCE_EVAL=1 to re-eval."
        else
            # The track_id_to_location sidecar is only meaningful for the
            # per-location inferencer; the class-keyed inferencer config has no
            # such field, so passing it would fail Hydra's struct validation.
            SIDECAR_OVERRIDE=""
            if [[ "$GROUPING" == "location" ]]; then
                SIDECAR_OVERRIDE="inferencer.track_id_to_location_file=$TEST_SIDECAR"
            fi

            # shellcheck disable=SC2086
            python eval_persistence.py \
                --config-name="$EVAL_CFG" \
                eval_scenario="$EVAL_SCENARIO" \
                experiment.name="$EXP_NAME" \
                experiment.subsample_fraction="$FRACTION" \
                experiment.subsample_seed="$SEED" \
                $SIDECAR_OVERRIDE \
                $TEST_OVERRIDE
        fi

        # Health check on the trained model before declaring this cell "done".
        # Non-fatal — warns but keeps sweeping so one bad cell doesn't kill the
        # whole run. The full breakdown is one --verbose invocation away.
        if ! python scripts/inspect_trained_model.py "$MODEL_DIR"; then
            echo "WARNING: $MODEL_DIR failed inspect_trained_model checks." >&2
            echo "  Re-run with --verbose for details:" >&2
            echo "    python scripts/inspect_trained_model.py --verbose $MODEL_DIR" >&2
        fi

        LATEST_EVAL=$(find "$EVAL_PATH_PREFIX" \
            -path "*/${EXP_NAME}_frac${FRACTION_SLUG}_seed${SEED}/*/metric_results.json" \
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
row = ['$DIST', '$GROUPING', '$FRACTION', '$SEED', '$N_BUCKETS'] + [f\"{metrics.get(k, float('nan')):.6f}\" for k in fields] + ['$LATEST_EVAL']
print(','.join(row))
" >> "$OUT_CSV"
        else
            echo "WARNING: no metric_results.json found newer than $MODEL_DIR/persistence_params.json"
            echo "$DIST,$GROUPING,$FRACTION,$SEED,$N_BUCKETS,nan,nan,nan,nan,nan,nan,nan,MISSING" >> "$OUT_CSV"
        fi
    done
done

echo
echo "Sweep complete. Results: $OUT_CSV"
if command -v column >/dev/null 2>&1; then
    column -ts, "$OUT_CSV" | head -20
else
    head -20 "$OUT_CSV"
fi
