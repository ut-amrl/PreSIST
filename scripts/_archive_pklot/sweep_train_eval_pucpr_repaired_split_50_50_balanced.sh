#!/bin/bash
# Eval-only sibling of sweep_train_eval_pucpr_repaired_split_50_50.sh that
# swaps in the per-track-balanced benchmark preset so each track contributes
# roughly equally to the test-side metrics (instead of the flat-pool top-1000
# shuffle used by the baseline).
#
# This wrapper still calls sweep_train_eval.sh, which gates training behind
# the existence of $MODEL_DIR/persistence_params.json — so as long as the
# baseline 50/50 sweep has already produced checkpoints for every (variant,
# fraction, seed) cell, no retraining happens. Missing cells WILL train from
# scratch unless you intervene first. Verify checkpoints exist (e.g. by
# pre-checking /workspace/data/output/perpetua/perpetua_*_pucpr_repaired_split_50_50/)
# before running, or set a sentinel and bail.
#
# Outputs go to a separate CSV so the balanced metrics don't overwrite the
# flat-pool baseline. Eval artifacts also land in a separate dataset subdir
# under persistence_eval_outputs because the balanced benchmark_dataset has a
# distinct `name:` field.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split/splits/50_50

# Train preset is unchanged — checkpoints from the baseline sweep are reused.
# Benchmark preset swaps to the balanced variant; its `name:` field
# (..._balanced) drives a fresh eval-output dataset subdir.
export TRAIN_PRESET="${TRAIN_PRESET:-annotation_tool_pklot_repaired_split_train}"
export BENCH_PRESET="${BENCH_PRESET:-annotation_tool_pucpr_repaired_split_test_balanced}"

export TEST_SIDECAR="${TEST_SIDECAR:-$SPLIT_DIR/test_from_2012-10-16.track_id_to_location_cell${CELL_SIZE:-128}.json}"

# Keep balanced metrics in a distinct CSV.
export OUT_CSV="${OUT_CSV:-/workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_repaired_split_50_50_balanced.csv}"

# EXP_SUFFIX is intentionally identical to the baseline 50/50 sweep so the
# model_dir lookup hits the existing trained checkpoints. The eval-side
# dataset subdir is what changes (driven by BENCH_PRESET's `name:`).
export EXP_SUFFIX="${EXP_SUFFIX:-_pucpr_repaired_split_50_50}"

exec bash "$SCRIPT_DIR/sweep_train_eval.sh" "$@"
