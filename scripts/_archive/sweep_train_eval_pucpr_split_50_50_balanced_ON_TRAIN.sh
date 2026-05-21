#!/bin/bash
# Diagnostic eval-on-train sibling of
# sweep_train_eval_pucpr_split_50_50_balanced.sh. Same trained checkpoints
# (training is skipped), but evaluated against the per-track-balanced TRAIN
# benchmark (annotation_tool_pucpr_split_train_balanced) via the ON_TRAIN
# eval configs.
#
# FORCE_EVAL=1 is required because the eval-skip glob in sweep_train_eval.sh
# isn't scoped by benchmark dataset and would otherwise reuse the test-side
# metric_results.json. Output CSV is suffixed _ON_TRAIN so it lives alongside
# the test-side CSV without colliding.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_50

export TRAIN_PRESET="${TRAIN_PRESET:-annotation_tool_pklot_split_train}"
export BENCH_PRESET="${BENCH_PRESET:-annotation_tool_pucpr_split_train_balanced}"

# Point the upstream existence check + manifest_path override at the TRAIN
# manifest, and feed the train-side sidecar to the location-keyed inferencer.
export TEST_MANIFEST="${TEST_MANIFEST:-$SPLIT_DIR/train_through_2012-10-15.json}"
export TEST_SIDECAR="${TEST_SIDECAR:-$SPLIT_DIR/train_through_2012-10-15.track_id_to_location_cell${CELL_SIZE:-128}.json}"

export OUT_CSV="${OUT_CSV:-/workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_split_50_50_balanced_ON_TRAIN.csv}"

export TRAIN_CFG_TEMPLATE="train_config_perpetua_{DIST}_{GROUPING}_pucpr_split_50_50"
export EVAL_CFG_TEMPLATE="eval_config_perpetua_{DIST}_{GROUPING}_pucpr_split_50_50_balanced_ON_TRAIN"

export EXP_SUFFIX="${EXP_SUFFIX:-_pucpr_split_50_50}"

# Required: prevent reuse of test-side metric_results.json (eval-skip glob
# is not benchmark-scoped).
export FORCE_EVAL="${FORCE_EVAL:-1}"

exec bash "$SCRIPT_DIR/sweep_train_eval.sh" "$@"
