#!/bin/bash
# Day-count train+eval wrapper for the Perpetua family on the 50/20/30
# split of PUCPR_repaired_split_flanks_fixed. Sibling of
# sweep_train_eval_pucpr_split_50_50_balanced.sh (fraction sweep) — this
# one varies the number of contiguous days of training data instead.
#
# The day-count subsetting must be materialized first via:
#   python generate_daycount_train_manifests.py \
#     --source-train-manifest .../splits/50_20_30/train_through_2012-10-15.json \
#     --days 1 2 3 4 5 7 10 14 21 28 35

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30

export DAYCOUNT_ROOT="${DAYCOUNT_ROOT:-$SPLIT_DIR/daycount}"

export TRAIN_PRESET="${TRAIN_PRESET:-annotation_tool_pklot_split_50_20_30_train}"
export BENCH_PRESET="${BENCH_PRESET:-annotation_tool_pucpr_split_50_20_30_test_balanced}"

export TEST_MANIFEST="${TEST_MANIFEST:-$SPLIT_DIR/test_from_2012-10-30.json}"
export TEST_SIDECAR="${TEST_SIDECAR:-$SPLIT_DIR/test_from_2012-10-30.track_id_to_location_cell${CELL_SIZE:-128}.json}"

export OUT_CSV="${OUT_CSV:-/workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_split_50_20_30_daycount_balanced.csv}"

export TRAIN_CFG_TEMPLATE="train_config_perpetua_{DIST}_{GROUPING}_pucpr_split_50_20_30_daycount"
export EVAL_CFG_TEMPLATE="eval_config_perpetua_{DIST}_{GROUPING}_pucpr_split_50_20_30_daycount_balanced"

# experiment.name in the new daycount cfgs is already
# `perpetua_<dist>_<grouping>_pucpr_split_50_20_30_daycount`, so EXP_SUFFIX
# is empty (the sweep driver appends DAYS_SLUG/seed under that).
export EXP_SUFFIX="${EXP_SUFFIX:-_pucpr_split_50_20_30_daycount}"

exec bash "$SCRIPT_DIR/sweep_train_eval_daycount.sh" "$@"
