#!/bin/bash
# Wrapper around sweep_train_eval.sh that targets the PUCPR_repaired_split
# annotations (export_augmented.json) with a 50/50 temporal split.
#
# The underlying sweep supports TRAIN_PRESET / BENCH_PRESET (Hydra group
# overrides), TRAIN_MANIFEST / TEST_MANIFEST (manifest_path overrides on the
# active preset), TEST_SIDECAR, EXP_SUFFIX, and OUT_CSV. We swap in the
# 50/50-split presets built by generate_temporal_split.py --train-fraction 0.5
# against
#   /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/new_pklot/
#     postprocessed/annotations/PUCPR_repaired_split/export_augmented.json
#
# Any of the upstream env vars (FRACTIONS, SEED, CELL_SIZE, VARIANTS,
# EVAL_SCENARIO, MIXTURE_FAST, FORCE_TRAIN, FORCE_EVAL, OUT_CSV) can still be
# set by the caller and will pass through.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split/splits/50_50

# Swap the Hydra group presets wholesale so that:
#   - train_dataset.name is `pklot_pucpr_repaired_split_train_50_50`
#   - benchmark_dataset.name is `pucpr_repaired_split_new_pklot_PUCPR_test_50_50`
# This makes the eval output land under a top-level dataset subdir that
# identifies the 50/50 split, matching the layout used by other eval configs
# (e.g. eval_config_closed_set_clip_iqr_percentile.yaml).
export TRAIN_PRESET="${TRAIN_PRESET:-annotation_tool_pklot_repaired_split_train}"
export BENCH_PRESET="${BENCH_PRESET:-annotation_tool_pucpr_repaired_split_test}"

# Sidecar still has to exist on disk; the upstream script's existence check
# uses TEST_SIDECAR directly (it doesn't peek into the BENCH_PRESET yaml).
export TEST_SIDECAR="${TEST_SIDECAR:-$SPLIT_DIR/test_from_2012-10-16.track_id_to_location_cell${CELL_SIZE:-128}.json}"

export OUT_CSV="${OUT_CSV:-/workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_repaired_split_50_50.csv}"

# Keep model_dir / eval-output paths separate from the prior 70/30 sweep so
# we don't reuse stale trained models or stale metric_results.json.
export EXP_SUFFIX="${EXP_SUFFIX:-_pucpr_repaired_split_50_50}"

exec bash "$SCRIPT_DIR/sweep_train_eval.sh" "$@"
