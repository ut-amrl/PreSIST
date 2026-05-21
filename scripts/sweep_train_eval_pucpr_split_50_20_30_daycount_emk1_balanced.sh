#!/bin/bash
# K=1 EM variant of sweep_train_eval_pucpr_split_50_20_30_daycount_balanced.sh.
# Replaces the naive-MLE single-component trainers with their K=1 EM
# counterparts (upstream perpetua's em_step with n_components=1), which
# handles right-censored tracks correctly via the persistence-filter forward
# pass.
#
# VARIANTS defaults to the four single-component combinations only. Mixture
# variants don't have a fit_method flag — run them via the regular daycount
# wrapper (sweep_train_eval_pucpr_split_50_20_30_daycount_balanced.sh). The
# regular wrapper's mixture cells and this wrapper's single cells together
# give a full picture; their model_dirs do not collide because experiment.name
# differs (_daycount vs _daycount_emk1).
#
# Must be run inside spf_perpetua_container (jax required for upstream EM).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SPLIT_DIR=/workspace/data/input/new_pklot/postprocessed/annotations/PUCPR_repaired_split_flanks_fixed/splits/50_20_30

export DAYCOUNT_ROOT="${DAYCOUNT_ROOT:-$SPLIT_DIR/daycount}"

export TRAIN_PRESET="${TRAIN_PRESET:-annotation_tool_pklot_split_50_20_30_train}"
export BENCH_PRESET="${BENCH_PRESET:-annotation_tool_pucpr_split_50_20_30_test_balanced}"

export TEST_MANIFEST="${TEST_MANIFEST:-$SPLIT_DIR/test_from_2012-10-30.json}"
export TEST_SIDECAR="${TEST_SIDECAR:-$SPLIT_DIR/test_from_2012-10-30.track_id_to_location_cell${CELL_SIZE:-128}.json}"

export OUT_CSV="${OUT_CSV:-/workspace/data/output/persistence_eval_outputs/sweep_metrics_pucpr_split_50_20_30_daycount_emk1_balanced.csv}"

# Singles only — the only variants that have a fit_method axis.
export VARIANTS="${VARIANTS:-single_exp:location single_exp:class single_lognormal:location single_lognormal:class}"

export TRAIN_CFG_TEMPLATE="train_config_perpetua_{DIST}_{GROUPING}_pucpr_split_50_20_30_daycount_emk1"
export EVAL_CFG_TEMPLATE="eval_config_perpetua_{DIST}_{GROUPING}_pucpr_split_50_20_30_daycount_emk1_balanced"

# experiment.name in the _emk1 cfgs is `perpetua_<dist>_<grouping>_pucpr_split_50_20_30_daycount_emk1`,
# so EXP_SUFFIX matches and the sweep driver's path construction lands at the
# right model_dir.
export EXP_SUFFIX="${EXP_SUFFIX:-_pucpr_split_50_20_30_daycount_emk1}"

exec bash "$SCRIPT_DIR/sweep_train_eval_daycount.sh" "$@"
