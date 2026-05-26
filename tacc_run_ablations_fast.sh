#!/usr/bin/env bash
set -eo pipefail

# ============================================================
# Fast-ablation wrapper around tacc_run_ablations_inner.sh.
# 5 configs on a single-node allocation (4 H100s). The first 4 in the
# list start immediately on GPUs 0-3; run_ablations.py's job pool
# queues the 5th and dispatches it as a GPU frees up.
#
# Invocation: from inside the interactive Slurm allocation (or via
# srun if the user prefers to attach a task explicitly),
#   ./tacc_run_ablations_fast.sh
# Override any var inline, e.g.:
#   ABLATION_LOG_ROOT=$SCRATCH/foo ./tacc_run_ablations_fast.sh
#
# WARNING: __p25p75_linear and __p25p75_loggap currently carry
# placeholder z-score stats (???). Run scripts/compute_target_stats.py
# once per transform and paste the numbers into their config files
# before submitting, otherwise those two runs will fail at config-load
# time. The other three configs run regardless. p25p75_loggap is queued
# last (position 5) so any failure doesn't block the first four GPUs.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

############################################
# What to run (order matters: positions 1-4 start immediately,
# position 5 queues and dispatches on free GPU)
############################################

CONFIGS_STR="${CONFIGS_STR:-\
cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__no_conf \
cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__no_adapter \
cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__masked_only \
cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__p25p75_linear \
cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__p25p75_loggap}"

############################################
# Paths (override via env)
############################################

IMAGE="${IMAGE:-$SCRATCH/containers/spf_latest.sif}"
CODE_DIR="${CODE_DIR:-$WORK/workspaces/semantic-persistence-filter}"
DATA_DIR="${DATA_DIR:-$SCRATCH/data/semantic-persistence-filter}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$SCRATCH/.cache/huggingface}"
JOB_ROOT="${JOB_ROOT:-$SCRATCH/jobs/spf-ablations/${SLURM_JOB_ID:-interactive}}"
ABLATION_LOG_ROOT="${ABLATION_LOG_ROOT:-$JOB_ROOT/ablation_logs_fast}"

CONFIGS_STR="$CONFIGS_STR" \
IMAGE="$IMAGE" \
CODE_DIR="$CODE_DIR" \
DATA_DIR="$DATA_DIR" \
HF_CACHE_DIR="$HF_CACHE_DIR" \
JOB_ROOT="$JOB_ROOT" \
ABLATION_LOG_ROOT="$ABLATION_LOG_ROOT" \
"$SCRIPT_DIR/tacc_run_ablations_inner.sh" "$@"
