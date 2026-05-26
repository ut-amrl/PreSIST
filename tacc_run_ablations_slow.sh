#!/usr/bin/env bash
set -eo pipefail

# ============================================================
# Slow-ablation wrapper around tacc_run_ablations_inner.sh.
# 3 configs on a single-node allocation (4 H100s).
# 3 GPUs busy, 1 idle.
#
# Invocation: from inside the interactive Slurm allocation (or via
# srun if the user prefers to attach a task explicitly),
#   ./tacc_run_ablations_slow.sh
# Override any var inline, e.g.:
#   ABLATION_LOG_ROOT=$SCRATCH/foo ./tacc_run_ablations_slow.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

############################################
# What to run
############################################

# Four ablations on 4 GPUs. The slow allocation owns all three pooler
# variants plus __no_pretrain — keeps the pooler comparison on one node
# and saturates the GPUs.
#   __no_pretrain                       : full ViT-L unfrozen, bs=16,
#                                         random init. By far the slowest
#                                         of the 9 ablations.
#   __pooler_local_map                  : backbone produces an extra dense
#                                         local-map output on every forward.
#   __pooler_dilated_token_soft         : dilated-token gather is heavier
#                                         than the mask_downsample variants.
#   __pooler_mask_downsample_nearest    : nearest vs Ours' bilinear mask
#                                         downsample. Per-step ~ Ours;
#                                         grouped here for the pooler
#                                         comparison.
CONFIGS_STR="${CONFIGS_STR:-\
cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__no_pretrain \
cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__pooler_local_map \
cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__pooler_dilated_token_soft \
cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50__pooler_mask_downsample_nearest}"

############################################
# Paths (override via env)
############################################

IMAGE="${IMAGE:-$SCRATCH/containers/spf_latest.sif}"
CODE_DIR="${CODE_DIR:-$WORK/workspaces/semantic-persistence-filter}"
DATA_DIR="${DATA_DIR:-$SCRATCH/data/semantic-persistence-filter}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$SCRATCH/.cache/huggingface}"
JOB_ROOT="${JOB_ROOT:-$SCRATCH/jobs/spf-ablations/${SLURM_JOB_ID:-interactive}}"
ABLATION_LOG_ROOT="${ABLATION_LOG_ROOT:-$JOB_ROOT/ablation_logs_slow}"

CONFIGS_STR="$CONFIGS_STR" \
IMAGE="$IMAGE" \
CODE_DIR="$CODE_DIR" \
DATA_DIR="$DATA_DIR" \
HF_CACHE_DIR="$HF_CACHE_DIR" \
JOB_ROOT="$JOB_ROOT" \
ABLATION_LOG_ROOT="$ABLATION_LOG_ROOT" \
"$SCRIPT_DIR/tacc_run_ablations_inner.sh" "$@"
