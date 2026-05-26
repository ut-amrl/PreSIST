#!/usr/bin/env bash
# Canonical ablation config list in priority order (earliest first).
# Edit this list when adding/removing ablations.
#
# Usage:
#   ./run_all_ablations.sh                              # default GPUs 0,1,2,3
#   ./run_all_ablations.sh 0,1,2,3                      # explicit GPUs
#   ./run_all_ablations.sh 0,1,2,3 --force              # re-run completed configs
#   ./run_all_ablations.sh 0,1,2,3 --on-busy poll       # wait for busy GPUs
#   ./run_all_ablations.sh 0,1,2,3 -- +train_dataset.train_subsample_fraction=0.02 \
#                                     +train_dataset.val_subsample_fraction=0.1
#                                                       # smoke run across all
#
# Everything after the GPU list is forwarded to run_ablations.py.

set -euo pipefail

GPUS="${1:-0,1,2,3}"
shift || true

LOG_DIR="${LOG_DIR:-ablation_logs}"

# Reduce fragmentation OOMs on memory-tight GPUs. Per-ablation batch_size
# overrides handle the big cuts; this handles the long-tail fragment.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# On Blackwell (cap 12) there is no xformers memory_efficient_attention
# fp32 kernel. Set XFORMERS_DISABLED=1 in your environment before running
# this script in that case to fall back to the in-class PyTorch attention
# (see persistence_parameter_networks/.../dinov2/attention.py). H100 and
# below run xformers as normal.

OURS=cross_attn_seg_langSam3Q8T_behav4_mO20_imgbal_lowlr_huber_norm_loose50

# Code-touching ablations (masked_only, p25p75_linear, p25p75_loggap) are
# intentionally omitted until their code changes land. Add them here after.
CONFIGS=(
  # --- Priority 1: pooler variants (config-only, high-priority comparators) ---
  "${OURS}__pooler_mask_downsample_nearest"
  "${OURS}__pooler_dilated_token_soft"
  # feature_upsample crashes at 518x518: PyTorch upsample_bilinear2d_nhwc
  # has an INT_MAX element limit and the [B*K, 1024, 518, 518] target tensor
  # exceeds it. Would need batch_size<=1 to fit; not a fair comparison.
  # "${OURS}__pooler_feature_upsample"
  "${OURS}__pooler_local_map"
  # --- Priority 2: drop-confidence (config-only) ---
  "${OURS}__no_conf"
  # --- Priority 3: kicked off early because it trains slowly ---
  "${OURS}__no_pretrain"
  # --- Priority 4: no adapters (config-only) ---
  "${OURS}__no_adapter"
  # --- Priority 5: masked-only (Phase C, code-touching) ---
  "${OURS}__masked_only"
)

exec python run_ablations.py \
  --gpus "$GPUS" \
  --log-dir "$LOG_DIR" \
  --configs "${CONFIGS[@]}" \
  "$@"
