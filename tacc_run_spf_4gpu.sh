#!/usr/bin/env bash
set -eo pipefail

# ============================================================
# 4-GPU wrapper around
#   tacc_revised_pgid_run_spf_trtllm_dam_seg_cap_desc_multi.sh
# Override any var inline, e.g.:
#   GEMMA_TP=4 ./tacc_run_spf_4gpu.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

############################################
# What to run
############################################

CONFIG_NAME="${CONFIG_NAME:-q3_30bt_lang_q3T_pU3_sam3b_maxOb_30_mConf_0p35_mA0p001_dd0p9_capQ38bTproc2_idGemmaBBgCCMTD1HL_v3_iqr_only_behav}"
RUN_PHASES="${RUN_PHASES:-object_descriptions}"
DATASETS="${DATASETS:-train_mix_3 train_mix_4 train_mix_5 train_mix_6 train_mix_7}"

############################################
# Per-phase backend routing
#   qwen | gemma | dam | "" (use config default)
############################################

LANG_PROMPT_BACKEND="${LANG_PROMPT_BACKEND:-gemma}"
SCENE_CAPTION_BACKEND="${SCENE_CAPTION_BACKEND:-gemma}"
OBJECT_DESC_BACKEND="${OBJECT_DESC_BACKEND:-gemma}"
PERSISTENCE_BACKEND="${PERSISTENCE_BACKEND:-}"

############################################
# GPU allocation (4-GPU machine, all-Gemma)
############################################

TOTAL_GPUS="${TOTAL_GPUS:-4}"
QWEN_GPUS="${QWEN_GPUS:-0}"
GEMMA_GPUS="${GEMMA_GPUS:-4}"
# TP per Gemma replica; DP is derived = GEMMA_GPUS / GEMMA_TP.
# TP=2 -> 2 replicas; TP=4 -> 1 replica.
GEMMA_TP="${GEMMA_TP:-2}"

############################################
# Models
############################################

GEMMA_MODEL="${GEMMA_MODEL:-google/gemma-4-31B-it}"
TRTLLM_MODEL="${TRTLLM_MODEL:-Qwen/Qwen3-VL-8B-Thinking}"

############################################
# Runtime knobs
############################################

MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
WALL_HOURS="${WALL_HOURS:-48}"
DAM_WORKERS="${DAM_WORKERS:-3}"
# Pipeline workers when segmentation isn't in RUN_PHASES; match GPU count.
CPU_WORKERS="${CPU_WORKERS:-$TOTAL_GPUS}"

############################################
# Rarely changed
############################################

GEMMA_MIN_GPUS_PER_REPLICA="${GEMMA_MIN_GPUS_PER_REPLICA:-2}"

CONFIG_NAME="$CONFIG_NAME" \
RUN_PHASES="$RUN_PHASES" \
DATASETS="$DATASETS" \
LANG_PROMPT_BACKEND="$LANG_PROMPT_BACKEND" \
SCENE_CAPTION_BACKEND="$SCENE_CAPTION_BACKEND" \
OBJECT_DESC_BACKEND="$OBJECT_DESC_BACKEND" \
PERSISTENCE_BACKEND="$PERSISTENCE_BACKEND" \
TOTAL_GPUS="$TOTAL_GPUS" \
QWEN_GPUS="$QWEN_GPUS" \
GEMMA_GPUS="$GEMMA_GPUS" \
GEMMA_TP="$GEMMA_TP" \
GEMMA_MODEL="$GEMMA_MODEL" \
TRTLLM_MODEL="$TRTLLM_MODEL" \
MAX_SEQ_LEN="$MAX_SEQ_LEN" \
WALL_HOURS="$WALL_HOURS" \
DAM_WORKERS="$DAM_WORKERS" \
CPU_WORKERS="$CPU_WORKERS" \
GEMMA_MIN_GPUS_PER_REPLICA="$GEMMA_MIN_GPUS_PER_REPLICA" \
"$SCRIPT_DIR/tacc_revised_pgid_run_spf_trtllm_dam_seg_cap_desc_multi.sh" "$@"
