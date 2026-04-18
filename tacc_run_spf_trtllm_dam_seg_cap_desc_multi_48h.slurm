#!/usr/bin/env bash
#SBATCH -J spf-multi                             # Job name
#SBATCH -o logs/spf/spf-multi.%j.out            # Stdout
#SBATCH -e logs/spf/spf-multi.%j.err            # Stderr
#SBATCH -p amd-rtx                               # Partition / queue
#SBATCH -N 1                                     # Number of nodes
#SBATCH -n 1                                     # Number of tasks
#SBATCH -t 48:00:00                              # Wall time
#SBATCH -A IRI26004                              # Allocation

set -eo pipefail

echo "=== SPF multi-worker seg+caption+desc (48 h) ==="
echo "Job ID: $SLURM_JOB_ID"
echo

# ── TRT-LLM / GPU knobs ──────────────────────────────────────────────────────
# TP controls GPU split; the rest automatically go to pipeline workers:
#   TP=1  →  1 TRT-LLM GPU + 1 DAM GPU + 6 pipeline workers
#   TP=2  →  2 TRT-LLM GPU + 1 DAM GPU + 5 pipeline workers  (default)
#   TP=4  →  4 TRT-LLM GPU + 1 DAM GPU + 3 pipeline workers
# export TRTLLM_TP=2
# export TOTAL_GPUS=8
# export TRTLLM_MODEL="Qwen/Qwen3-VL-8B-Thinking"
# export TRTLLM_PORT=9011
# export MAX_SEQ_LEN=32768
# export DAM_PORT=9014

# ── Datasets ─────────────────────────────────────────────────────────────────
# Space-separated list; workers round-robin through them in order.
# export DATASETS="train_mix_0 train_mix_1 train_mix_2 train_mix_3 train_mix_4 \
#                  train_mix_5 train_mix_6 train_mix_7 train_mix_8 train_mix_9 train_mix_10"

# ── Pipeline config ───────────────────────────────────────────────────────────
# Must use a sceneCaptioner backed by QwenVLSceneLevelCaptioner (not dummy).
# Configs with "cap" in their name have this pre-configured; see configs/dataset_gen_confs/.
# export CONFIG_NAME="q3_30bt_lang_q3T_pU3_sam3b_maxOb_30_mConf_0p35_mA0p001_dd0p9_capQ38bTlong_v3_iqr_only_behav"

CODE_DIR="${CODE_DIR:-$WORK/workspaces/semantic-persistence-filter}"
cd "$CODE_DIR"

export WALL_HOURS=48
./tacc_run_spf_trtllm_dam_seg_cap_desc_multi.sh
