#!/usr/bin/env bash
#SBATCH -J spf-trtllm-dam                        # Job name
#SBATCH -o logs/spf/spf-trtllm-dam.%j.out       # Stdout
#SBATCH -e logs/spf/spf-trtllm-dam.%j.err       # Stderr
#SBATCH -p amd-rtx                               # Partition / queue
#SBATCH -N 1                                     # Number of nodes
#SBATCH -n 8                                     # Number of tasks (= GPUs per amd-rtx node)
#SBATCH -t 48:00:00                              # Wall time
#SBATCH -A IRI26004                              # Allocation

mkdir -p logs/spf

exec "$WORK/workspaces/semantic-persistence-filter/tacc_revised_pgid_run_spf_trtllm_dam_seg_cap_desc_multi.sh"
