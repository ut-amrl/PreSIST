#!/usr/bin/env bash
#SBATCH -J sleep-node                            # Job name
#SBATCH -o logs/spf/sleep-node.%j.out           # Stdout
#SBATCH -e logs/spf/sleep-node.%j.err           # Stderr
#SBATCH -p amd-rtx                               # Partition / queue
#SBATCH -N 1                                     # Number of nodes
#SBATCH -n 8                                     # Number of tasks (= GPUs per amd-rtx node)
#SBATCH -t 48:00:00                              # Wall time
#SBATCH -A IRI26004                              # Allocation

echo "Node reserved: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Sleeping for 48 hours..."

sleep 48h
