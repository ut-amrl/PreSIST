#!/usr/bin/env bash
#set -euo pipefail
set -eo pipefail

#########################
# Configurable paths
#########################

# Apptainer image
IMAGE="${IMAGE:-$SCRATCH/containers/spf_latest.sif}"

# Host directories
CODE_DIR="${CODE_DIR:-$WORK/workspaces/semantic-persistence-filter}"
DATA_DIR="${DATA_DIR:-$SCRATCH/data/semantic-persistence-filter}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$SCRATCH/.cache/huggingface}"

# Container mount points
CODE_MNT="/workspace/code/semantic-persistence-filter"
DATA_MNT="/workspace/data"
HF_MNT="/opt/hf"

#########################
# Sanity checks
#########################

if [ ! -f "$IMAGE" ]; then
  echo "ERROR: Apptainer image not found at: $IMAGE"
  echo "Set IMAGE=/path/to/spf.sif or fix the default in the script."
  exit 1
fi

mkdir -p "$CODE_DIR" "$DATA_DIR" "$HF_CACHE_DIR"

if ! command -v apptainer >/dev/null 2>&1; then
  echo "ERROR: 'apptainer' not found in PATH."
  echo "On Stampede3 you probably need: module load tacc-apptainer"
  exit 1
fi

#########################
# Build common Apptainer args
#########################

APPTAINER_BASE_ARGS=(
  --nv
  -B "${CODE_DIR}:${CODE_MNT}"
  -B "${DATA_DIR}:${DATA_MNT}"
  -B "${HF_CACHE_DIR}:${HF_MNT}"
)

#########################
# Behavior:
#   - No args: interactive shell
#   - With args: run command
#########################

if [ "$#" -eq 0 ]; then
  echo "Launching interactive shell in container..."
  echo "  Image:   $IMAGE"
  echo "  Code:    $CODE_DIR -> $CODE_MNT"
  echo "  Data:    $DATA_DIR -> $DATA_MNT"
  echo "  HF cache:$HF_CACHE_DIR -> $HF_MNT"
  echo ""

  apptainer shell "${APPTAINER_BASE_ARGS[@]}" "$IMAGE"

else
  # Treat all arguments as a single command string
  CMD="$*"

  echo "Executing in container:"
  echo "  Image:   $IMAGE"
  echo "  Command: $CMD"
  echo "  Code:    $CODE_DIR -> $CODE_MNT"
  echo "  Data:    $DATA_DIR -> $DATA_MNT"
  echo "  HF cache:$HF_CACHE_DIR -> $HF_MNT"
  echo ""

  # Use bash -lc so things like env vars and multiple args work
  apptainer exec "${APPTAINER_BASE_ARGS[@]}" "$IMAGE" bash -lc "$CMD"
fi