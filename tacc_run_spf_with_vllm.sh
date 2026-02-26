#!/usr/bin/env bash
set -eo pipefail

mkdir -p logs/spf

echo "=== SPF + vLLM driver ==="
echo "  Date:   $(date)"
echo "  Host:   $(hostname)"
echo

############################################
# Modules (work both in batch and interactive)
############################################

module load tacc-apptainer
# module load nvidia   # if needed on your queue

############################################
# Paths (can override via env)
############################################

export IMAGE="${IMAGE:-$SCRATCH/containers/spf_latest.sif}"
export CODE_DIR="${CODE_DIR:-$WORK/workspaces/semantic-persistence-filter}"
export DATA_DIR="${DATA_DIR:-$SCRATCH/data/semantic-persistence-filter}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$WORK/.cache/huggingface}"

cd "$CODE_DIR"

echo "Using:"
echo "  Image:    $IMAGE"
echo "  Code dir: $CODE_DIR"
echo "  Data dir: $DATA_DIR"
echo "  HF cache: $HF_CACHE_DIR"
echo

############################################
# micromamba + vllm env activation
############################################

# Root prefix for micromamba (where you told it to install envs)
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$WORK/micromamba}"

# Autodetect micromamba executable
if command -v micromamba >/dev/null 2>&1; then
    MAMBA_EXE="$(command -v micromamba)"
elif [ -x "$WORK/.local/bin/micromamba" ]; then
    MAMBA_EXE="$WORK/.local/bin/micromamba"
elif [ -x "$HOME/.local/bin/micromamba" ]; then
    MAMBA_EXE="$HOME/.local/bin/micromamba"
else
    echo "ERROR: micromamba not found."
    echo "Make sure it's on your PATH or installed at:"
    echo "  \$WORK/.local/bin/micromamba"
    exit 1
fi

echo "Using micromamba at: $MAMBA_EXE"
echo "MAMBA_ROOT_PREFIX:   $MAMBA_ROOT_PREFIX"
echo

# Initialize micromamba shell hook
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"

ENV_NAME="${ENV_NAME:-vllm2}"

echo "Activating micromamba env '$ENV_NAME'..."
if ! micromamba activate "$ENV_NAME"; then
    echo "ERROR: Failed to activate micromamba env '$ENV_NAME'."
    echo "Make sure you created it (e.g. 'micromamba create -n vllm ...')"
    exit 1
fi


echo "After activation:"
echo "  PATH: $PATH"
which python || echo "python not found on PATH"
which vllm || echo "vllm not found on PATH"

python -c "import sys; print('Python:', sys.executable)" || echo "Python failed"
python -c "import vllm; print('vLLM module:', vllm.__file__)" || echo "Python vllm import failed"

pip list

# At this point, we assume vllm is already installed in the env.
# No creation / pip install is done here.

############################################
# vLLM configuration (model, port, naming)
############################################

# Overridable via env:
#   export VLLM_MODEL="..."
#   export VLLM_PORT=...
#   export VLLM_TP=...
#   export VLLM_DTYPE=...
#   export VLLM_QUANT="awq"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-30B-A3B-Thinking}"
# VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-235B-A22B-Instruct}"
VLLM_PORT="${VLLM_PORT:-9006}"
VLLM_TP="${VLLM_TP:-8}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_QUANT="${VLLM_QUANT:-}"   # e.g. "awq" or "" for none

# Derive served model name like your start_vllm_server.sh
MODEL_BASENAME="$(basename "$VLLM_MODEL")"
MODEL_NORMALIZED="$(echo "$MODEL_BASENAME" | tr '[:upper:]' '[:lower:]')"

VLLM_SERVED_NAME="${MODEL_NORMALIZED}__${VLLM_DTYPE}"
if [ -n "$VLLM_QUANT" ]; then
  VLLM_SERVED_NAME="${VLLM_SERVED_NAME}__${VLLM_QUANT}"
fi

echo "=== vLLM config ==="
echo "  Model:             $VLLM_MODEL"
echo "  Port:              $VLLM_PORT"
echo "  Tensor parallel:   $VLLM_TP"
echo "  Dtype:             $VLLM_DTYPE"
echo "  Quantization:      ${VLLM_QUANT:-<none>}"
echo "  Served model name: $VLLM_SERVED_NAME"
echo

# Use $WORK for HF cache (avoid $HOME quota)
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"
mkdir -p "$HF_HOME"

# Optional: OpenAI-style env for client code
export OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export OPENAI_API_KEY="unused"
export SPF_VLLM_MODEL_NAME="$VLLM_SERVED_NAME"

############################################
# Start vLLM in the background
############################################

VLLM_LOG="logs/spf/vllm_${SLURM_JOB_ID:-interactive}.out"

echo "Starting vLLM server (logs: $VLLM_LOG)..."
vllm serve "$VLLM_MODEL" \
  --port "$VLLM_PORT" \
  --tensor-parallel-size "$VLLM_TP" \
  --gpu-memory-utilization 0.9 \
  --dtype "$VLLM_DTYPE" \
  --served-model-name "$VLLM_SERVED_NAME" \
  ${VLLM_QUANT:+--quantization "$VLLM_QUANT"} \
  >"$VLLM_LOG" 2>&1 &

VLLM_PID=$!

cleanup() {
  if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Cleaning up vLLM server (PID $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

############################################
# Wait for vLLM to be ready
############################################

echo "Waiting for vLLM server on port $VLLM_PORT ..."
READY=0
for i in $(seq 1 120); do
  if curl -s "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "vLLM server is ready (waited ~$((i * 10)) seconds)"
    READY=1
    break
  fi

  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM server process exited unexpectedly while starting."
    echo "Check logs in: $VLLM_LOG"
    exit 1
  fi

  sleep 10
done

if [ "$READY" -ne 1 ]; then
  echo "ERROR: vLLM server did not become ready in time."
  echo "Check logs in: $VLLM_LOG"
  exit 1
fi

echo
echo "vLLM is up. Starting containerized pipeline..."
echo

############################################
# Run your Apptainer pipeline
############################################

chmod +x "$CODE_DIR/tacc_run_spf_container.sh"

# If you want to split GPUs between vLLM and the pipeline, set:
#   export CUDA_VISIBLE_DEVICES="4,5,6,7"
# here (and configure vLLM with different GPUs earlier).

CONTAINER_CMD="cd /workspace/code/semantic-persistence-filter && \
HYDRA_FULL_ERROR=1 python3 run_llm_labeling_pipeline.py \
  --config-name qwen3_vl_30b_thinking_v2_b_semantic_sam_minArea0p001 \
  +image_dataset=openloris_location_all \
  ++dataset_generator.llmPersistenceQuerier.vlmQuerier.use_vllm_server=True \
  ++dataset_generator.llmPersistenceQuerier.vlmQuerier.vllm_port=${VLLM_PORT}"

echo "Running container command:"
echo "  $CONTAINER_CMD"
echo

IMAGE="$IMAGE" CODE_DIR="$CODE_DIR" DATA_DIR="$DATA_DIR" HF_CACHE_DIR="$HF_CACHE_DIR" \
  "$CODE_DIR/tacc_run_spf_container.sh" "$CONTAINER_CMD"

PIPELINE_EXIT=$?

echo
echo "Pipeline finished with exit code: $PIPELINE_EXIT"
echo "End time: $(date)"

exit $PIPELINE_EXIT
