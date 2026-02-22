#!/usr/bin/env bash

# ==============================
# DAM Server Launch Script
# ==============================

# ---- CONFIG ----
MODEL_PATH="nvidia/DAM-3B"
CONV_MODE="v1"
PROMPT_MODE="focal_prompt"
PORT=9002
HOST="0.0.0.0"

# ---- Ensure CUDA_VISIBLE_DEVICES is preserved ----
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "WARNING: CUDA_VISIBLE_DEVICES not set. Using all visible GPUs."
else
    echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi

echo "Starting DAM server..."
echo "Model: $MODEL_PATH"
echo "Conv mode: $CONV_MODE"
echo "Prompt mode: $PROMPT_MODE"
echo "Host: $HOST"
echo "Port: $PORT"
echo "==========================================="

python3 ../../describe-anything/dam_server.py \
    --model-path "$MODEL_PATH" \
    --conv-mode "$CONV_MODE" \
    --prompt-mode "$PROMPT_MODE" \
    --host "$HOST" \
    --port "$PORT"
