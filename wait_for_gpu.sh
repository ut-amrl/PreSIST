#!/bin/bash

usage() {
    echo "Usage: $0 -g <gpu_ids> -c <command> [-t <threshold_mib>] [-p <poll_interval_s>] [-w <confirm_wait_s>]"
    echo "  -g  Comma-separated GPU IDs to wait on (e.g. 1,2)"
    echo "  -c  Command to run once GPUs are available (quote if multi-word)"
    echo "  -t  Memory threshold in MiB (default: 10240)"
    echo "  -p  Poll interval in seconds (default: 120)"
    echo "  -w  Confirmation wait in seconds (default: 180)"
    exit 1
}

# Defaults
THRESHOLD_MIB=10240
POLL_INTERVAL=120
CONFIRM_WAIT=180
GPU_IDS=()
CMD=""

while getopts "g:c:t:p:w:" opt; do
    case "$opt" in
        g) IFS=',' read -ra GPU_IDS <<< "$OPTARG" ;;
        c) CMD="$OPTARG" ;;
        t) THRESHOLD_MIB="$OPTARG" ;;
        p) POLL_INTERVAL="$OPTARG" ;;
        w) CONFIRM_WAIT="$OPTARG" ;;
        *) usage ;;
    esac
done

[[ ${#GPU_IDS[@]} -eq 0 ]] && { echo "Error: -g is required"; usage; }
[[ -z "$CMD" ]] && { echo "Error: -c is required"; usage; }

gpu_available() {
    local gpu_id="$1"
    local used_mib
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null)
    [[ -n "$used_mib" && "$used_mib" -lt "$THRESHOLD_MIB" ]]
}

all_gpus_available() {
    for gpu_id in "${GPU_IDS[@]}"; do
        if ! gpu_available "$gpu_id"; then
            return 1
        fi
    done
    return 0
}

echo "Polling GPUs ${GPU_IDS[*]} for availability (all < ${THRESHOLD_MIB} MiB used)..."

while true; do
    if all_gpus_available; then
        echo "All GPUs ${GPU_IDS[*]} look available ($(date)). Waiting ${CONFIRM_WAIT}s to confirm..."
        sleep "$CONFIRM_WAIT"
        if all_gpus_available; then
            echo "All GPUs ${GPU_IDS[*]} confirmed still available ($(date)). Launching job..."
            export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_IDS[*]}")"
            break
        else
            echo "A GPU became busy during confirmation window. Resuming poll..."
        fi
    fi
    echo "Not all GPUs available yet. Retrying in ${POLL_INTERVAL}s... ($(date))"
    sleep "$POLL_INTERVAL"
done

eval "$CMD"