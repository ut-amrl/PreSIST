#!/bin/bash

usage() {
    echo "Usage: $0 -g <gpu_ids> -c <command> [-t <threshold_mib>] [-p <poll_interval_s>] [-w <confirm_wait_s>]"
    echo "  -g  Comma-separated GPU IDs to wait on (e.g. 1,2), or 'any' / 'any:N' to grab the"
    echo "      first N free GPUs from all visible devices (default N=1)"
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
ANY_MODE=0
ANY_COUNT=1
CMD=""

while getopts "g:c:t:p:w:" opt; do
    case "$opt" in
        g)
            if [[ "$OPTARG" == any || "$OPTARG" == any:* ]]; then
                ANY_MODE=1
                [[ "$OPTARG" == any:* ]] && ANY_COUNT="${OPTARG#any:}"
            else
                IFS=',' read -ra GPU_IDS <<< "$OPTARG"
            fi
            ;;
        c) CMD="$OPTARG" ;;
        t) THRESHOLD_MIB="$OPTARG" ;;
        p) POLL_INTERVAL="$OPTARG" ;;
        w) CONFIRM_WAIT="$OPTARG" ;;
        *) usage ;;
    esac
done

[[ ${#GPU_IDS[@]} -eq 0 && $ANY_MODE -eq 0 ]] && { echo "Error: -g is required"; usage; }
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

find_any_free_gpus() {
    local need="$1"
    local found=()
    local all_ids
    all_ids=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null)
    for gpu_id in $all_ids; do
        if gpu_available "$gpu_id"; then
            found+=("$gpu_id")
            [[ ${#found[@]} -eq "$need" ]] && { echo "${found[*]}"; return 0; }
        fi
    done
    return 1
}

if [[ $ANY_MODE -eq 1 ]]; then
    echo "Polling all visible GPUs for $ANY_COUNT free device(s) (< ${THRESHOLD_MIB} MiB used)..."
    while true; do
        candidates=$(find_any_free_gpus "$ANY_COUNT") || candidates=""
        if [[ -n "$candidates" ]]; then
            read -ra GPU_IDS <<< "$candidates"
            echo "Candidate GPUs ${GPU_IDS[*]} look available ($(date)). Waiting ${CONFIRM_WAIT}s to confirm..."
            sleep "$CONFIRM_WAIT"
            if all_gpus_available; then
                echo "GPUs ${GPU_IDS[*]} confirmed still available ($(date)). Launching job..."
                export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_IDS[*]}")"
                break
            else
                echo "A candidate GPU became busy during confirmation window. Resuming poll..."
                GPU_IDS=()
            fi
        else
            echo "No $ANY_COUNT GPU(s) free yet. Retrying in ${POLL_INTERVAL}s... ($(date))"
        fi
        sleep "$POLL_INTERVAL"
    done
else
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
fi

eval "$CMD"