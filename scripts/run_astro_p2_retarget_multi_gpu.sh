#!/bin/bash
# Run one independent Astro P2 retarget shard per GPU.
set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <pyroki_python> <keypoints_dir> <output_dir> [retargeter arguments...]"
    echo "Set ASTRO_P2_GPUS to a comma-separated GPU list (default: 0,1,2,3,4,5,6,7)."
    exit 1
fi

PYROKI_PYTHON="$1"
KEYPOINTS_DIR="$2"
OUTPUT_DIR="$3"
shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_CSV="${ASTRO_P2_GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_IDS <<< "$GPU_CSV"
NUM_SHARDS="${#GPU_IDS[@]}"

if [ "$NUM_SHARDS" -lt 1 ]; then
    echo "Error: ASTRO_P2_GPUS does not contain any GPU IDs"
    exit 1
fi

mkdir -p "$OUTPUT_DIR/logs"
PIDS=()
for SHARD_INDEX in "${!GPU_IDS[@]}"; do
    GPU_ID="${GPU_IDS[$SHARD_INDEX]}"
    if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
        echo "Error: invalid GPU ID '$GPU_ID' in ASTRO_P2_GPUS"
        exit 1
    fi
    LOG_PATH="$OUTPUT_DIR/logs/gpu_${GPU_ID}_shard_${SHARD_INDEX}.log"
    echo "Starting shard $SHARD_INDEX/$((NUM_SHARDS - 1)) on GPU $GPU_ID; log: $LOG_PATH"
    ASTRO_P2_RETARGET_DEVICE=gpu CUDA_VISIBLE_DEVICES="$GPU_ID" \
        "$SCRIPT_DIR/run_astro_p2_retarget.sh" "$PYROKI_PYTHON" \
        --keypoints-folder-path "$KEYPOINTS_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$SHARD_INDEX" \
        --no-visualize \
        --skip-existing \
        "$@" >"$LOG_PATH" 2>&1 &
    PIDS+=("$!")
done

FAILED=0
for SHARD_INDEX in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$SHARD_INDEX]}"; then
        echo "Shard $SHARD_INDEX failed; inspect $OUTPUT_DIR/logs/."
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    exit 1
fi
echo "All $NUM_SHARDS Astro P2 retarget shards completed."
