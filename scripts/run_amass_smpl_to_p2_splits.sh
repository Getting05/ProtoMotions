#!/bin/bash
# Retarget selected packaged AMASS SMPL splits to Astro P2.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/chenguanting/ProtoMotions}"
INPUT_ROOT="${INPUT_ROOT:-/data/chenguanting/datasets/AMASS/smpl_motionlib}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/chenguanting/datasets/AMASS/p2_motionlib}"
PROTO_PYTHON="${PROTO_PYTHON:-/data/chenguanting/.venvs/protomotions-mujoco/bin/python}"
PYROKI_PYTHON="${PYROKI_PYTHON:-/data/chenguanting/.venvs/astro-p2-retarget/bin/python}"
# One worker per physical GPU gives the best measured throughput for 450-frame solves.
RETARGET_GPUS="${RETARGET_GPUS:-0,1,2,3,4,5,6,7}"

if [ $# -eq 0 ]; then
    set -- test validation train
fi

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT"
for SPLIT in "$@"; do
    INPUT_FILE="$INPUT_ROOT/amass_smpl_${SPLIT}.pt"
    SPLIT_OUTPUT="$OUTPUT_ROOT/$SPLIT"
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Error: missing input $INPUT_FILE"
        exit 1
    fi
    mkdir -p "$SPLIT_OUTPUT"
    echo "[$(date --iso-8601=seconds)] Starting $SPLIT"
    ./scripts/retarget_amass_to_robot.sh \
        "$PROTO_PYTHON" \
        "$PYROKI_PYTHON" \
        "$INPUT_FILE" \
        astro_p2 1 \
        --source-skeleton smpl \
        --retarget-gpus "$RETARGET_GPUS" \
        --output-root "$SPLIT_OUTPUT" \
        2>&1 | tee -a "$SPLIT_OUTPUT/pipeline.log"
    test -s "$SPLIT_OUTPUT/proto-astro_p2.pt"
    ln -sfn "$SPLIT_OUTPUT/proto-astro_p2.pt" \
        "$OUTPUT_ROOT/amass_smpl_${SPLIT}_astro_p2.pt"
    echo "[$(date --iso-8601=seconds)] Completed $SPLIT"
done
