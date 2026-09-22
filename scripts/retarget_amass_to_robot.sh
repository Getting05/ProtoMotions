#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convenience script to retarget AMASS SMPL motions to a robot (G1, H1_2, or Astro P2)
#
# IMPORTANT: ProtoMotions and PyRoki require separate Python environments.
# You must provide paths to both Python interpreters.
#
# Usage: ./scripts/retarget_amass_to_robot.sh <proto_python> <pyroki_python> <amass_pt_file> <robot_type> [skip_freq] [--source-skeleton smpl|smplx] [--retarget-device cpu|gpu] [--retarget-gpus 0,1,...] [--output-root DIR] [--clean]
#
# Example:
#   ./scripts/retarget_amass_to_robot.sh \
#       ~/miniconda3/envs/protomotions/bin/python \
#       ~/miniconda3/envs/pyroki/bin/python \
#       /path/to/amass.pt g1 15 --clean
#
# Arguments:
#   proto_python:  Path to Python interpreter with ProtoMotions installed
#   pyroki_python: Path to Python interpreter with PyRoki installed
#   amass_pt_file: Path to packaged AMASS MotionLib .pt file (outputs saved in same directory)
#   robot_type:    Target robot: 'g1', 'h1_2', or 'astro_p2'
#   skip_freq:     (Optional) Skip every N motions for subset processing (default: 1 = all motions)
#   --source-skeleton: Packaged source skeleton (default: smpl)
#   --clean:       (Optional) Remove all intermediate outputs before running, ensuring a fresh pipeline

set -e  # Exit on error

# Parse arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <proto_python> <pyroki_python> <amass_pt_file> <robot_type> [skip_freq] [--source-skeleton smpl|smplx] [--retarget-device cpu|gpu] [--clean]"
    echo ""
    echo "Arguments:"
    echo "  proto_python   Path to Python interpreter with ProtoMotions installed"
    echo "  pyroki_python  Path to Python interpreter with PyRoki installed"
    echo "  amass_pt_file  Path to packaged AMASS MotionLib .pt file (outputs saved in same dir)"
    echo "  robot_type     Target robot: 'g1', 'h1_2', 'astro_p2', or 'p2'"
    echo "  skip_freq      (Optional) Skip every N motions (default: 1 = all motions)"
    echo "  --source-skeleton  Packaged source skeleton: smpl or smplx (default: smpl)"
    echo "  --retarget-device  PyRoki solve device: cpu or gpu (default: cpu)"
    echo "  --retarget-gpus    Comma-separated GPUs for parallel Astro P2 retargeting"
    echo "  --output-root      Directory for this input's intermediates and final MotionLib"
    echo "  --clean        (Optional) Remove all intermediate outputs before running"
    echo ""
    echo "Example:"
    echo "  $0 ~/miniconda3/envs/protomotions/bin/python ~/miniconda3/envs/pyroki/bin/python /data/amass.pt g1 15 --clean"
    exit 1
fi

PROTO_PYTHON="$1"
PYROKI_PYTHON="$2"
AMASS_PT_FILE="$3"
ROBOT_TYPE="$4"
shift 4
SKIP_FREQ="1"
CLEAN=""
SOURCE_SKELETON="smpl"
RETARGET_DEVICE="cpu"
RETARGET_GPUS=""
PIPELINE_OUTPUT_ROOT=""
SKIP_SET="false"
while [ $# -gt 0 ]; do
    case "$1" in
        --clean)
            CLEAN="--clean"
            shift
            ;;
        --source-skeleton)
            if [ $# -lt 2 ]; then
                echo "Error: --source-skeleton requires smpl or smplx"
                exit 1
            fi
            SOURCE_SKELETON="$2"
            shift 2
            ;;
        --retarget-device)
            if [ $# -lt 2 ]; then
                echo "Error: --retarget-device requires cpu or gpu"
                exit 1
            fi
            RETARGET_DEVICE="$2"
            shift 2
            ;;
        --retarget-gpus)
            if [ $# -lt 2 ]; then
                echo "Error: --retarget-gpus requires a comma-separated GPU list"
                exit 1
            fi
            RETARGET_GPUS="$2"
            shift 2
            ;;
        --output-root)
            if [ $# -lt 2 ]; then
                echo "Error: --output-root requires a directory"
                exit 1
            fi
            PIPELINE_OUTPUT_ROOT="$2"
            shift 2
            ;;
        *)
            if [ "$SKIP_SET" == "false" ] && [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
                SKIP_FREQ="$1"
                SKIP_SET="true"
                shift
            else
                echo "Error: unknown argument '$1'"
                exit 1
            fi
            ;;
    esac
done

if [ "$SOURCE_SKELETON" != "smpl" ] && [ "$SOURCE_SKELETON" != "smplx" ]; then
    echo "Error: --source-skeleton must be 'smpl' or 'smplx'"
    exit 1
fi

if [ "$RETARGET_DEVICE" != "cpu" ] && [ "$RETARGET_DEVICE" != "gpu" ]; then
    echo "Error: --retarget-device must be 'cpu' or 'gpu'"
    exit 1
fi

if [ -n "$RETARGET_GPUS" ] && [ "$ROBOT_TYPE" != "astro_p2" ] && [ "$ROBOT_TYPE" != "p2" ]; then
    echo "Error: --retarget-gpus is currently supported for astro_p2/p2"
    exit 1
fi

# Validate robot type
if [ "$ROBOT_TYPE" != "g1" ] && [ "$ROBOT_TYPE" != "h1_2" ] && [ "$ROBOT_TYPE" != "astro_p2" ] && [ "$ROBOT_TYPE" != "p2" ]; then
    echo "Error: robot_type must be 'g1', 'h1_2', 'astro_p2', or 'p2'"
    exit 1
fi

# Validate Python interpreters exist
if [ ! -f "$PROTO_PYTHON" ]; then
    echo "Error: ProtoMotions Python not found: $PROTO_PYTHON"
    exit 1
fi

if [ ! -f "$PYROKI_PYTHON" ]; then
    echo "Error: PyRoki Python not found: $PYROKI_PYTHON"
    exit 1
fi

# Prefer CUDA libraries installed inside the dedicated PyRoki environment.
# This server's system cuDNN may be older than the version required by JAX.
PYROKI_SITE_PACKAGES="$($PYROKI_PYTHON -c 'import site; print(site.getsitepackages()[0])')"
PYROKI_CUDA_LIBS=""
for CUDA_LIB_DIR in "$PYROKI_SITE_PACKAGES"/nvidia/*/lib; do
    if [ -d "$CUDA_LIB_DIR" ]; then
        PYROKI_CUDA_LIBS="${PYROKI_CUDA_LIBS:+$PYROKI_CUDA_LIBS:}$CUDA_LIB_DIR"
    fi
done
if [ -n "$PYROKI_CUDA_LIBS" ]; then
    export LD_LIBRARY_PATH="${PYROKI_CUDA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

run_pyroki() {
    if [ "$RETARGET_DEVICE" == "cpu" ]; then
        JAX_PLATFORMS="cpu" "$PYROKI_PYTHON" "$@"
    else
        "$PYROKI_PYTHON" "$@"
    fi
}

# Validate input file exists
if [ ! -f "$AMASS_PT_FILE" ]; then
    echo "Error: AMASS .pt file not found: $AMASS_PT_FILE"
    exit 1
fi

# Output directories are in the same location as input.
if [ -n "$PIPELINE_OUTPUT_ROOT" ]; then
    OUTPUT_DIR="$PIPELINE_OUTPUT_ROOT"
    mkdir -p "$OUTPUT_DIR"
else
    OUTPUT_DIR="$(dirname "$AMASS_PT_FILE")"
fi
KEYPOINTS_DIR="${OUTPUT_DIR}/keypoints-for-retarget"
RETARGETED_DIR="${OUTPUT_DIR}/pyroki-retargeted-${ROBOT_TYPE}"
CONTACTS_DIR="${OUTPUT_DIR}/contacts"
PROTO_DIR="${OUTPUT_DIR}/proto-${ROBOT_TYPE}"
FINAL_PT="${OUTPUT_DIR}/proto-${ROBOT_TYPE}.pt"

if [ "$CLEAN" == "--clean" ]; then
    echo "=============================================="
    echo "--clean: Removing all intermediate outputs..."
    echo "=============================================="
    for DIR_TO_CLEAN in "$KEYPOINTS_DIR" "$RETARGETED_DIR" "$CONTACTS_DIR" "$PROTO_DIR"; do
        if [ -d "$DIR_TO_CLEAN" ]; then
            echo "  rm -rf $DIR_TO_CLEAN"
            rm -rf "$DIR_TO_CLEAN"
        fi
    done
    if [ -f "$FINAL_PT" ]; then
        echo "  rm $FINAL_PT"
        rm "$FINAL_PT"
    fi
    echo ""
fi

echo "=============================================="
echo "Retargeting AMASS to ${ROBOT_TYPE^^}"
echo "=============================================="
echo "ProtoMotions Python: $PROTO_PYTHON"
echo "PyRoki Python:       $PYROKI_PYTHON"
echo "Input:               $AMASS_PT_FILE"
echo "Output dir:          $OUTPUT_DIR"
echo "Skip freq:           $SKIP_FREQ (1 = all motions)"
echo "Source skeleton:     $SOURCE_SKELETON"
echo "Retarget device:     $RETARGET_DEVICE"
if [ -n "$RETARGET_GPUS" ]; then
    echo "Parallel GPUs:       $RETARGET_GPUS"
fi
echo "=============================================="

# Step 1: Extract keypoints from packaged MotionLib (uses ProtoMotions)
echo ""
echo "[Step 1/5] Extracting keypoints from ${SOURCE_SKELETON^^} motions..."
$PROTO_PYTHON data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py \
    "$AMASS_PT_FILE" \
    --output-path "$KEYPOINTS_DIR" \
    --skeleton-format "$SOURCE_SKELETON" \
    --start-idx 0 \
    --skip-freq "$SKIP_FREQ"

# Step 2: Run PyRoki retargeting (uses PyRoki)
echo ""
echo "[Step 2/5] Running PyRoki retargeting to ${ROBOT_TYPE^^}..."
if [ "$ROBOT_TYPE" == "g1" ]; then
    run_pyroki pyroki/batch_retarget_to_g1_from_keypoints.py \
        --subsample-factor 1 \
        --keypoints-folder-path "$KEYPOINTS_DIR" \
        --source-type smpl \
        --output-dir "$RETARGETED_DIR" \
        --no-visualize \
        --skip-existing
elif [ "$ROBOT_TYPE" == "h1_2" ]; then
    run_pyroki pyroki/batch_retarget_to_h1_2_from_keypoints.py \
        --subsample-factor 1 \
        --keypoints-folder-path "$KEYPOINTS_DIR" \
        --source-type smpl \
        --output-dir "$RETARGETED_DIR" \
        --no-visualize \
        --skip-existing
else
    if [ -n "$RETARGET_GPUS" ]; then
        ASTRO_P2_GPUS="$RETARGET_GPUS" scripts/run_astro_p2_retarget_multi_gpu.sh \
            "$PYROKI_PYTHON" "$KEYPOINTS_DIR" "$RETARGETED_DIR" \
            --subsample-factor 1 \
            --source-type smpl
    else
        run_pyroki pyroki/batch_retarget_to_astro_p2_from_keypoints.py \
            --subsample-factor 1 \
            --keypoints-folder-path "$KEYPOINTS_DIR" \
            --source-type smpl \
            --output-dir "$RETARGETED_DIR" \
            --no-visualize \
            --skip-existing
    fi
fi

# Step 3: Extract contact labels from source motions (uses PyRoki)
echo ""
echo "[Step 3/5] Extracting foot contact labels from source SMPL motions..."
if [ "$ROBOT_TYPE" == "g1" ]; then
    run_pyroki pyroki/batch_retarget_to_g1_from_keypoints.py \
        --subsample-factor 1 \
        --keypoints-folder-path "$KEYPOINTS_DIR" \
        --source-type smpl \
        --save-contacts-only \
        --contacts-dir "$CONTACTS_DIR" \
        --skip-existing
elif [ "$ROBOT_TYPE" == "h1_2" ]; then
    run_pyroki pyroki/batch_retarget_to_h1_2_from_keypoints.py \
        --subsample-factor 1 \
        --keypoints-folder-path "$KEYPOINTS_DIR" \
        --source-type smpl \
        --save-contacts-only \
        --contacts-dir "$CONTACTS_DIR" \
        --skip-existing
else
    run_pyroki pyroki/batch_retarget_to_astro_p2_from_keypoints.py \
        --subsample-factor 1 \
        --keypoints-folder-path "$KEYPOINTS_DIR" \
        --source-type smpl \
        --save-contacts-only \
        --contacts-dir "$CONTACTS_DIR" \
        --skip-existing
fi

# Step 4: Convert to ProtoMotions format with contact labels (uses ProtoMotions)
echo ""
echo "[Step 4/5] Converting to ProtoMotions format..."
$PROTO_PYTHON data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
    --retargeted-motion-dir "$RETARGETED_DIR" \
    --output-dir "$PROTO_DIR" \
    --robot-type "$ROBOT_TYPE" \
    --contact-labels-dir "$CONTACTS_DIR" \
    --apply-motion-filter \
    --force-remake

# Step 5: Package into MotionLib (uses ProtoMotions)
echo ""
echo "[Step 5/5] Packaging into MotionLib..."
$PROTO_PYTHON protomotions/components/motion_lib.py \
    --motion-path "$PROTO_DIR" \
    --output-file "$FINAL_PT"

echo ""
echo "=============================================="
echo "Retargeting complete!"
echo "=============================================="
echo "Output MotionLib: $FINAL_PT"
echo ""
echo "To verify the result:"
echo "  python examples/motion_libs_visualizer.py --motion_files $FINAL_PT --robot $ROBOT_TYPE --simulator isaacgym"
echo ""
