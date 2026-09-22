#!/bin/bash
# Run the Astro P2 PyRoki retargeter. CPU is the reliable default on this host.
set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <pyroki_python> <retargeter arguments...>"
    exit 1
fi

PYROKI_PYTHON="$1"
shift
if [ ! -f "$PYROKI_PYTHON" ]; then
    echo "Error: PyRoki Python not found: $PYROKI_PYTHON"
    exit 1
fi

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

RETARGET_DEVICE="${ASTRO_P2_RETARGET_DEVICE:-cpu}"
if [ "$RETARGET_DEVICE" != "cpu" ] && [ "$RETARGET_DEVICE" != "gpu" ]; then
    echo "Error: ASTRO_P2_RETARGET_DEVICE must be 'cpu' or 'gpu'"
    exit 1
fi
if [ "$RETARGET_DEVICE" == "cpu" ]; then
    export JAX_PLATFORMS="cpu"
fi
echo "Astro P2 retarget device: $RETARGET_DEVICE"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
exec "$PYROKI_PYTHON" \
    "$REPO_ROOT/pyroki/batch_retarget_to_astro_p2_from_keypoints.py" "$@"
