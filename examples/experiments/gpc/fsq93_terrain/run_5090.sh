#!/usr/bin/env bash
# Execute inside the physics-drift-supermimic93-5090 container.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROTO="/root/code/ProtoMotions-GPC-FSQ93-stage2"
RUNTIME="/root/code/physics-drift-supermimic93"
export PYTHONPATH="$(dirname "$HERE"):$PROTO/vendor:$RUNTIME/deployment-runtime/site-packages:$PROTO:$RUNTIME/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export GIT_PYTHON_REFRESH=quiet
exec /usr/bin/python3.12 -m "fsq93_terrain.$1" "${@:2}"
