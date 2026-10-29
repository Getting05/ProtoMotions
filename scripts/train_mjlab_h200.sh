#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MJLAB_ENV=${MJLAB_ENV:-/opt/venvs/protomotions-mjlab}
export MUJOCO_GL=${MUJOCO_GL:-disable}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
exec "$MJLAB_ENV/bin/python" -m protomotions.train_agent --simulator mjlab --headless "$@"
