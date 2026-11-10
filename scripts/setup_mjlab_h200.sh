#!/usr/bin/env bash
# Run after installing python3.12-dev, build-essential, libegl1, libgl1,
# libgles2 and libosmesa6 (Ubuntu package names).
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MJLAB_ENV=${MJLAB_ENV:-/opt/venvs/protomotions-mjlab}
UV=${UV:-uv}
cd "$REPO_ROOT"
if [[ ! -x "$MJLAB_ENV/bin/python" ]]; then
  "$UV" venv --python 3.12 "$MJLAB_ENV"
fi
"$MJLAB_ENV/bin/python" - <<'PY'
from pathlib import Path
import sys
import sysconfig
if sys.version_info[:2] != (3, 12):
    raise SystemExit('This lock was validated with Python 3.12')
if not (Path(sysconfig.get_path('include')) / 'Python.h').exists():
    raise SystemExit('Install python3.12-dev before Torch/Triton compilation')
PY
"$UV" pip install --python "$MJLAB_ENV/bin/python" -e . \
  -r requirements_mjlab.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match
"$UV" pip check --python "$MJLAB_ENV/bin/python"
MUJOCO_GL=disable "$MJLAB_ENV/bin/python" - <<'PY'
import torch
import mjlab
import mujoco
import warp
assert torch.cuda.is_available(), 'CUDA unavailable'
print('mjlab environment ready:', torch.__version__, mujoco.__version__, warp.__version__)
print('GPU:', torch.cuda.get_device_name(0), 'visible count:', torch.cuda.device_count())
PY
