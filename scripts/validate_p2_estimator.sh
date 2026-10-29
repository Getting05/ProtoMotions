#!/usr/bin/env bash
# One-machine convenience launcher. Override any default with the Python flags.
# Example: CUDA_VISIBLE_DEVICES=2 bash scripts/validate_p2_estimator.sh --motion-id 12
set -euo pipefail
p2_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$p2_repo"
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" && "${1:-}" != "--help" ]]; then
    echo 'Choose an idle GPU explicitly, e.g. CUDA_VISIBLE_DEVICES=2 bash scripts/validate_p2_estimator.sh' >&2
    exit 2
fi
p2_python=${P2_VIEWER_PYTHON:-"$(dirname "$p2_repo")/.venvs/protomotions-isaaclab/bin/python"}
if [[ ! -x "$p2_python" ]]; then p2_python=python; fi
export PYTHONPATH="$p2_repo${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec "$p2_python" scripts/validate_p2_estimator.py \
    --controller results/p2_fsq_bones_8gpu_lift_foot_smooth/inference_score_based.ckpt \
    --estimator results/p2_estimator_comparison_20261016/baseline/best.pt \
    --motion-file results/p2_estimator_comparison_20261016/reference_subset.pt \
    "$@"
