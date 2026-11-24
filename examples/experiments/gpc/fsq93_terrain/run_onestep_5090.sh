#!/usr/bin/env bash
# Run inside physics-drift-supermimic93-5090.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
onestep=/root/code/physics-drift-gpc-stage3-onestep-20261109
runtime=/root/code/physics-drift-supermimic93
export PYTHONPATH="$(dirname "$here"):$onestep/src:$runtime/deployment-runtime/site-packages"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BARRIER_NUMA_PIN_5090=1
export NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
exec /usr/bin/python3.12 -m torch.distributed.run --standalone --nproc-per-node="${NPROC:-8}" \
  --module fsq93_terrain.onestep --config "$here/configs/barrier_onestep.yaml" "$@"
