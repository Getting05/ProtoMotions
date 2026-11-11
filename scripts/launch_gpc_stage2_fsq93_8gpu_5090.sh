#!/usr/bin/env bash
# Run INSIDE physics-drift-supermimic93-5090 on 172.16.9.22.
set -euo pipefail

ROOT=/root/code/ProtoMotions-GPC-FSQ93-stage2
export PYTHONPATH="$ROOT/vendor:/root/code/physics-drift-supermimic93/deployment-runtime/site-packages:$ROOT"
export GIT_PYTHON_REFRESH=quiet
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
# CDI runtime on this host cannot import NCCL CUDA IPC SHM buffers.
# Fallback to socket transport; all eight GPUs remain active.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
cd "$ROOT"

EXTRA=()
if [[ -n "${RESUME_STAGE2:-}" ]]; then
  EXTRA+=(--resume "$RESUME_STAGE2")
fi

exec /usr/bin/python3.12 -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/train_gpc_stage2_supermimic_fsq93_ddp.py \
  --tracker-checkpoint "$ROOT/checkpoints/ckpt_0014000.pt" \
  --data-dir /root/code/datasets/gpc_fsq93_real_rollouts_20261111 \
  --output-dir "$ROOT/output/stage2_fsq93_8gpu_20261111" \
  --epochs 100 --steps-per-epoch 200 --max-steps 20000 \
  --batch-size 64 --d-model 1024 --num-heads 4 \
  --num-layers 6 --ff-size 4096 --dropout 0.1 \
  --fsq-scalars-per-token 5 --lr 0.0001 \
  --grad-clip 1.0 --label-smoothing 0.01 \
  --log-every 10 --save-every 100 "${EXTRA[@]}"