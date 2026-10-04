#!/usr/bin/env bash
set -euo pipefail

cd /data/chenguanting/ProtoMotions
source /data/chenguanting/.venvs/protomotions-isaaclab/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export WANDB_MODE=online

# foot_height_rew: total weight 0.15, height_std 0.05 m.
# contact_match_rew: weight -0.1, as requested.
exec python -u protomotions/train_agent.py \
  --robot-name p2 \
  --simulator isaaclab \
  --experiment-path examples/experiments/mimic/fsq.py \
  --experiment-name p2_fsq_bones_8gpu_lift_foot_reward \
  --motion-file /data/chenguanting/datasets/BONES-SEED/p2_motionlib_lze_contact_orin/bones_seed_train_astro_p2_slurmrank.pt \
  --ngpu 8 \
  --nodes 1 \
  --num-envs 4096 \
  --batch-size 16384 \
  --headless \
  --use-wandb \
  --wandb-project p2-gpc \
  --wandb-video \
  --wandb-video-every 200 \
  --eval-num-motions 512
