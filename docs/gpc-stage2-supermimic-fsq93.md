# GPC Stage2 with SuperMimic P2 FSQ93

This branch adapts the original GPC autoregressive Stage2 prior to the frozen
SuperMimic FSQ tracker. Stage1 and SuperMimic Physics are unchanged.

- Physics observation: `supermimic_obs_hist` = [B, 10*93], oldest first, native features.
- Motion target: `supermimic_goal` = [B, 730], used only for expert FSQ token labels.
- Frozen encoder: 730D -> 40 FSQ scalars, 9 levels each (-4..4).
- Frozen decoder: 40 FSQ scalars + 930D history -> 29D native P2 actions.
- Learned prior input: **only** the 930D history; never the future-motion goal.
- Default 8 AR tokens, each packing 5 FSQ scalars (vocabulary 59049).

Original docker150 tracker checkpoint:
`/root/code/SuperMimic-vibed-parity/logs/p2_fsq_vibed_parity_no_ref6d_return_rankshard32_8192x4_20261107/ckpt_0014000.pt`.
Only load a trusted checkpoint; the loader requires pickle metadata.

On server118:
```bash
source /root/code/activate-protomotions.sh
cd /root/code/ProtoMotions
python scripts/verify_supermimic_fsq93.py --checkpoint checkpoints/ckpt_0014000.pt
python scripts/train_gpc_stage2_supermimic_fsq93.py \
  --tracker-checkpoint checkpoints/ckpt_0014000.pt \
  --data-dir /path/to/real/supermimic/rollouts \
  --output-dir runs/gpc-stage2-fsq93 --device cuda:0
```

Training samples must be taken from actual SuperMimic simulator trajectories.
Each .npz shard must have `obs_hist` float32[N,930] and `goal` float32[N,730]
from the same timestep. Keep feature order and history order identical to
SuperMimic; do not substitute ProtoMotions observations.

The offline Stage2 training loop is implemented, but collecting those physics
rollouts and verifying online closed-loop execution are still separate tasks.