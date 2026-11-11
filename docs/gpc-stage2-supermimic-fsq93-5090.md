# Eight-GPU SuperMimic FSQ93 Stage2 on 172.16.9.22

This deployment uses the existing GPC Stage2 discrete autoregressive transformer
and the frozen SuperMimic P2 tracker. Nothing is changed in GPC Stage1.

- Git branch: `feat/gpc-stage2-supermimic-fsq93-20261110`
- Server repo: `/data/atom5/code/ProtoMotions-GPC-FSQ93-stage2`
- Physics runtime: `physics-drift-supermimic93-5090` (MJLab, 8 RTX 5090)
- Original checkpoint: `checkpoints/ckpt_0014000.pt`
- Original SHA256: `d2b1c6b5b51d00af7d716156834f90d54c08e3406f94741304b0e82c4f8782ba`
- Reference motions: `/data/atom5/code/datasets/mix5clean_v2_supermimic_p2_50hz/train/mix5clean_v2_train_00.pt`
- Real Physics rollout shards: `/data/atom5/code/datasets/gpc_fsq93_real_rollouts_20261111/` (32 shards, 262144 samples)
- Training outputs: `output/stage2_fsq93_8gpu_20261111/`

## Data and architecture

`scripts/collect_supermimic_fsq93_rollouts.py` runs the **original frozen
SuperMimic tracker in MJLab Physics**, capturing native `[B,930]` observation
history and `[B,730]` motion goals into completed atomic `.npz` shards.
Only the observations condition the autoregressive prior. Goals are used to
produce teacher-forced targets through the frozen encoder and never enter the
prior context. The fixed decoder converts 40 nine-level FSQ scalar codes to
29-dimensional native P2 actions.

The source code used for Physics collection is a separate copy of the exact
docker150 SuperMimic `src/` and MJCF assets:
`/data/atom5/code/SuperMimic-vibed-stage2/`.

The DDP runner `scripts/train_gpc_stage2_supermimic_fsq93_ddp.py` shards the
rollout dataset by GPU rank, wraps **only the prior** in DDP, and writes
atomic optimizer/model checkpoints from rank 0. It uses 8 tokens with
`9**5=59049` categories per token and per-GPU batch 64 (global 512).

## Launch and monitoring

Inside the physics-drift-supermimic93-5090 container:

```bash
bash /root/code/ProtoMotions-GPC-FSQ93-stage2/scripts/launch_gpc_stage2_fsq93_8gpu_5090.sh
```

The official launch wrapper selects all GPUs 0–7 and uses
`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1`. This is
required for the current CDI container where CUDA IPC shared-memory
import failed with `invalid device ordinal`. The eight-rank DDP smoke
test passed with this setting.

Monitor from the server host:

```bash
tail -f /data/atom5/code/ProtoMotions-GPC-FSQ93-stage2/output/stage2_fsq93_8gpu_20261111/train.log
nvidia-smi
```

`stage2_last.pt` is saved to the output directory every 100 steps.
For restart, set `RESUME_STAGE2=/root/code/.../output/stage2_fsq93_8gpu_20261111/stage2_last.pt`
before invoking the launcher inside the container. Existing checkpoints must
be backed up if running a different experiment in the same output directory.

The loss/token accuracy metrics evaluate next-token supervision, not yet
closed-loop control. A separate prior-in-the-loop benchmark is required.