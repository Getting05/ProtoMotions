# mjlab training on H200

The `mjlab` simulator retains the ProtoMotions trainer, PPO/FSQ models, MotionLib,
rewards and Lightning Fabric DDP. mjlab supplies Scene/Entity/Simulation backed
by MuJoCo Warp. It does not introduce a second RL runner or require Isaac Sim.

## Installed environment

Validated on 2026-10-29 at `/data/workspace/ProtoMotions` on an 8 × H200 machine:

- Runtime: `/opt/venvs/protomotions-mjlab` (local storage, not the NFS workspace).
- Python 3.12.3, Torch 2.11.0+cu128, mjlab 1.6.0.
- MuJoCo / MuJoCo Warp 3.11.0, Warp 1.18.0, Lightning 2.5.6.
- Driver 580.95.05; no driver replacement was necessary.
- `requirements_mjlab.lock.txt` pins all 158 third-party runtime/test packages.
  This is a Linux x86_64 / Python 3.12 deployment snapshot. The repository is
  installed separately in editable mode.

Install Ubuntu prerequisites, then run the setup script with uv available:

```bash
apt-get install -y python3.12-dev build-essential libegl1 libgl1 libgles2 libosmesa6
UV=/data/workspace/mjlab-build/uv scripts/setup_mjlab_h200.sh
```

The script checks Python headers (needed by Torch/Triton), installs the locked
packages, runs `uv pip check`, and checks CUDA imports. Set `MJLAB_ENV` to change
the environment directory. On the configured H200, an offline rerun with
`UV_OFFLINE=1` was successful. If a shell inherits the unavailable
`127.0.0.1:1089` proxy, clear that proxy for the installation command or supply a
working proxy; the setup does not change global network configuration.

The earlier `/data/workspace/.venvs/protomotions-mjlab` directory is an incomplete
NFS installation and is **not** the training environment.

The P2 checkout was missing its STL files. The matching 37 meshes have been
installed under `protomotions/data/assets/astro_p2/meshes`; all sizes and SHA256
hashes match `docs/mjlab_p2_assets.json`. These externally supplied binaries are
not included in the backend commit. A new checkout needs the same asset bundle.
The matching source MJCF SHA256 is
`13fae34efca48c88a65da783ae2ef682697884d84e89a9d058132f982e642d64`.

For root Git commands on this checkout, use the scoped trust option if Git
reports a dubious-ownership error:

```bash
git -c safe.directory=/data/workspace/ProtoMotions status
```

The initial archive transfer changed some directory ownership to the archive's
UID 1000. Automated approval rejected a subsequent ownership-restoration command;
no further ownership change was attempted. A per-command Git trust option is
used instead of a persistent global exception. The H200 has no Git author
configured, so this backend commit uses the explicit automation identity
`Codex <codex@local.invalid>` without changing global Git settings.

## Launch

Run from the repository root. `num-envs` and `batch-size` are per rank. Fabric
launches the ranks; do not wrap this in torchrun or mjlab's own training launcher.
Use a new experiment name when changing simulator backends.

```bash
CUDA_VISIBLE_DEVICES=0,1 scripts/train_mjlab_h200.sh \
  --robot-name astro_p2 \
  --motion-file /path/to/your/p2.motion \
  --experiment-path examples/experiments/mimic/fsq_reduced_coord_obs.py \
  --experiment-name p2_mjlab_run \
  --num-envs 1024 --batch-size 4096 --ngpu 2
```

The environment/batch numbers above are starting settings, not a measured
throughput optimum. For eight GPUs use `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` and
`--ngpu 8`, after checking available memory. Only one- and two-GPU training have
been runtime validated; existing allocations prevented an eight-GPU acceptance
run. Do not terminate other jobs to free memory.

Repeat the original command to resume the same experiment. The existing trainer
loads its frozen `resolved_configs.pt`, model and optimizer states, resets
rollouts, and skips the first policy update. It also skips an update after
scheduled evaluation. CLI config overrides are ignored on same-name resume.
This is the repository's training-resume behavior, not bitwise continuation of
all physical state across process restarts.

Inference through the existing evaluation entry point:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=disable OMP_NUM_THREADS=1 \
 /opt/venvs/protomotions-mjlab/bin/python -m protomotions.inference_agent \
 --checkpoint results/p2_mjlab_run/inference_last.ckpt \
 --simulator mjlab --headless --num-envs 2 --full-eval \
 --overrides simulator.projectile.num_projectiles=0
```

The P2 experiment enables projectiles in its inference defaults; the explicit
zero override is required for this backend. Motion files must include the
reference contact labels used by the chosen experiment. The existing G1 example
and all-zero contact labels do not satisfy the contact-matching reward.

## Control and state contracts

- The base simulator converts native WXYZ quaternions to ProtoMotions' common
  representation. Link velocities use world coordinates at link origins.
- Body, joint and contact ordering is mapped by name. BUILT_IN_PD uses native
  position actuators; PROPORTIONAL and TORQUE use motors. Existing robot config
  determines stiffness, damping, effort limits, armature and joint friction.
- P2 runs at 200 Hz physics / 50 Hz control, including post-substep sensor updates
  and the existing 71D observation / 29D action contract. Physics guard recovery
  happens before P2 sensor sampling.
- Evaluation snapshots include MuJoCo state, warm-start arrays, action history,
  control targets and P2 sensor history. Normal training checkpoints retain the
  existing trainer's lifecycle described above.
- Recording uses ImageIO/FFmpeg instead of MoviePy. This removes the Pillow
  version conflict between MoviePy and mjlab's viewer dependency, retaining the
  H.264/yuv420p encoding settings. A two-frame encoded video was decoded again.

## Scope and limitations

Supported: headless articulated MJCF robots on flat terrain, GPU batches,
per-body net contact forces, subset reset, P2 reduced observations, velocity
pushes, action bias and CoM offsets, and native physics guard slot recovery.
Physical DR other than the listed mappings fails explicitly. PhysX separate
static/dynamic friction and restitution randomization is not silently translated
to MuJoCo. Complex terrain, SceneLib objects and projectiles are not implemented.
Offscreen frame capture is implemented but is not part of the validated path.
The existing automatic W&B video launcher remains IsaacLab-specific.

MuJoCo contact behavior and solver parameters differ from PhysX. No long-run
policy-quality or throughput equivalence is claimed. A PyTorch AccumulateGrad
stream warning remains during optimization; it did not prevent the checked
updates or DDP, but performance tuning remains separate work.

## Validation evidence

Logs live in `/data/workspace/mjlab-build/`:

| Check | Evidence |
| --- | --- |
| GPU contract: FK / world velocity vs CPU MuJoCo, subset reset isolation, 100 control steps, 400 callbacks, snapshot restore | `physics-smoke-guard.log` |
| Injected NaN recovered only environment 0 | `PHYSICS_GUARD_RECOVERED`, followed by `MJLAB_PHYSICS_SMOKE_PASS` |
| P2 reduced single-GPU optimization, two iterations, 256 transitions | `train-p2-v2.log`, `results/mjlab_p2_reduced_smoke_v2/last.ckpt` |
| Two GPU NCCL/Fabric DDP, two iterations, 256 global transitions | `train-ddp.log`, `results/mjlab_p2_ddp_smoke/last.ckpt` |
| Stop after epoch 1, native resume to epoch 4, another optimizer update | `resume-updates-initial.log`, `resume-updates-continued.log`, `results/mjlab_p2_resume_updates/last.ckpt` |
| Checkpoint tensors finite; 40 model tensors changed after resume, including encoder weights | `artifact-verification.log` |
| Native inference checkpoint full evaluation | `inference-p2.log` |
| Locked install rerun and 159-package compatibility check | `setup-verification.log` |
| 100 tests and 24 subtests: simulator and P2 regression checks | `final-regression.log` |

All training/evaluation checks use a clearly synthetic 300-frame standing
fixture at `/data/workspace/mjlab-build/p2_standing_test_fixture.motion`, generated
by the GPU smoke with synthetic foot-support labels. It is not a production
motion corpus. The short-trained policy scored **0% success** on this fixture;
validation establishes execution and checkpoint lifecycle, not balance or
tracking quality. The constant-PD physics test also falls while staying finite.

Reproduce the GPU contract/fixture generation:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=disable OMP_NUM_THREADS=1 \
 /opt/venvs/protomotions-mjlab/bin/python -m protomotions.tests.smoke_mjlab \
 --num-envs 16 --steps 100 \
 --motion-output /data/workspace/mjlab-build/p2_standing_test_fixture.motion
```
