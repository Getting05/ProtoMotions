# Upstream reliability integration

This branch merges upstream through `f69fdcaabff31846b4d2247207417b57dc788828`:

- `7a8417a`: training launch and simulator reliability.
- `675789b`: simulator correctness and training reliability.
- `f69fdca`: viewer keyboard documentation.

The fork baseline is `d978cc469f42107e16a706cdb504132c160b1628`, including the
current P2 source-motion exclusion policy and URDF control parameters. The
environment/configuration rewrite `0a650c5` and browser/native follow-up
`0841ed2` are not included.

## Compatibility choices

- Keep the P2/GPC recipes, robot values, 71-channel deployment observation,
  29 policy actions, and 200 Hz physics / 50 Hz policy contract.
- Keep `MimicEvaluatorConfig.eval_action_ema_alpha`, action EMA, inactive-env
  parking, scored-env scheduling, sampled evaluation, sparse metric rows,
  CPU trajectory export, and policy-action-width recording. Retain Newton's
  parking no-op. Add upstream model rollout-context snapshot/reset/restore.
- Keep BM scaling at `effort_limit / stiffness` and the existing BM L2C2
  processed-action smoothness reward. Retain the new positive-stiffness check.
- Keep the `.pt` resolved configuration and checkpoint-loading interfaces,
  parameter EMA, PEFT code, physics guard, and existing video recording flow.
  Forward video options through the new shared cluster launch argument builder.
- New latency and body-mass randomization remain disabled by default; no P2/GPC
  recipe opts into them here. Body masses in the new API are absolute kg.

## Maintenance-state fixes

Contact-cache flushing is marked as maintenance so the temporary underground
pose cannot enter P2 IMU samples. Evaluation snapshots save and restore the
complete P2 sensor bank, including finite-difference and history buffers.
Reset sampling is suppressed at the actual `BaseEnv.reset` entry point.
Regression tests exercise normal, guard and trace paths and post-restore IMU
continuity. These checks do not claim bitwise restoration of every simulator
internal or of an enabled latency randomizer.

Segmented contact smoothing uses local float64 prefix accumulation in bounded
frame blocks with a window halo, returning float32 on the original device.
Float32 global prefix sums lose unit contact counts above `2**24`, changing
labels on large corpora. Blocks contain at most approximately `2**20` contact
elements, so temporary prefix/index storage does not grow with corpus length.
For 31 bodies and window 7, one local float64 prefix is approximately 8 MiB;
other block temporaries add a constant multiple. The output still requires
`4 * num_frames * num_bodies` bytes. Regressions cover the large-input precision
boundary, motion/block boundaries, probabilities and long windows. GPU speed
and peak memory were not measured.

Old test doubles for video/CLI/guard interfaces and steering lifecycle have
been brought into line with the existing fork behavior. Their original
failures were reproduced against the fork baseline before updating fixtures;
production steering, PEFT and video code were retained.

## Validation

Run the CPU suite with the project's dependencies and P2 mesh assets available:

```bash
MUJOCO_GL=egl PYTHONPATH=. python -m pytest protomotions/tests/ -q -m 'not needs_lfs'
python scripts/check_model_licenses.py
```

P2 meshes are existing external test dependencies and are not added by this
merge. Hardware/simulator-dependent skips and Git LFS payload exclusions must
be reported alongside test counts. CPU and MuJoCo regressions are not an
IsaacLab GPU training or inference smoke test.

A separate actual P2 MuJoCo CPU smoke advances default-pose targets for 200
control steps / 4000 physics substeps using its existing 1000 Hz MuJoCo config
and 50 Hz control rate; all robot-state tensors remain finite. This is a backend
execution check, not evidence of learned-policy tracking or standing quality.
The deployment-aligned IsaacLab recipe retains its separate 200 Hz physics /
50 Hz policy contract.
