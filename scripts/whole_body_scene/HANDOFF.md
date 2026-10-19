# Terrain reconstruction handoff

This is the entry point for resuming the BONES-SEED / Astro P2 scene work after
returning to `main`. The implementation, exclusion policy, tests and validation
records are committed on `codex/bones-p2-whole-body-scenes`.

## Checkouts and history

Machine: `atom8@172.16.9.7` (8 x 4090, machine 2).

| Purpose | Checkout | Branch |
| --- | --- | --- |
| Terrain reconstruction | `/data/chenguanting/ProtoMotions-wbc-scene` | `codex/bones-p2-whole-body-scenes` |
| Main development | `/data/chenguanting/ProtoMotions` | `main` |

These are linked Git worktrees. To return to main development, change directory
to the existing main checkout; do not force a second checkout of `main` or remove
the terrain worktree. Its ignored asset snapshot is needed for existing results.
Unrelated steering edits in the main checkout are outside this handoff.

The terrain branch starts at `f5fa8d89772050ec45e44d1939c678fd4681d3ed`.
Main subsequently received P2 parameter changes in `0d7cdd7`; terrain has not been
rebased onto those changes or merged into main. The feature branch's exclusion
and contact-aware training behavior do not automatically apply to main.

| Commit | Preserved work |
| --- | --- |
| `172e514` | Whole-body scene pipeline and contact-aware Stage 1 interface |
| `73c29d4` | Shared motion exclusion policy for reconstruction and training/evaluation |
| `b5e7fd7` | Evidence-backed stair profiles and robot/scene rendering |
| `ba3177d` | 0.08-second contact dwell and eight-motion validation |
| `1256973` | Independent sample consensus and continuous supported intervals |
| `087d8d3` | V2 temporal patches, deterministic surfaces and articulated support selection |
| `132ae28` | Final V2 development-cohort results and Isaac Lab Stage 1 smoke |

The handoff commit adds documentation only; the validated algorithm remains the
one recorded by `V2_VALIDATION.json`.

## Code entry points

- [V2_DESIGN.md](V2_DESIGN.md): current algorithm, runnable commands and limits.
- [README.md](README.md): full pipeline, legacy experiments, packing and Stage 1.
- [configs/whole_body_v2.json](configs/whole_body_v2.json): current V2 profile.
- `run.py`, `run_shards.py`: per-file and sharded reconstruction.
- `render_scene.py`: kinematic replay of motion with retained scene geometry.
- `pack.py`: paired MotionLib, SceneLib and contact bundle; review records are
  excluded by default.
- `examples/experiments/mimic/fsq_whole_body_scene.py`: Stage 1 experiment entry
  point (path relative to the repository root).
- `protomotions/data/motion_exclusions.json` and `scripts/motion_exclusions.py`:
  persistent key-based exclusions. Validation local IDs 4522/4524 and their
  mirrors 4523/4525 are excluded; numeric IDs are source references, not global
  motion identities.

## Runtime and data to preserve

```bash
cd /data/chenguanting/ProtoMotions-wbc-scene
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
export PROTOMOTIONS_ASSET_ROOT=/data/chenguanting/ProtoMotions-wbc-scene/results/wbc_assets
PYTHON=/data/chenguanting/.venvs/protomotions-isaaclab/bin/python
```

Source MotionLib directory:
`/data/chenguanting/datasets/BONES-SEED/p2_motionlib_lze_contact_orin`.
The final V2 result directory is
`/data/chenguanting/datasets/BONES-SEED/whole_body_scenes/v2_20261018_final`.
It contains per-motion scenes/contacts/quality records, videos, comparison
results, interface validation, Stage 1 smoke outputs and `SHA256SUMS`.

The frozen collision model is
`results/wbc_assets/astro_p2/mjcf/astro_p2_protomotions.xml`, SHA256
`8fd5b8b969a52852519f998bd91699de7cd4b2b43f9d8c9b6b069e60bcb7d906`.
Do not substitute the newer main asset for an existing bound scene bundle.
MotionLibs, result media, checkpoints and the runtime asset tree remain external
or ignored artifacts; a Git checkout alone does not reproduce those files.

## Validation boundary

[V2_VALIDATION.json](V2_VALIDATION.json) records the completed checks, rather than
a new test run during Git housekeeping:

- 33 development clips / 4,677 recorded frames checked; all 24 upstairs clips
  plus nine other motions. Five pass offline eligibility; 28 need review.
- 194 tests passed; one unrelated missing-LFS-asset test was deselected.
- Real MotionLib/SceneLib/contact binding and default review-bundle rejection.
- Isaac Lab integration smoke: two environments, eight physics steps, FSQ
  backward/one optimizer step and checkpoint roundtrip with an untrained policy.

No new GPC training or tracking-quality evaluation was performed. Ladder, wall
and obstacle reconstruction remain unresolved in some examples, and four
auxiliary wall hypotheses remain in upstairs scenes. Keep the review gates and
rejection reasons when expanding the corpus.

## Return to main / resume terrain

```bash
# Main is already checked out here; existing local edits remain intact.
cd /data/chenguanting/ProtoMotions
git branch --show-current

# Resume the preserved terrain branch later.
cd /data/chenguanting/ProtoMotions-wbc-scene
git branch --show-current
```

Reconstruction output directories are immutable with respect to their source,
asset, configuration and algorithm hashes. Use a new output directory when any
of those inputs changes. Do not promote inspection bundles into training.
