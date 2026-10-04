# SwanLab experiment logging

SwanLab is an optional alternative to W&B. TensorBoard remains enabled in every
mode, and existing `--use-wandb` commands retain their behavior. No SwanLab SDK
is imported or initialized unless that backend is enabled.

## Install and enable

In your existing ProtoMotions training environment, install the extra and log in:

```bash
pip install -e '.[swanlab]'
swanlab login
```

Append these options to your usual training command:

```bash
--logging-backend swanlab --swanlab-project physical_animation
```

For a fresh experiment, `--use-swanlab` also works. It is additive, so
`--use-wandb --use-swanlab` enables both. The explicit backend selector is more
useful when continuing an existing experiment:

| Selection | Logs |
|---|---|
| `--logging-backend swanlab` | TensorBoard + SwanLab |
| `--logging-backend wandb` | TensorBoard + W&B |
| `--logging-backend both` | TensorBoard + W&B + SwanLab |
| `--logging-backend tensorboard` | TensorBoard only |

`--swanlab-project` defaults to `physical_animation`. Use `--swanlab-workspace`
for an organization. `--swanlab-mode online`, `offline`, `local`, or `disabled`
overrides the SDK mode; if omitted, SDK/environment settings such as
`SWANLAB_MODE` apply. Authentication uses `swanlab login` or `SWANLAB_API_KEY`;
API keys are not CLI arguments or saved in the training configuration.

## Coverage

The integration covers the W&B features currently used by this repository:

| Existing feature | SwanLab behavior |
|---|---|
| Training scalars, losses, rewards, timing, FSQ diagnostics | Same metric keys and epoch steps through `fabric.log_dict` |
| Training/validation evaluation | Same evaluator namespaces, including `eval_train/*` and `eval_validation/*` |
| Resolved hyperparameters | Robot, simulator, terrain, scenes, motions, environment, agent, and Fabric configuration |
| Same-experiment resume | Independent `swanlab_id` saved in `results/<name>/config.yaml`; SDK `resume="allow"` |
| Distributed training | Existing metric aggregation retained; only global rank zero initializes/logs a run |
| Best-policy videos | Fixed motion plus two random motions, reference overlay, captions, source epoch/score/motion metadata |
| Offline operation | SDK logs in `results/<name>/swanlog/`; later sync to SwanLab |
| SLURM launcher | Forwards logging and video options; installs the optional SDK |

This is feature parity with this repository's W&B integration. The repository
does not currently upload model checkpoints as W&B artifacts, use W&B sweeps,
or watch model gradients; this change does not add those separate features.

## Switch an existing run from W&B

Resume with the same experiment name and usual required arguments, adding:

```bash
--logging-backend swanlab --swanlab-project physical_animation
```

The selector overrides saved logger settings. Model/optimizer/training counters
still come from the same `last.ckpt`; this is not a warm start. The SwanLab run
starts collecting at the resumed epoch. Existing W&B history is not migrated.
The old `wandb_id` is retained so switching back can continue the W&B run.

New logging settings and the actual SDK run ID are persisted before training.
The next resume reuses them without repeating flags. Explicit SwanLab project,
workspace, and mode options override saved values; changing project/workspace
clears the old SwanLab ID to create a run in the new destination. A new experiment
name plus `--checkpoint ...` remains a warm start and gets a new logging run.

Without new logging flags, legacy W&B resume precedence remains unchanged:
saved `use_wandb`/`wandb_project` win. Use `--logging-backend wandb` to explicitly
enable W&B on a run that previously logged only to TensorBoard.

## Best-policy videos (IsaacLab)

```bash
--logging-backend swanlab --swanlab-video \
--swanlab-video-every 200 --swanlab-video-duration 10 \
--swanlab-video-motion-id 0
```

Video settings have both `--wandb-video-*` and `--swanlab-video-*` spellings, with
the same defaults and one shared recording schedule. For both dashboards use
`--logging-backend both --wandb-video --swanlab-video`. When a backend selector
switches a resumed experiment with recording already enabled, recording follows
the selected backends; explicit video flags override this automatic selection.
`--no-swanlab-video` disables SwanLab media independently.

The existing renderer still produces the full-resolution MP4 and JSON metadata.
The child process additionally converts it to `best_policy.gif` for SwanLab,
limited to 640 pixels wide and 15 FPS with the same playback duration. This
conversion runs under the existing per-video timeout, outside training. W&B
continues to receive the original MP4. Each backend's upload failure is isolated.

SwanLab currently accepts GIF videos ([SDK documentation](https://docs.swanlab.cn/en/api/py-video.html)).
The same `score_based.ckpt`, evaluator, rank-zero, and IsaacLab prerequisites
described in [best-policy recording](wandb_best_policy_video.md) apply. Enabling
logging does not enable an evaluator or create a best checkpoint.

## Offline and SLURM

Use `--swanlab-mode offline` without logging in. Later upload one recorded run:

```bash
swanlab sync results/EXPERIMENT/swanlog/run-REPLACE_WITH_RUN_DIRECTORY
```

For multiple offline segments from resumed training, sync the first, then use
`swanlab sync <segment-directory> --id <cloud-experiment-id>` for subsequent
segments to keep a single cloud history. See the
[sync documentation](https://docs.swanlab.cn/en/api/cli-swanlab-sync.html).

`train_slurm.py` accepts the same logging/video flags. Configure `swanlab login`
or `SWANLAB_API_KEY` in the cluster/container environment before online jobs.
SwanLab-only jobs never check W&B credentials. The launcher does not copy local
SwanLab secrets into generated or printed batch scripts.

## Validation

The tests exercise old W&B paths, backend switching and ID persistence,
rank-zero behavior, failure isolation, SLURM forwarding, and a real offline
SwanLab SDK/Fabric run with scalar/config/GIF recording. Run them with the
`swanlab` and `dev` extras installed. Full IsaacLab policy rendering and online
cloud upload require the training environment and account credentials.

```bash
pytest -q protomotions/tests/test_experiment_logging.py \
  protomotions/tests/test_swanlab_logger.py \
  protomotions/tests/test_train_agent_helpers.py \
  protomotions/tests/test_wandb_video.py
```

Validated on 2026-10-04 with SwanLab 0.10.1, W&B 0.23.1, and Lightning 2.5.6:
107 related tests passed; one USD/reference-rendering test was skipped because
`pxr` was unavailable. A real two-process CPU Fabric run with W&B imports blocked
created a SwanLab run only on rank zero and logged the correct aggregated metric.
A real dual-backend offline smoke recorded scalars/configuration, W&B MP4, and
SwanLab GIF successfully. Online upload and full IsaacLab/GPU training were not
run for this change.
