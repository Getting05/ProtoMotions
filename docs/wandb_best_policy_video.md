# Best-policy videos in W&B (IsaacLab)

Append these options to the normal `protomotions/train_agent.py` command:

```bash
--use-wandb --wandb-video --wandb-video-every 200
```

The feature is disabled by default. It runs only after a new launch or resume; it
does not modify a running training process. `train_slurm.py` forwards the same options.

For example, append the following when resuming the existing experiment with its
usual required arguments and `--experiment-name g1_fsq_amass_8gpu`:

```bash
--use-wandb --wandb-video --wandb-video-every 500 \
--wandb-video-duration 10 --wandb-video-motion-id 0
```

Explicit video arguments override the experiment's saved arguments on resume and
are persisted for subsequent resumes. Omitted options retain their saved values;
old experiments use the defaults below. Use `--no-wandb-video` to disable recording.

| Option | Default | Meaning |
|---|---:|---|
| `--wandb-video-every` | 200 | Completed epochs between recording attempts |
| `--wandb-video-duration` | 10 | Maximum simulated seconds |
| `--wandb-video-fps` | 30 | Encoded frames per second |
| `--wandb-video-width` | 1280 | Even output width |
| `--wandb-video-height` | 720 | Even output height |
| `--wandb-video-motion-id` | 0 | Motion ID within rank 0's resolved motion library |
| `--wandb-video-gpu` | rank 0 GPU | Logical index in the training process's CUDA_VISIBLE_DEVICES |
| `--wandb-video-timeout` | 300 | Wall-clock seconds allowed for the recording job |

At each interval global rank 0 copies `score_based.ckpt` and inference configs,
then launches one isolated, headless process with one robot. It uses the saved
best model even if the best score has not changed. There is no fallback to the
latest model before a best checkpoint exists. The evaluation frequency and best
score selection remain unchanged. With motion-shard switching, the motion ID
refers to rank 0's active shard at the trigger epoch; the exact path is recorded.

The policy uses deterministic actions where the model exposes `mean_action`,
starts the selected motion at time zero, and stops at motion end, episode end,
or the configured duration. A fixed-offset camera follows the policy robot's root. The same view also shows
a full-body blue reference robot at 40% opacity, aligned with the policy robot
in the same world coordinates. Reference poses use the motion manager's current
time and the same spawn/terrain correction as the mimic target. The reference
contains visual geometry only: it has no articulation, collision, or dynamics
and cannot affect the policy rollout. Transparency is composited from two views
of the same simulation instant, so it does not depend on RTX material-opacity
support. This adds rendering work in the isolated recorder.
Playback speed follows simulation time independently of output FPS.

Videos appear in the existing W&B run under `videos/best_policy`. Captions and
`videos/trigger_epoch`, `videos/best_epoch`, `videos/best_score`, and
`videos/motion_id` identify the source. Because rendering is asynchronous, the
upload uses the current `trainer/global_step`, not an old W&B history step.
Offline W&B runs retain normal offline synchronization behavior.

Outputs are stored under the experiment's
`videos/epoch_XXXXXXXX/{best_policy.mp4,best_policy.json,render.log}`. A unique
suffix preserves previous videos if a resumed run repeats an epoch. Completed
videos remain available locally. Temporary checkpoint copies are deleted after
success, failure, timeout, or ordinary training shutdown. Only one job runs at a
time; busy intervals are skipped. Rendering or upload failures produce warnings
without failing training. Pending rendering is terminated when training exits.

On the current IsaacLab 3 server, the recorder uses the headless RTX render
helpers. Packed motion libraries without linked scenes are sliced on CPU so only
the selected motion is loaded onto the GPU, and collision buffers are reduced
for one robot. Scene-linked libraries retain their original motion indices.

The child needs extra GPU memory and `imageio`/`imageio-ffmpeg` with an H.264
encoder in the same Python environment. Use an available GPU, or reduce resolution
and frequency if resource pressure causes failures. First shader compilation can
exceed the default timeout; adjust `--wandb-video-timeout` if necessary. No
additional simulator or W&B process is launched when the feature is disabled.

## Fixed plus random motion videos

Each interval now records the configured fixed motion (default 0), followed by
two different uniformly sampled motions from rank 0's active library. The random
motions exclude the fixed ID; sampling uses a separate system RNG and does not
alter training randomness. Libraries with fewer than three motions record all
available distinct motions and emit a warning. Across intervals, random IDs may
repeat. No extra CLI options are required.

All three videos share the same checkpoint/config snapshot and resolved motion
shard. They render sequentially, with at most one child process alive. The existing
300-second timeout applies separately to each video. A rendering/upload failure
skips that motion and continues with the remaining motions; a batch still running
at the next interval causes that new interval to be skipped. At shutdown no more
motions are launched.

The fixed video remains at `videos/best_policy`; extras use
`videos/best_policy_random_1` and `videos/best_policy_random_2` in the same W&B run,
with their own motion ID and checkpoint metadata. Extra local outputs are in
`videos/epoch_XXXXXXXX/random_N_motion_ID/`. The blue translucent reference overlay
is retained for every motion. Since this changes the training-side scheduler,
existing training processes must be resumed/restarted to use the three-video batch.
