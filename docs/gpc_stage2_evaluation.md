# GPC stage-two evaluation

New `examples/experiments/gpc/prior.py` runs use `GPCPriorEvaluator` every 200
iterations. Both TensorBoard and an enabled W&B logger receive `eval_train/*`
and, when requested, `eval_validation/*`. The evaluator is bounded to 32 initial
motions per rank, 64 expert steps (labels every 4 steps), and 1000 autonomous
steps by default. At 50 Hz the autonomous horizon is 20 seconds.

## Existing checkpoints / standalone evaluation

```bash
python protomotions/inference_agent.py \
  --checkpoint results/gpc_prior_amass_8gpu/inference_last.ckpt \
  --simulator isaaclab --headless --full-eval --num-envs 32 \
  --motion-file /data/chenguanting/datasets/AMASS/p2_motionlib/shards_8/amass_smpl_train_astro_p2_0.pt \
  --gpc-prior-eval \
  --prior-validation-motion-file /data/chenguanting/datasets/AMASS/p2_motionlib/validation/proto-astro_p2.pt \
  --prior-eval-output output/gpc_stage2_eval \
  --overrides 'robot.contact_bodies=["left_ankle_roll_link","right_ankle_roll_link"]'
```

Explicit `--gpc-prior-eval` and `--prior-*` evaluation options are also supported
by `train_agent.py` after loading frozen resume configs. They upgrade old empty
evaluators without rewriting old result/config files. A resume must have a
remaining training budget to reach evaluation; standalone evaluation is suitable
for completed runs. Add the same flags to a new training command to enable
validation. No training is launched by the evaluator itself.

Use `--prior-eval-num-envs`, `--prior-eval-steps`,
`--prior-eval-teacher-steps`, and `--prior-eval-seed` to bound cost. Fewer than
250/500/1000 steps at 50 Hz omit the corresponding 5/10/20-second survival
metrics; they are never silently extrapolated. Reports include zero-based
motion IDs, concrete source files, sampling temperature/top-p, thresholds,
per-motion survival times, and scalar metrics. Each rank writes its own JSON, including the checkpoint path for standalone
inference. Standalone inference does not restore training counters, so its
filename uses `epoch_0`; this does not mean the loaded model is untrained.

## Metric interpretation

* `prior/nll`: unsmoothed teacher-forced categorical NLL; dropout and running
  observation-statistic updates are disabled. Token/top-5/whole-sequence and
  per-token accuracy are separate. `mean_motion_perplexity` averages exp of
  each motion's mean NLL, so distributed item-weighted aggregation is valid.
* `prior/tf_argmax_action_mse`: decoder action error for teacher-forced argmax
  tokens versus the frozen expert. This is not autonomous action error.
* `rollout/upright_survival_{5,10,20}s` and `_horizon`: fraction of initial
  states remaining upright. Failure is anchor height <0.30 m OR anchor local
  up-Z <0.25 for 5 consecutive steps. Initially non-upright states fail at t=0;
  `initial_upright_fraction` exposes this. These are configurable upright-motion
  diagnostics, not valid success criteria for arbitrary lying/rolling motions.
* `mean_capped_survival_seconds` is capped at the requested horizon.
* `action_delta_rms` averages per-motion RMS change in raw policy action per
  control step while alive. It is not an actuator-torque or physical jerk metric.
* `mean_contact_body_xy_speed_mps` measures horizontal speed of configured
  contact-body origins while the simulator reports contact. It is a foot-slip
  proxy when those bodies are feet, not exact contact-point sliding velocity.
  No-contact motions contribute zero; `contact_observed_fraction` exposes them.
  When sensors are not configured, `contact_sensing_available=0` and the speed
  metric is omitted. The P2 command above explicitly enables both foot sensors;
  its checkpoint has `observe_contacts=False`, so observation dimensions stay
  unchanged. Do not change the contact-body list for contact-observing checkpoints.

Autonomous evaluation explicitly samples the prior with its saved temperature
and top-p. It never teacher-forces and never resets for reference clip completion
or reference-tracking termination. It does not score reference tracking, update
motion weights, or export a misleading tracking-success percentage. Best-model
selection uses validation upright survival at the horizon when validation is
configured, otherwise training upright survival. Assess NLL and behavior diversity
alongside that score: standing still can score well on survival alone.

The cohort is deterministic for the seed. Shared validation motions are
partitioned by global rank, without duplication; separate training shards use
local motion IDs. Sampling/physics state, original motion manager/library, reset
noise, pushes, model train/eval mode, and Python/NumPy/Torch RNG state are restored
in a finally block even if evaluation fails. The existing generic post-evaluation
policy-update skip remains unchanged.
