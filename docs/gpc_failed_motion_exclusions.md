# GPC training: screened tracking failures

P2 FSQ trackers (including reduced-coordinate variants), GPC priors, and GPC
SFT/RLFT training automatically exclude the 2,633 sources listed in
`protomotions/data/motion_exclusions/p2_gpc_tracking_failed_20261020.json`.
This is the `tracking_failed=True` cohort from the 150,378-motion Stage1
tracker screening with `inference_epoch_14000.ckpt`. Tail-jitter-only candidates
are not excluded. Of the entries, 2,403 are in train and 230 in validation/test;
each loaded library excludes only its matching entries.

No additional launch argument is needed for P2 GPC/FSQ training. The policy is
applied after loading both new and frozen resume configurations. It takes
effect when starting/restarting a process; already running jobs are unchanged.
To deliberately use the original cohort, pass `--no-gpc-exclude-failed-motions`.
To opt in for another experiment, pass `--gpc-exclude-failed-motions`.
A custom JSON manifest can be selected with
`--overrides env.motion_manager.exclude_motion_sources_file=/path/to/manifest.json`.

Matching uses the dataset-relative source path retained in packed
`MotionLib.motion_files`, not shard-local numeric IDs or basenames. Dataset
prefixes are `BONES-SEED/p2_retarget/bvh/` and
`AMASS/p2_umr_full_20261001/motions/`; host/container prefixes may differ.
Mirror suffixes, subdirectories, and the versioned AMASS export namespace are
preserved. Repacking/reordering or switching shards recomputes local exclusions
without deleting data or renumbering existing motions.

Sampling weights are zeroed at manager creation, after curriculum updates,
after restored checkpoint weights, and before random sampling. Explicit/fixed
training resets selecting blacklisted sources fail with a clear error.
Each manager logs the matching count for its loaded library. A missing or
invalid manifest stops startup. An unrelated dataset can legitimately match
zero entries; its motions are not silently discarded.

Inference configs saved by training clear this bundled training-only filter,
so failed motions remain available for review. Full evaluators may still assess
failed references; their evaluation trajectories are not optimizer training
data. Non-P2 and non-GPC training keep their existing defaults.
