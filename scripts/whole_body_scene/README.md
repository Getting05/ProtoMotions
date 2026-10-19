# BONES-SEED / Astro P2 whole-body scenes

Start with [HANDOFF.md](HANDOFF.md) for branch history, preserved artifacts and
returning to the main checkout. The current algorithm and commands are in
[V2_DESIGN.md](V2_DESIGN.md); [V2_VALIDATION.json](V2_VALIDATION.json) records its
validation. The sections below also retain the legacy experiments.

This implementation reconstructs **conservative static collision hypotheses**
from the existing retargeted P2 MotionLib. It preserves unresolved contacts and
dynamic-object hypotheses rather than treating them as flat-ground negatives.
It does not claim to recover a unique true scene from motion alone.

## Implemented path

1. Read a packaged MotionLib with memory mapping; validate its body poses against
   the actual P2 MJCF with forward kinematics. Sample every effective collision
   body, including geom-local position/orientation and exposed union surfaces.
   Visual meshes are excluded. The current P2 collision asset uses analytic
   primitives; unsupported collision mesh types fail explicitly.
2. Compute surface velocities from body translation and angular velocity.
   Detect low-speed/deceleration/dwell and sliding candidates with hysteresis.
   Apply whole-motion solid-volume exclusion before fitting any environment.
   Stationary non-ground surfaces without an observed normal approach/stop
   remain unknown; low speed alone cannot justify a support around an idle hand.
   Scores are engineering evidence scores, not calibrated probabilities.
3. Cluster using position, oriented normal and normal-direction offset. Fit
   finite 3D boxes in tangent PCA frames, split sparse/non-convex extents, expand
   only within sampled free space, and add checked low-confidence ground
   supports for elevated slabs. Final MuJoCo primitive-distance **and collision
   manifold** checks cover every recorded frame, including contained solids.
   Small overlaps may be reduced by retreating the surface along its normal,
   bounded by the 25 mm contact tolerance, then checking the entire motion again.
4. Fit bimanual rigid-frame trajectories and retain rigidity residuals. Without
   object-shape/grasp evidence, geometry, mass and friction remain unknown.
   Dynamic candidates appear in the shared graph but are not spawned as
   guessed physical boxes. Static and dynamic hypotheses remain distinguishable.
5. Save three-state per-link labels and ragged per-frame/body contact patches.
   A link can have several simultaneous scene contacts. Unknowns are excluded
   from negative supervision. A bounded inverse-dynamics least-squares QP scores
   unilateral friction-pyramid forces and P2 torque limits. This is a necessary
   feasibility diagnostic, not proof of controller success.
6. Pack only eligible motions into a derived MotionLib with a paired SceneLib,
   preserving original keys, source local IDs, hashes, timing, collision-asset
   version and coordinate offsets. Scene slots are padded with fixed 1 mm boxes
   below z=-100 m. Ground labels use the simulator's existing z=0 terrain;
   packing excludes coplanar ground boxes to avoid duplicate contact forces.
   Every training environment has a fixed matching motion ID.
7. `fsq_whole_body_scene.py` adds the future contact plan to the FSQ encoder and
   critic, plus unknown-aware full-body contact reward and a separate severe
   unexpected-contact penalty. The decoder still consumes proprioception and
   latent. Net contact forces are combined with expected patch proximity;
   this is geometric association, not an exact contact-pair sensor.

## Run on machine 2

The checkout is `/data/chenguanting/ProtoMotions-wbc-scene`. The source checkout
and ongoing training are independent. Use the existing runtime and asset tree:

```bash
cd /data/chenguanting/ProtoMotions-wbc-scene
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
export PROTOMOTIONS_ASSET_ROOT=/data/chenguanting/ProtoMotions-wbc-scene/results/wbc_assets
PYTHON=/data/chenguanting/.venvs/protomotions-isaaclab/bin/python
```

`results/wbc_assets` contains the reconstruction branch's MJCF and reuses the
existing visual meshes. Its MJCF SHA256 is
`8fd5b8b969a52852519f998bd91699de7cd4b2b43f9d8c9b6b069e60bcb7d906`.
The main checkout's model changed during this task; scene bindings correctly
reject it. Use the matching asset snapshot for this existing scene bundle.

Example representative validation cohort (no GPU required):

```bash
$PYTHON -m scripts.whole_body_scene.run \
  --motion-file /data/chenguanting/datasets/BONES-SEED/p2_motionlib_lze_contact_orin/bones_seed_validation_astro_p2.pt \
  --output /data/chenguanting/datasets/BONES-SEED/whole_body_scenes/validation_demo \
  --ids 2054,1408,4336,3588,1074,6146 --support-frames 0

$PYTHON -m scripts.whole_body_scene.pack \
  --manifest /data/chenguanting/datasets/BONES-SEED/whole_body_scenes/validation_demo/manifest.jsonl \
  --output /data/chenguanting/datasets/BONES-SEED/whole_body_scenes/validation_demo/packed/scenes.pt
```

`--support-frames 0` scores every frame; a positive number explicitly subsamples
only the force-feasibility diagnostic. Geometry always checks every stored frame.
Output is locked against concurrent writers. Completed records are reused only
when source, asset, configuration and algorithm hashes match, all referenced
artifacts exist, and contact/scene artifact hashes are intact. Different settings
require a new output directory. Failures are saved per motion and return nonzero.

For train shards (keep train, validation and test in separate output roots):

```bash
$PYTHON -m scripts.whole_body_scene.run_shards \
  --motion-pattern /data/chenguanting/datasets/BONES-SEED/p2_motionlib_lze_contact_orin/bones_seed_train_astro_p2_slurmrank.pt \
  --output /data/chenguanting/datasets/BONES-SEED/whole_body_scenes/train \
  --workers 1 --support-frames 0 --pack
```

This is a full-corpus command, not a claim that the entire corpus has already
been reconstructed. Pilot runtime and acceptance should be reviewed first.
Add `--limit-per-shard` for a bounded pilot. A shard with no eligible scenes fails
packing; it never silently substitutes flat ground or another data split.

## Stair reconstruction and joint visualization

Use `--static-scene-mode stairs` for an explicitly selected ascending-stair
cohort. The full-body detector still runs. The extra structural prior groups
normal-approach/stop evidence on the soles into support windows, estimates
horizontal support heights, and bounds each riser against the entire recorded
robot sweep. Tread width is the observed foot corridor plus a margin, and solid
completion down to the ground is a prior. Missing support levels are not
interpolated. Exact MuJoCo checks still cover every frame and every retained
collider; reference joint poses and root trajectories are unchanged.

```bash
$PYTHON -m scripts.whole_body_scene.run \
  --motion-file /path/to/source_shard.pt --ids 9977 \
  --output /path/to/stairs --static-scene-mode stairs --support-frames 0

$PYTHON -m scripts.whole_body_scene.render_scene \
  --manifest /path/to/stairs/manifest.jsonl --motion-id 9977 \
  --model "$PROTOMOTIONS_ASSET_ROOT/astro_p2/mjcf/astro_p2_protomotions.xml" \
  --output /path/to/p2_with_stairs.mp4 --title "Walk upstairs"
```

The video shows the actual retained scene boxes with the reference robot, plus
a travel/height projection of the same geometry, body origins and sole paths.
Teal/blue denotes inferred treads; amber denotes other contact hypotheses from
the generic full-body fitter. It checks source-motion, scene and model identity
before rendering and displays the quality status. It is kinematic replay, not
a policy rollout. A visually plausible staircase can remain `needs_review`.

The default minimum individual contact event duration is now **0.08 seconds**
(three frames at this dataset's approximately 30 Hz), with an explicit profile
in `configs/stairs_duration008.json`. The separate 0.22-second dwell for slow
pelvis/waist/knee support and the geometry thresholds remain unchanged.
The duration-only experiment below preceded the consensus fixes described next. Pass the profile with
`--config scripts/whole_body_scene/configs/stairs_duration008.json`.

The initial 0.12-second run retained eight treads each for train-7/9977,
train-1/9963 and train-5/9985, but only three for train-0/9969. The complete
0.08-second rerun and four additional non-mirrored ascending motions give:

| Source shard / local ID | Cohort | Treads at 0.12 s | Treads at 0.08 s |
| --- | --- | ---: | ---: |
| train7 / 9977 | original walk | 8 | 8 |
| train1 / 9963 | original walk | 8 | 8 |
| train5 / 9985 | original jog | 8 | 8 |
| train0 / 9969 | original jog | 3 | 9 |
| train1 / 9962 | additional jog | 3 | 6 |
| train5 / 9986 | additional jog | 4 | 7 |
| train7 / 9979 | additional walk | 7 | 7 |
| train2 / 9961 | additional walk | 7 | 8 |

All 1,088 reference frames were checked for geometry and support forces. In that duration-only baseline, only
train7/9979 passed the offline eligibility thresholds; the other seven
remain `needs_review`. No new training or closed-loop controller evaluation was
started. Shorter dwell also affects non-foot candidates: additional generic
contact hypotheses are visible in amber and require review. More detected
treads does not establish better force feasibility or correct auxiliary geometry.
The two additional jogs still contain gaps in their recovered stair profiles.
`DURATION008_VALIDATION.json` records the complete comparison and video hashes.
Output on machine 2 is
`/data/chenguanting/datasets/BONES-SEED/whole_body_scenes/stairs_duration008_20261016`.

## Fixed consensus and continuous supported intervals

The updated stair detector counts unique surface sample IDs once per frame and
requires at least three simultaneous samples for 0.08 seconds. It no longer
scales this threshold with the strongest footfall elsewhere in the clip.
Overlapping or duplicate events cannot inflate the sample count.

If the full observed window fails the 50% support-coverage test, the fitter can
retain its longest continuous interval with at least three downward-facing sole
samples inside the existing contact-distance and swept-space constraints. The
interval must still last at least 0.08 seconds; disconnected short fragments
cannot be combined. The original window, coverage and trimmed interval are
recorded in the scene. Existing full-window candidates take precedence.

Each tread ends no farther than its observed forward foot extent plus 3 cm.
Unobserved gaps remain explicit instead of becoming overlong treads. Exact
all-frame collision validation and offline training-eligibility checks still
apply; source joint trajectories are not fitted to the reconstructed geometry.

The 2026-10-16 validation cohort uses 12 distinct non-mirrored ascending motions:

| Source shard / local ID | Final retained treads |
| --- | ---: |
| train0 / 9969 | 9 |
| train1 / 9962 | 8 |
| train1 / 9963 | 8 |
| train2 / 9961 | 8 |
| train3 / 9973 | 8 |
| train5 / 9985 | 8 |
| train5 / 9986 | 9 |
| train6 / 9982 | 8 |
| train6 / 9983 | 8 |
| train7 / 9975 | 8 |
| train7 / 9977 | 8 |
| train7 / 9979 | 7 |

Compared with the duration-only baseline, 9962 improves from six to eight treads
and 9986 from seven to nine. On the expanded cohort, 9983 initially retained six:
its two rejected windows contained 10-frame supported segments, diluted to
43% and 48% coverage over their original windows. The interval fallback recovers
both observed levels, producing eight retained treads with no interpolated gap.

Four generic-mode controls cover train0/9953 (ladder ascent), 10029 (obstacle
vault), 5061 (crawl transition), and 263 (kneeling to standing). Generic fitting
still produces incomplete or spurious auxiliary geometry for these motions;
this is a limitation, not successful recovery of a complete ladder or obstacle.
All 16 samples remain `needs_review`; none are newly training-eligible. In
particular, train7/9979 changes from eligible in the duration-only baseline to
review because moment residual p95 is 0.200109, just above the 0.2 threshold.
The thresholds were not relaxed. Crawling and kneeling retain explicit constant
floor-clearance origin offsets (17.38 mm and 11.27 mm); all stair offsets are zero.

The final regression suite passes 152 tests. Both geometry and support checks
cover all 2,345 frames. No new training or controller rollout was launched.
Machine-2 result roots are:

- `/data/chenguanting/datasets/BONES-SEED/whole_body_scenes/stairs_supported_intervals_20261016`
- `/data/chenguanting/datasets/BONES-SEED/whole_body_scenes/diverse_contacts_20261016`

## Persistent motion exclusions

`protomotions/data/motion_exclusions.json` is loaded automatically by offline
reconstruction, scene packing and MotionLib for every training stage and
evaluation on this branch. The two rejected downstairs sources are excluded by
key, including their terminal `_M` mirrors. In the current validation shard,
these resolve to IDs **4522, 4523, 4524, 4525**. The full ten-shard audit found
no copies in train or test. Paths may move and derived packages may renumber
motions: matching uses a path-component suffix rather than a global integer ID.

Add another rejected source using the ID in its exact input shard:

```bash
$PYTHON -m scripts.motion_exclusions add \
  --motion-file /path/to/source_shard.pt --ids 12,34 \
  --reason 'Bad retarget trajectory'
```

This updates the repository policy. Mirrors are included by default; use
`--exact-only` to exclude only the selected variant. Use `--file /path/to/extra.json`
to write an additional policy instead. Pass that additional file with
`--exclude-file` to reconstruction/packing/audit and with
`--motion-exclusion-file` to `protomotions.train_agent`. These options are
repeatable and additive; repository rules always apply, including when an old
`resolved_configs.pt` is restored. Policy edits take effect on the next load;
already running processes are not modified.

```bash
$PYTHON -m scripts.motion_exclusions audit \
  --motion-file /path/to/train_slurmrank.pt /path/to/validation.pt /path/to/test.pt \
  --output /path/to/exclusion_audit.json

$PYTHON -m scripts.motion_exclusions filter-manifest \
  --manifest /path/to/reconstruction/manifest.jsonl
```

Filtering an existing manifest preserves a hash-named backup and excluded-row
audit; source files and historical geometry remain available for diagnosis.
The release pilot's active manifest now contains 10 records after removing
4522 and 4524. Its accepted packed motion (source 1694) remains unchanged.

MotionLib retains source IDs and frame tensors. Blocked sampling weights stay
zero after curriculum updates and checkpoint restore. Evaluators enumerate
only eligible IDs; an explicitly forced blocked motion or a scene paired with
one fails before use. Repack such an old scene bundle after filtering instead
of assigning a different motion to that scene. Empty eligible libraries fail
explicitly. Packages missing source keys must be repackaged with their source
identity so exclusions can be checked. Legacy MotionManager integer exclusions
remain supported for sampling; use the shared key policy for exclusions across
reconstruction, packing, training and evaluation.

## Coordinates and identities

Descending motions may cross the source z=0 plane. An **entire-clip constant**
`scene_origin_offset` is computed from the exact vertical lower extent of the
collision sweep. Reconstruction uses source position plus this offset. The
packed MotionLib applies the identical offset to all body positions; joint
poses, rotations, velocities, durations and relative motion remain unchanged.
Source files are never edited. There is no per-frame floor snapping.

Packing generates `scenes.pt`, `scenes.motions.pt`, `scenes.manifest.jsonl` and
`scenes.pt.binding.json`. Use the generated motion file with the generated scene
file. The packed motion ID is contiguous; `source_motion_local_id` and the
original key preserve the original shard index. Hash checks fail before
simulator creation if the motion, scene, manifest or collision asset differs.
Contact arrays are checked when loaded. Unbound motion resampling is forbidden.

Contact NPZs contain `[T,B]` state (-1 unknown, 0 clear, 1 inferred contact),
confidence, type and object ID. `-2` object ID means multiple objects. Ragged
`patch_offsets[T*B+1]` index world points, normals, body-local points, confidence,
object IDs and event IDs. JSON events/graphs preserve hypotheses and rejection
reasons. Training uses one strongest patch per body/object with an explicit
capacity check; full patch information is retained offline.

## Stage 1 and validation

```bash
$PYTHON -m unittest scripts.whole_body_scene.test_pipeline -v

$PYTHON -u -m scripts.whole_body_scene.smoke_stage1 \
  --simulator isaaclab --device cpu \
  --motion-file /path/to/packed/scenes.motions.pt \
  --scenes-file /path/to/packed/scenes.pt \
  --output /path/to/smoke_report
```

The smoke uses 2 environments and an untrained contact-aware FSQ. It checks
scene/robot reset, contact-conditioned observations/rewards, forward/backward,
one optimizer step, checkpoint save/load equality, subset contexts and real
physics steps. **It does not measure tracking success.** The repository's
MuJoCo simulator cannot instantiate SceneLib objects; offline MuJoCo collision
checks use an independently assembled model. Scene integration uses Isaac Lab.

After generating a **training** cohort, a new experiment can use:

```bash
$PYTHON -m protomotions.train_agent --robot-name p2 --simulator isaaclab \
  --headless --experiment-path examples/experiments/mimic/fsq_whole_body_scene.py \
  --experiment-name p2_fsq_whole_body_scene \
  --motion-file /path/to/train/packed/scenes_slurmrank.motions.pt \
  --scenes-file /path/to/train/packed/scenes_slurmrank.pt \
  --num-envs 1024 --batch-size 1024 --ngpu 8
```

This creates a different encoder input contract from earlier FSQ checkpoints.
Existing flat-ground Stage 2 tokens are not compatible by assumption: token
extraction must use the matching contact-conditioned encoder and plan. This
change does not implement a new Stage 2 contact predictor or Stage 3 controller.

## Limits and interpretation

- `geometry_validated` means retained objects passed the recorded-frame geometry
  checks and the configured quality screening. `physical_execution_validated`
  remains false until a trained controller is evaluated. All force scores,
  self-collisions and rejected objects remain inspectable.
- Current screening uses 8 mm penetration tolerance, p95 base-force residual
  below 0.3 robot weights, p95 base-moment residual below 0.2 of robot weight
  times 0.5 m, and p95 joint residual below 0.2 of effort limits. These are
  engineering screening thresholds, not calibrated physical success rates.
- Continuous collision between recorded frames is not verified. Conservative
  sampling can miss candidate contacts; exact replay can reject fitted boxes.
  No manual annotation set has yet established contact precision/recall.
- Input trajectories may already have lost the original terrain relationship.
  In the installed `p2_motionlib_lze_contact_orin` validation set, downstairs
  clips 4522 and 4524 keep the feet near the floor while the pelvis descends
  almost to it. Their FK matches the P2 model; scene fitting cannot recover
  missing step heights without changing those reference trajectories. They
  and their mirrors are now explicitly excluded by the shared key policy. Source
  filenames are used for cohort selection, never as geometry evidence.
- The explicit z=0 ground is a prior. Unobserved support legs and compound-object
  associations are also priors, separately labeled with low confidence.
- Static 3D primitives include vertical/inclined support. The system does not
  claim semantic completion of a true chair or table from sparse contact points.
- Dynamic objects are underdetermined hypotheses unless additional shape and
  grasp constraints exist. Free dynamic object training and A0–A4 ablations
  require subsequent validated data/controller experiments.
- The smoke currently uses geometric patch association with net link forces.
  It cannot prove which nearby collider contributed a force in an ambiguous
  multi-object contact. No unknown label is used as negative ground truth.

## Method references

The implementation is original code guided by the supplied WBC-SR design:
[GigaBrain-WBC-0.5, Section 3.3](https://arxiv.org/html/2608.18234v4),
[SceneBot](https://arxiv.org/abs/2606.27581), and
[TERRA](https://github.com/amathislab/terra). These references are not a claim
that this implementation reproduces their benchmark results.

## V2 whole-body reconstruction

The new profile `configs/whole_body_v2.json` enables deterministic symmetric
sampling, temporal plane hypotheses, rolling sole patches, connected solid
fitting, articulated support-cost selection and scene-conditioned contact
re-estimation. It removes automatic thin-column completion from this path.
See [V2_DESIGN.md](V2_DESIGN.md) for the algorithm, commands, priors and limits.
Legacy modes and the persistent exclusion policy remain available.

The v2 development cohort checks all 4,677 frames in 33 motions (24 upstairs and
nine other examples). Treads total 198 versus 193 in the earlier full-stair
baseline; the 9982/9976 mirror pair both retain eight levels and 9969 retains
nine. Automatic thin columns are absent. Five records pass the offline gates;
28 remain `needs_review`, including unresolved ladder/obstacle cases and
self-colliding references. Four auxiliary wall hypotheses remain in upstairs
scenes. This is not ground-truth reconstruction accuracy. The old stair baseline
uses 24 random samples, the generic baseline 96; v2 uses symmetric resolution 96.

`V2_VALIDATION.json` records 194 passing tests (one unrelated LFS asset test was
excluded), real MotionLib/SceneLib/contact binding checks, default rejection of
review bundles, and an Isaac Lab smoke on the new eight-tread scene: two
environments, eight physics steps, FSQ backward and checkpoint roundtrip.
No GPC training or tracking-quality evaluation was started. Machine-2 artifacts:
`/data/chenguanting/datasets/BONES-SEED/whole_body_scenes/v2_20261018_final`.
