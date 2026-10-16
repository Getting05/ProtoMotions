# BONES-SEED / Astro P2 whole-body scenes

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
export PROTOMOTIONS_ASSET_ROOT=/data/chenguanting/ProtoMotions/protomotions/data/assets
PYTHON=/data/chenguanting/.venvs/protomotions-isaaclab/bin/python
```

Example representative validation cohort (no GPU required):

```bash
$PYTHON -m scripts.whole_body_scene.run \
  --motion-file /data/chenguanting/datasets/BONES-SEED/p2_motionlib_lze_contact_orin/bones_seed_validation_astro_p2.pt \
  --output /data/chenguanting/datasets/BONES-SEED/whole_body_scenes/validation_demo \
  --ids 2054,1408,4524,4522,4336,3588,1074,6146 --support-frames 0

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
  remain excluded by the support/self-collision quality screening. Source
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
