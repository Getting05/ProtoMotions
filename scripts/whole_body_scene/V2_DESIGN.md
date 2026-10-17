# Whole-body scene reconstruction v2

V2 is selected with `--config scripts/whole_body_scene/configs/whole_body_v2.json`.
It retains the original motion and the existing motion/scene/contact packing and
Stage 1 interface. It does not start training. The existing exclusions apply to
reconstruction, packing, training and evaluation.

## Why a separate version

The previous sample-vote detector was unstable across sampling seeds and mirrors.
Increasing random sample count did not fix truncated contact windows, unsupported
thin columns, missing seats, or false objects near slowly moving hands and heads.
V2 uses a separate inference path; `generic` and `stairs` remain available for
reproduction. Use a new output directory because the algorithm signature changes.

## Inference

1. **Symmetric collision surfaces.** Deterministic grids/rings cover every
   effective collision link. Samples are independent of RNG seed and geometry
   traversal order. `samples_per_geom=96` is a resolution budget, not exactly 96
   points per primitive. Area is measured in square metres rather than point
   count when generating plane/foot-support candidates.
2. **Temporal planes.** Search horizontal, vertical and inclined oriented
   envelopes over all collision links; merge nearby coplanar evidence across
   time and links. The default observed dwell remains 0.08 s. For feet, also
   detect a finite stable patch while the identity of the lowest sole point
   changes. This covers rolling feet in fast stair ascents. Candidate discovery
   records each window, area, approach, departure and normal.
3. **Connected geometry.** Fit grounded volumes, or a slab with one broad
   pedestal, and grounded wall candidates. Search cropped observed footprints
   when a union rectangle intersects another moving limb. No four-column
   automatic completion is used in v2. Stair treads keep the existing full-body
   swept-space bound and observed extent. A ladder hypothesis searches a shared
   rung plane and grounded side rails. Every completion is explicitly a prior.
4. **Conservative model selection.** A generic object needs an observed normal
   arrival before contact, persistent geometric contact and a positive marginal
   improvement after an object-complexity penalty. Foot-side proximity alone
   does not establish a wall. Score root equilibrium together with a soft joint
   effort prior using the MJCF motor limits. This is distinct from the final
   bounded torque/force validation. Previously selected objects are included
   when scoring the next candidate, so redundant supports gain little.
5. **Re-estimate contact on the final scene.** After exact collision validation,
   compute link/plane patch tracks using 12 mm entry and 25 mm exit proximity,
   normal-speed hysteresis and a minimum 0.08 s body-patch duration. A rolling
   contact can exchange surface sample identities. Real temporal gaps are not
   bridged. Ragged contact points retain body-local coordinates and object IDs.
6. **Independent validation and eligibility.** MuJoCo analytic primitive distance
   and contact-manifold checks cover every recorded frame. The original bounded
   full-joint inverse-dynamics solver then checks every frame with
   `--support-frames 0`. Existing penetration and residual thresholds are not
   relaxed. Self-colliding references, unresolved ladder structures, significant
   dynamic-object hypotheses and excessive residuals remain `needs_review`.

The selection objective is root-force error normalized by body weight, root
moment error normalized by weight times 0.5 m, plus 0.1 times the sum of squared
joint torques normalized by actuator limits. It is evaluated on up to 48 uniform
frames during candidate selection. The object penalty is 0.012 plus 0.002 for
each additional primitive. These are engineering priors, not learned confidence
or estimates of the original scene. Final force validation covers all frames in
the supplied evaluation profile.

## Usage

```bash
cd /data/chenguanting/ProtoMotions-wbc-scene
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
PYTHON=/data/chenguanting/.venvs/protomotions-isaaclab/bin/python
MODEL=results/wbc_assets/astro_p2/mjcf/astro_p2_protomotions.xml

$PYTHON -m scripts.whole_body_scene.run \
  --motion-file /path/to/bones_seed_train_astro_p2_0.pt \
  --ids 9969,2053,12057 --model "$MODEL" \
  --config scripts/whole_body_scene/configs/whole_body_v2.json \
  --support-frames 0 --output /path/to/new_v2_output

$PYTHON -m scripts.whole_body_scene.render_scene \
  --manifest /path/to/new_v2_output/manifest.jsonl --motion-id 2053 \
  --model "$MODEL" --output /path/to/sitting_with_scene.mp4 --title V2

$PYTHON -m scripts.whole_body_scene.pack \
  --manifest /path/to/new_v2_output/manifest.jsonl \
  --output /path/to/new_v2_output/packed/scenes.pt
```

`run_shards --config .../whole_body_v2.json` also selects v2; it does not override
the profile with generic mode unless an explicit mode argument is supplied.
Packing selects eligible records by default. `--include-needs-review` is an
inspection-only bundle; the Stage 1 loader rejects it by default. Ground-only
records still require all quality checks; there is no fallback that silently
turns a failed reconstruction into a training scene.

## Scope and limits

- Output is a finite static box scene, not a recovered visual mesh or unique
  scene truth. An underdetermined obstacle is not invented from its filename.
- Inclined-plane evidence is retained as unknown; fitting connected arbitrary
  sloped solids is not implemented in this version.
- Ladder hypotheses may remain unresolved when recorded whole-body trajectories
  intersect possible rungs or do not support a coherent ladder. A few isolated
  contact supports must not be described as a recovered ladder.
- Missing geometry, airborne motion, reference retargeting inconsistencies and
  finite-difference dynamics noise can all cause force/moment residuals. A
  smaller residual alone does not prove that a selected object is correct.
- Exact checks cover recorded poses, not continuous collision between frames.
  No closed-loop tracking success is claimed by reconstruction or rendering.
- The 33-clip benchmark is the development cohort: all 24 upstairs clips plus
  nine non-stair examples. It is not a held-out estimate of corpus-wide accuracy.
