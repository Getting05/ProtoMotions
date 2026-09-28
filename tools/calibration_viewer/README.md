# Calibration Viewer

An interactive geometry-calibration tool for motion retargeting. The current
adapter loads an SMPL `.pkl` and a robot URDF, displays both skeletons in one
Viser scene, fits morphology parameters, and exports a reusable YAML file.

The package separates human loaders, robot loaders, calibration, and rendering
so later versions can add SMPL-X, BVH, FBX, MJCF, or USD without rewriting the
viewer.

## Features

- Loads three common SMPL pickle layouts:
  - precomputed `joints`, `J`, `keypoints`, `joint_positions`, or `positions`;
  - an SMPL model containing `v_template` and `J_regressor`;
  - pose parameters such as `poses`/`pose` when an SMPL model is supplied.
- Loads arbitrary URDF robots and maps semantic human keypoints to link origins.
- Renders URDF visual geometry and the SMPL body mesh when the source data makes
  them available. If referenced URDF mesh files are absent, collision geometry
  is used as a diagnostic fallback.
- Provides independent visibility switches for the URDF mesh, SMPL mesh, SMPL
  skeleton, robot keypoints, and residual lines.
- Provides a bone mode with torso, upper-arm, forearm, thigh and shank scales,
  optional independent left/right lengths, and symmetric shoulder/elbow/hip XYZ offsets.
- Retains the original upper/lower XYZ and lateral-offset model as legacy mode.
- Provides a joint slider for every actuated URDF joint to establish a robot
  canonical pose.
- Fits the active morphology parameters with bounded least squares and reports
  RMSE, numerical rank, and conditioning.
- Displays shoulder width, elbow span, hip width, torso, arm, and leg lengths.
- Exports the mapping, robot canonical pose, axis convention, and calibrated
  values to YAML.

## Install

```bash
git clone https://github.com/Getting05/calibration-viewer.git
cd calibration-viewer
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Pose-parameter pickle files also require the optional SMPL runtime:

```bash
pip install -e '.[smpl]'
```

SMPL body-model files are licensed separately and are not included.

## Included robot presets

| Robot | ProtoMotions URDF | Viewer config |
|---|---|---|
| Astro P2 | `protomotions/data/assets/astro_p2/urdf/astro_p2_retarget.urdf` | `configs/astro_p2.yaml` |
| Unitree G1 29-DOF | `protomotions/data/assets/urdf/for_retargeting/g1.urdf` | `configs/unitree_g1.yaml` |
| Unitree H1-2 27-DOF | `protomotions/data/assets/urdf/for_retargeting/h1_2.urdf` | `configs/unitree_h1_2.yaml` |

The H1 preset follows the `H1_2` model shipped by ProtoMotions. Original H1 and
H1-2 have different kinematic structures and should not share the same preset.

## Run with Astro P2

From a ProtoMotions checkout:

```bash
calibration-viewer \
  --smpl /path/to/smpl.pkl \
  --urdf protomotions/data/assets/astro_p2/urdf/astro_p2_retarget.urdf \
  --config /path/to/calibration-viewer/configs/astro_p2.yaml \
  --output astro_p2_calibration.yaml
```

For Unitree G1:

```bash
calibration-viewer \
  --smpl /path/to/smpl.pkl \
  --urdf protomotions/data/assets/urdf/for_retargeting/g1.urdf \
  --config /path/to/calibration-viewer/configs/unitree_g1.yaml \
  --output g1_calibration.yaml
```

For the Unitree H1-2 model included in ProtoMotions:

```bash
calibration-viewer \
  --smpl /path/to/smpl.pkl \
  --urdf protomotions/data/assets/urdf/for_retargeting/h1_2.urdf \
  --config /path/to/calibration-viewer/configs/unitree_h1_2.yaml \
  --output h1_2_calibration.yaml
```

The Unitree presets include `robot.mesh_dir` values matching the ProtoMotions
asset layout. Relative mesh directories are resolved from the URDF directory;
`--mesh-dir PATH` can override them for another checkout layout.

Open the URL printed by Viser, normally `http://localhost:8080`. For a remote
server, forward the port first:

```bash
ssh -L 8080:localhost:8080 user@server
```

If the pkl only stores pose parameters, also pass the SMPL model:

```bash
calibration-viewer ... --smpl-model /path/to/SMPL_NEUTRAL.pkl
```

Use `--frame N` to select a frame from a motion pickle. A noninteractive fit is
available for scripts and CI:

```bash
calibration-viewer ... --fit-only
```

Use `--hide-urdf-mesh` or `--hide-smpl-mesh` when you want either surface hidden
at startup. The same visibility options are available interactively in the
viewer. If a mesh cannot be loaded, the Diagnostics panel reports why; skeleton
and keypoint calibration remains available.

The SMPL surface is the pelvis-centered source mesh. Morphology sliders continue
to operate on the calibration skeleton/keypoints; the surface is intentionally
left unwarped so it can expose source-axis, pose, and model-loading problems.

The bundled Astro P2 preset bends both elbow joints by +90 degrees because that
URDF's elbow zero pose points each forearm forward. This makes the complete arm
straight in the initial T-pose; it is not a wrist-joint correction.

## Configure another robot

Copy `configs/generic_smpl_to_urdf.yaml`, then set:

1. `base_link` to the robot root used as the pelvis-centered frame.
2. Each semantic keypoint to the corresponding URDF link.
3. Joint angles under `t_pose_joint_positions` so the robot is in a convenient
   symmetric canonical pose.
4. `human.axes` to `[forward, left, up]` in the source SMPL coordinates. Signed
   axes such as `-z` are accepted.
5. Optional `robot.mesh_dir` when mesh files live outside the URDF directory.
   Relative values are interpreted from the URDF directory.

The included SMPL default is `[z, x, y]`. Exporters differ, so verify the axis
triad in the scene before interpreting fitted XYZ scales.

## Calibration semantics

The viewer has two explicit modes. Existing YAML without `calibration.mode` uses
`legacy`, preserving the original eight-parameter pelvis-centered XYZ scaling
and point-only lateral shoulder/elbow corrections.

Use `--mode bone` or the **Mode** dropdown for kinematic bone scaling. The
bundled presets now select this mode. Bone mode applies, in robot coordinates:

```text
p'[pelvis] = p[pelvis]
p'[j] = p'[parent(j)] + scale[j] * (p[j] - p[parent(j)]) + offset[j]
```

- `torso` scales the pelvis → spine1 → spine2 → spine3 → neck chain.
- `upper_arm`, `forearm`, `thigh`, `shank` scale shoulder → elbow,
  elbow → wrist, hip → knee, and knee → ankle, respectively.
- Other bones retain their vectors. Downstream joints follow their moved parent;
  shortening an upper arm preserves its forearm length and the torso.
- Shoulder, elbow and hip offsets use **meters**, `[forward, left, up]`.
  X/Z are shared by both sides; Y is mirrored (positive widens the pair).
  Offsets are applied at that joint and inherited by all descendants.
- **Independent left/right bones** (`--asymmetric`) enables side-specific
  overrides such as `left_upper_arm` and `right_shank`. Missing overrides use
  the shared scale. Shared scales are not also optimized in this mode.
- Legacy scales/offsets are inactive in bone mode and vice versa. Switching
  mode changes the model, and is not an exact conversion of old parameters.
- Bone mode requires the standard SMPL parent joints, including spine/collar
  intermediates; incomplete or unknown topology raises an explicit error.

Auto fit optimizes only the selected mode: 8 legacy parameters, 14 symmetric
bone parameters (5 scales + 9 offsets), or 18 asymmetric parameters. Scales
are bounded to 0.3–2.0 and offsets to ±0.20 m. Imported values outside fitting
bounds remain usable manually; fitting clips its starting point to the bounds.
Regularization favors scale 1 and offset 0. Numerical rank and conditioning are
computed from correspondence residuals without regularization. A single pose
can leave parameters correlated (especially elbow offset vs upper-arm scale);
a rank warning means the parameter values are not uniquely identified.

Static calibration is the first stage. The motion CLI can additionally fit
synchronized poses, calibrate world root trajectories, and run PyRoki IK (see
below). Neither stage changes URDF geometry or warps the displayed source SMPL
mesh. `--frame` still selects a single pose for the static viewer.

## Output and reuse

Version 2 exports record the mode and all parameter groups. Load the exported
file with `--config calibration.yaml` in the viewer or `--fit-only`; both paths
use the saved calibration as their initial state. Version 1 inputs remain
supported. A bone-mode example:

```yaml
format_version: 2
calibration:
  mode: bone
  asymmetric: false
  bone_scales:
    torso: 0.9
    upper_arm: 0.85
    forearm: 0.95
    thigh: 0.9
    shank: 0.9
  joint_offsets:
    shoulder: [0.02, 0.01, 0.01]
    elbow: [0.0, 0.0, 0.0]
    hip: [0.0, 0.02, -0.01]
```

The full export also contains source axes, correspondence mapping and robot
canonical joint positions. Consumers must explicitly support version 2 bone
semantics; an old retarget script that only reads `upper_scale`/`lower_scale`
will not apply these new parameters automatically. Python consumers can apply
`CalibrationParams.from_mapping(config["calibration"])` and
`calibrate_human_points(points, params)` after axis conversion. Preserve the
full SMPL tree, then select mapped robot target points for the separate IK step.
Pelvis position is preserved by this API; the viewer centers it for display.

## Motion calibration and PyRoki retargeting

The pipeline is explicit:

```text
SMPL world joints → axis conversion → root/body separation
                 → bone morphology → calibrated target keypoints
                 → PyRoki + JAXLS (q per frame) → named robot motion NPZ
```

Install the optional IK backend on Python 3.12 (CPU is sufficient):

```bash
pip install -e '.[ik,test]'
```

PyRoki is pinned to an upstream commit. Its JAXLS dependency currently tracks
upstream; the tested revisions are recorded in `docs/validation.md`. Some Linux
Python installations require development headers to build `pyliblzfse`. A
standalone Python installed by `uv python install 3.12` includes them.

### Input and coordinate conventions

Human motion can be a trusted SMPL pickle or a non-pickled NPZ:

- `joints`: finite float array `[T, J, 3]`, in **world coordinates and meters**.
- `joint_names`: strings `[J]`; optional only for standard 24-joint SMPL order.
- `fps`: positive scalar, or supply `--fps`. No guessed frame rate.
- Optional `root_quat_xyzw`: `[T,4]`, body-to-world rotation in the source
  coordinate convention. Quaternions are normalized; zero/nonfinite values fail.
- SMPL pose-parameter pickle/NPZ files are also accepted with `--smpl-model` and
  the `smpl` extra. Their `global_orient` (or the first three pose values) provides
  root orientation. The licensed body model is loaded once per sequence.

`human.axes` converts positions into `[forward,left,up]`. Explicit orientations
are converted by `R_robot = B R_source B^T`, with the same axis permutation B.
If orientation is absent, yaw is inferred from the left-minus-right hip vector
projected onto the horizontal plane; root roll and pitch remain zero. Supply
explicit orientations for leaning/inverted motions. Degenerate hip headings
fail explicitly.

Bone scaling and 3D offsets operate in each frame's root/body coordinates.
Targets are then rotated back into the world. Thus forward shoulder offsets
follow a turning person. The static viewer assumes its selected canonical pose
is already aligned with the robot; motion handling removes the root orientation.

### Root trajectory and height

```yaml
root_trajectory:
  scale: [0.9, 0.9, 0.95]
  offset: [0.0, 0.0, 0.02]
```

For the selected motion window, let `o = [root[0].x, root[0].y, 0]`. Then
`root'[t] = o + scale * (root[t] - o) + offset`. XY scales change displacement
and walking stride relative to the first frame; Z scales absolute height above
the world ground plane. Offset is a world XYZ translation in meters. Use
`--root-scale X Y Z` and `--root-offset X Y Z` to override YAML. This changes the
root trajectory independently of body bone lengths; it does not enforce feet
on the ground.

### Fit synchronized robot motion

The reference NPZ must have `joint_positions[T,D]`, unique `joint_names[D]`
matching the URDF actuated joints (order may differ), and `fps`. Optional
`root_positions[T,3]` and `root_quat_xyzw[T,4]` describe the **configured
base_link** in robot-world axes; missing orientation means identity. Optional
`base_link` must match the config. Reference joints must satisfy URDF limits.

```bash
calibration-viewer --smpl human_motion.npz --urdf robot.urdf \
  --config configs/astro_p2.yaml --fit-motion \
  --robot-motion reference_robot.npz --output robot_morphology.yaml
```

The fitter evaluates robot FK at each `q_t`, then optimizes one shared set of
morphology parameters over all paired body-local poses. Pelvis-relative
correspondences remove root translation from this objective. When reference
roots are present, a separate bounded least-squares fit calibrates world root
scale/offset, accounting for the mapped pelvis link's offset from `base_link`.
Without reference roots, existing root parameters are preserved. Both fits
report RMSE and rank; a stationary trajectory cannot identify every root scale.
Regularization is normalized by frame count, so duplicating frames does not
change the relative strength of the prior.

The human and robot motions must already be synchronized and share a source
frame rate. The tool validates frame rate and selected index coverage; it does
not infer temporal alignment or resample. `--start`, `--stop` (exclusive), and
`--stride` apply the same frame indices to both. A stride of N divides output
fps by N. Motion fitting never repeats a single robot T-pose as a fake reference.

### Retarget and inspect

```bash
calibration-viewer --smpl human_motion.npz --urdf robot.urdf \
  --config robot_morphology.yaml --retarget \
  --motion-output robot_motion.npz --preview-motion
```

You can combine `--fit-motion --robot-motion reference.npz --retarget` to fit
and then retarget in one run. `--preview-motion` starts Viser with a frame
slider, play/pause, calibrated targets, robot FK, world root trajectory and
per-frame error. It shows the URDF mesh when available.

The solver uses PyRoki FK and JAXLS least squares, explicit URDF joint-limit
constraints, a rest-pose prior and a previous-frame smoothness prior. The last
solution initializes the next frame. Input/calibrated pelvis trajectory and
root orientation are fixed; the solver optimizes actuated joint angles.
`--ik-position-weight`, `--ik-rest-weight`, `--ik-smoothness-weight` and
`--ik-max-iterations` control this solve. An independent yourdfpy FK check
verifies the PyRoki joint/link/base conventions before solving.

Output NPZ contains no pickled objects:

| Field | Shape / meaning |
|---|---|
| `joint_names`, `joint_positions` | `[D]`, `[T,D]`; named URDF joint order, radians or meters |
| `root_positions`, `root_quat_xyzw` | `[T,3]`, `[T,4]`; configured `base_link` world pose |
| `pelvis_positions` | `[T,3]`; calibrated world pelvis trajectory |
| `fps`, `source_frames` | effective frame rate, original input indices |
| `keypoint_names` | `[K]`; semantic correspondence names |
| `target_keypoints`, `robot_keypoints` | `[T,K,3]`; world coordinates after final bounded IK |
| `frame_rmse_m` | `[T]`; RMS Euclidean keypoint distance |
| `solver_iterations`, `solver_termination_criterion` | per-frame solve iterations and whether a numerical stopping criterion was reached |
| `metadata_json` | diagnostics, morphology and root parameters |

For a non-root configured base, playback composes its world pose with the
inverse URDF root-to-base FK transform. No assumption about joint array order
is needed downstream. This NPZ is not a ProtoMotions MotionLib `.pt` file.

This is position-only **kinematic retargeting**. Twist can be underconstrained;
priors select a solution. Unreachable targets remain approximate. Joint-limit
projection is reported, and errors are recomputed using the final bounded
joints. `--ik-error-threshold` (default 0.05 m) counts poor-fit frames; results
are saved with diagnostics rather than silently described as successful.
There are no foot-contact, self-collision, dynamics or balance guarantees.

### Reproducible demo

A generated 8-DOF humanoid provides a complete test without licensed body models:

```bash
python examples/generate_demo.py --output /tmp/calibration-demo
calibration-viewer --smpl /tmp/calibration-demo/human.npz \
  --urdf /tmp/calibration-demo/robot.urdf \
  --config /tmp/calibration-demo/config.yaml \
  --fit-motion --robot-motion /tmp/calibration-demo/reference.npz \
  --output /tmp/calibration-demo/fitted.yaml \
  --retarget --motion-output /tmp/calibration-demo/result.npz --preview-motion
```

## Test

```bash
pip install -e '.[test]'
pytest
```
