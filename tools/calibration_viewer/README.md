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
- Shows the robot model, SMPL target skeleton, robot keypoints, and residuals.
- Provides sliders for upper/lower XYZ scale and shoulder/elbow lateral offsets.
- Provides a joint slider for every actuated URDF joint to establish a robot
  canonical pose.
- Fits the eight morphology parameters with bounded least squares and reports
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

## Run with Astro P2

From a ProtoMotions checkout:

```bash
calibration-viewer \
  --smpl /path/to/smpl.pkl \
  --urdf protomotions/data/assets/astro_p2/urdf/astro_p2_retarget.urdf \
  --config /path/to/calibration-viewer/configs/astro_p2.yaml \
  --output astro_p2_calibration.yaml
```

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

## Configure another robot

Copy `configs/generic_smpl_to_urdf.yaml`, then set:

1. `base_link` to the robot root used as the pelvis-centered frame.
2. Each semantic keypoint to the corresponding URDF link.
3. Joint angles under `t_pose_joint_positions` so the robot is in a convenient
   symmetric canonical pose.
4. `human.axes` to `[forward, left, up]` in the source SMPL coordinates. Signed
   axes such as `-z` are accepted.

The included SMPL default is `[z, x, y]`. Exporters differ, so verify the axis
triad in the scene before interpreting fitted XYZ scales.

## Calibration semantics

For every keypoint, calibration starts from its pelvis-relative vector. Upper
and lower body vectors receive independent diagonal XYZ scales. Shoulder and
elbow offsets move left/right keypoints outward with opposite signs along the
robot Y (left) axis. Offsets are point-level correspondence corrections; they
do not alter the URDF.

The auto-fit uses one canonical pose. Some parameter combinations can be weakly
observed or correlated in a T-pose, especially forward-axis scale and lateral
scale versus offsets. The viewer reports the fit rank and keeps regularization
near scale 1 and offset 0. Validate the exported values on several motions
before batch retargeting.

## Output

```yaml
calibration:
  upper_scale: [0.9, 0.9, 0.8]
  lower_scale: [0.9, 0.9, 0.85]
  shoulder_offset: 0.02
  elbow_offset: 0.045
```

The full file also contains the source axis convention, correspondence map, and
robot canonical joint positions, making a calibration reproducible.

## Test

```bash
pip install -e '.[test]'
pytest
```

