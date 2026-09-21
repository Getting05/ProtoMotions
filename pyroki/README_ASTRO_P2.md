# Astro P2 motion retargeting

`batch_retarget_to_astro_p2_from_keypoints.py` adapts the existing PyRoki
trajectory optimizer to Astro P2's 30 actuated joints. It uses the simulation
joint order from `astro_p2_retarget.urdf`, plus two fixed virtual toe links used
only as optimization keypoints.

The complete AMASS pipeline accepts a packaged SMPL or SMPL-X MotionLib:

```bash
cd /data/chenguanting/ProtoMotions

./scripts/retarget_amass_to_robot.sh \
  ~/.venvs/protomotions-mujoco/bin/python \
  ~/.venvs/astro-p2-retarget/bin/python \
  /path/to/amass_smplx_train.pt \
  astro_p2 50 --source-skeleton smplx
```

Use `50` for an initial sample and `1` for all motions. Add `--clean` only when
you intentionally want to delete the pipeline's existing intermediate output
directories beside the input MotionLib.

For the SMPL-X ACCAD files currently under `/data/chenguanting/datasets/ACCAD`,
first create a packaged source MotionLib with the existing AMASS converter:

```bash
cd /data/chenguanting/ProtoMotions
PYTHONPATH=. ~/.venvs/protomotions-mujoco/bin/python \
  data/scripts/convert_amass_to_motionlib.py \
  /data/chenguanting/datasets \
  /data/chenguanting/datasets/amass_smplx_motionlib \
  --humanoid-type smplx \
  --motion-config data/yaml_files/amass_smplx_train.yaml
```

Then pass the resulting `.pt` file to the command above with
`--source-skeleton smplx`.

The output is `/path/to/proto-astro_p2.pt`. Inspect a sample before training:

```bash
PYTHONPATH=. ~/.venvs/protomotions-mujoco/bin/python \
  examples/motion_libs_visualizer.py \
  --motion_files /path/to/proto-astro_p2.pt \
  --robot astro_p2 --simulator mujoco
```

For direct keypoint input:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/.venvs/astro-p2-retarget/bin/python \
  pyroki/batch_retarget_to_astro_p2_from_keypoints.py \
  --keypoints-folder-path /path/to/keypoints-for-retarget \
  --output-dir /path/to/pyroki-retargeted-astro_p2 \
  --source-type smpl --no-visualize --skip-existing
```

The retargeted NPZ stores `joint_names`; the ProtoMotions converter validates
and reorders by these names before forward kinematics. This protects P2's
distinct joint order, where `head_joint` follows the arm joints.

The initial P2 proportions and optimization weights follow the repository's G1
settings. The P2-specific changes are pelvis/link mapping, 0.10 m fixed toe
points, local +X palm points, and the P2 URDF limits. They are a working initial
retargeting configuration and should be tuned after visual inspection of
walking, turning, crouching, and airborne motions.
