# Validation of the morphology and motion pipeline

Validated on 2026-09-28 using CPU execution:

- Python 3.12.14
- PyRoki `388e43e1fc0d0ee382968d3dd72970fd62a0450c`
- JAXLS `50a58be88c5ef74532f09e3f55268b4f02c490e3`
- JAX / jaxlib 0.11.2, jaxlie 1.5.0
- NumPy 2.5.3, SciPy 1.18.1
- Viser 1.1.1, yourdfpy 0.0.60

The complete suite passed: **31 tests**, including real PyRoki solves. Core
calibration tests also ran under Python 3.14 without installing the IK extra.

Coverage includes legacy recovery; independent bone scaling; propagation and
symmetry of joint offsets; standard SMPL collar parent topology; YAML roundtrip;
invalid input rejection; world root scale/offset recovery; heading-aware local
calibration; selected-frame/reference FK alignment; multi-pose recovery of all
14 symmetric morphology parameters; reachable IK; and unreachable-target error
reporting with bounded, finite joint output.

The generated demo completed the full CLI sequence:

```bash
python examples/generate_demo.py --output /tmp/calibration-demo
calibration-viewer --smpl /tmp/calibration-demo/human.npz \
  --urdf /tmp/calibration-demo/robot.urdf \
  --config /tmp/calibration-demo/config.yaml \
  --fit-motion --robot-motion /tmp/calibration-demo/reference.npz \
  --output /tmp/calibration-demo/fitted.yaml \
  --retarget --motion-output /tmp/calibration-demo/result.npz --preview-motion
```

For this reachable synthetic 12-frame, 8-DOF sequence, morphology fit rank was
14/14 and retargeting RMS Euclidean keypoint error was approximately 0.000001 m.
This is an implementation check, not an accuracy claim for Astro P2 or real SMPL
motion. No licensed SMPL body model or user-specific Astro P2 motion was supplied
for this task, so those assets were not used in the end-to-end test.

Real Viser servers and browser interaction verified:

- Static bone controls, left/right mode switching, unilateral upper-arm edits,
  geometry metric updates, auto fitting and YAML export.
- Motion play/pause, frame selection, target/robot overlay and trajectory display.
- Correction of an initial preview bug where world FK points inherited the robot
  mesh root transform twice; debug points now use a separate world scene branch.

The output is kinematic: contact, collision, dynamics and physical-policy
tracking have not been validated. Position-only IK may leave joint twist
ambiguous. Errors and solver termination diagnostics must be inspected on real
motion before downstream deployment.
