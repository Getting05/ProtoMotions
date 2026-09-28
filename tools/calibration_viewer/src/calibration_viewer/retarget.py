"""Position-target PyRoki IK with named joints, limits, and temporal continuity."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .calibration import CalibrationParams
from .motion import HumanMotion, RootTrajectoryParams


def retarget_motion(motion: HumanMotion, robot, params: CalibrationParams,
                    root: RootTrajectoryParams, *, initial_joints: dict | None = None,
                    position_weight=50., rest_weight=.01, smoothness_weight=.1,
                    max_iterations=100, error_threshold=.05) -> tuple[dict, dict]:
    """Keep calibrated pelvis trajectory/orientation; solve robot q_t with PyRoki.

    Heading is fixed by input/inference. This is kinematic IK, without contact,
    dynamics or collision constraints. Residual quality is measured after limits.
    """
    try:
        import jax
        import jax.numpy as jnp
        import jaxlie
        import jaxls
        import pyroki as pk
    except ImportError as exc:
        raise RuntimeError("PyRoki runtime missing. Install `pip install -e '.[ik]'` on Python 3.12.") from exc
    weights = np.array([position_weight, rest_weight, smoothness_weight, error_threshold])
    if not np.isfinite(weights).all() or np.any(weights < 0) or position_weight == 0 or max_iterations < 1:
        raise ValueError("IK requires positive position weight/iterations and nonnegative finite weights/threshold")
    if not robot.actuated_joint_names:
        raise ValueError("IK requires at least one actuated URDF joint")
    names = tuple(n for n in robot.keypoint_links if n in motion.names)
    if "pelvis" not in names or len(names) < 4:
        raise ValueError("IK requires pelvis and at least three mapped human joints")
    unsupported = [j.name for j in robot.urdf.robot.joints if j.type not in ("fixed", "revolute", "continuous", "prismatic")]
    if unsupported:
        raise ValueError(f"Unsupported PyRoki joint types: {unsupported}")
    pk_robot = pk.Robot.from_urdf(robot.urdf)
    joint_names = tuple(pk_robot.joints.actuated_names)
    if set(joint_names) != set(robot.actuated_joint_names):
        raise ValueError("PyRoki and URDF disagree on actuated joint names")
    rest = np.array([(initial_joints or {}).get(n, 0.) for n in joint_names], dtype=float)
    bounds = np.array([robot.joint_limits[n] for n in joint_names])
    rest = np.clip(rest, bounds[:, 0], bounds[:, 1])
    target_indices = jnp.array([pk_robot.links.names.index(robot.keypoint_links[n]) for n in names])
    pelvis_index = names.index("pelvis")
    base_index = pk_robot.links.names.index(robot.base_link)
    JointVar = pk_robot.joint_var_cls
    joint_var = JointVar(0)

    def local_fk(q):
        transforms = jaxlie.SE3(pk_robot.forward_kinematics(q))
        base_inverse = jaxlie.SE3(transforms.wxyz_xyz[base_index]).inverse()
        return (base_inverse @ transforms).translation()[target_indices]

    @jaxls.Cost.factory
    def alignment(values, var, target):
        positions = local_fk(values[var])
        return ((positions - positions[pelvis_index] - target) * position_weight).ravel()

    @jax.jit
    def solve_frame(target, previous):
        costs = [alignment(joint_var, target),
                 pk.costs.rest_cost(joint_var, jnp.asarray(rest), rest_weight),
                 pk.costs.rest_cost(joint_var, previous, smoothness_weight),
                 pk.costs.limit_constraint(pk_robot, joint_var)]
        result, summary = jaxls.LeastSquaresProblem(costs, [joint_var]).analyze().solve(
            initial_vals=jaxls.VarValues.make([joint_var.with_value(previous)]),
            verbose=False, linear_solver="dense_cholesky",
            termination=jaxls.TerminationConfig(max_iterations=max_iterations), return_summary=True)
        return result[joint_var], summary.iterations, jnp.any(summary.termination_criteria)

    local, world = motion.calibrated(params, root)
    target = local[:, [motion.names.index(n) for n in names]]
    previous = rest
    configs, actual, base_positions, clipping = [], [], [], []
    iterations, terminated = [], []
    pelvis_world = root.apply(motion.roots)
    # Verify the adapters agree by names, independently of their joint ordering.
    robot.update(dict(zip(joint_names, rest, strict=True)))
    expected = np.array([robot.keypoints()[n] for n in names])
    if not np.allclose(np.asarray(local_fk(jnp.asarray(rest))), expected, atol=1e-5):
        raise ValueError("PyRoki FK does not match URDF base/link mapping")
    for frame, frame_target in enumerate(target):
        solution, iteration_count, criterion = solve_frame(jnp.asarray(frame_target), jnp.asarray(previous))
        solved = np.asarray(solution, dtype=float)
        iterations.append(int(iteration_count))
        terminated.append(bool(criterion))
        if not np.isfinite(solved).all():
            raise RuntimeError(f"PyRoki returned non-finite joints at frame {frame}")
        limited = np.clip(solved, bounds[:, 0], bounds[:, 1])
        clipping.append(float(np.max(np.abs(solved - limited))))
        # Independent FK on the final bounded joints is the authoritative diagnostic.
        robot.update(dict(zip(joint_names, limited, strict=True)))
        positions = np.array([robot.keypoints()[n] for n in names])
        rotation = motion.root_rotations[frame]
        base_position = pelvis_world[frame] - rotation @ positions[pelvis_index]
        actual.append(positions @ rotation.T + base_position)
        base_positions.append(base_position)
        configs.append(limited)
        previous = limited
    target_world = world[:, [motion.names.index(n) for n in names]]
    error = np.linalg.norm(np.asarray(actual) - target_world, axis=-1)
    frame_rmse = np.sqrt(np.mean(error**2, axis=1))
    info = {
        "backend": "pyroki+jaxls", "frame_count": len(configs), "fps": motion.fps,
        "position_rmse_m": float(np.sqrt(np.mean(error**2))),
        "max_keypoint_error_m": float(error.max()),
        "error_threshold_m": error_threshold,
        "frames_above_threshold": int(np.count_nonzero(frame_rmse > error_threshold)),
        "max_limit_projection_rad_or_m": max(clipping),
        "frames_without_termination_criterion": int(np.count_nonzero(~np.asarray(terminated))),
        "constraints": "URDF joint limits; fixed calibrated pelvis trajectory and input/inferred orientation",
        "limitations": "Position-only kinematic IK; no contact, collision or dynamics validation.",
    }
    output = {
        "format_version": np.array(1), "fps": np.array(motion.fps),
        "joint_names": np.array(joint_names), "joint_positions": np.asarray(configs),
        "base_link": np.array(robot.base_link),
        "root_positions": np.asarray(base_positions),
        "root_quat_xyzw": Rotation.from_matrix(motion.root_rotations).as_quat(),
        "pelvis_positions": pelvis_world,
        "keypoint_names": np.array(names), "target_keypoints": target_world,
        "robot_keypoints": np.asarray(actual), "frame_rmse_m": frame_rmse,
        "source_frames": motion.source_frames,
        "solver_iterations": np.array(iterations),
        "solver_termination_criterion": np.array(terminated),
        "metadata_json": np.array(json.dumps({"diagnostics": info, "calibration": params.serializable(), "root_trajectory": root.serializable()})),
    }
    return output, info


def save_motion(path: Path, output: dict) -> None:
    if path.suffix.lower() != ".npz":
        raise ValueError("Robot motion output must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)
