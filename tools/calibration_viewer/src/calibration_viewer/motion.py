"""Synchronized motion I/O and explicit root/body coordinate conventions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .calibration import CalibrationParams, calibrate_human_points
from .human import (HumanSkeleton, _array, _axis_vector, _names_from_data,
                    _read_pickle, _joints_from_smplx)


@dataclass
class RootTrajectoryParams:
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @classmethod
    def from_mapping(cls, data: Mapping | None) -> "RootTrajectoryParams":
        data = dict(data or {})
        if set(data) - {"scale", "offset"}:
            raise ValueError("root_trajectory supports scale and offset")
        result = cls(np.asarray(data.get("scale", [1., 1., 1.]), dtype=float),
                     np.asarray(data.get("offset", [0., 0., 0.]), dtype=float))
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in (result.scale, result.offset)) or np.any(result.scale <= 0):
            raise ValueError("Root scale/offset must be finite XYZ vectors with positive scales")
        return result

    def serializable(self) -> dict:
        return {"scale": [float(v) for v in self.scale], "offset": [float(v) for v in self.offset]}

    def apply(self, positions: np.ndarray) -> np.ndarray:
        """Scale displacements in XY and absolute height in Z, then translate."""
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0 or not np.isfinite(positions).all():
            raise ValueError("Root positions must be finite [T,3]")
        origin = positions[0] * [1., 1., 0.]
        return origin + (positions - origin) * self.scale + self.offset


def fit_root_trajectory(source: np.ndarray, target: np.ndarray, initial: RootTrajectoryParams) -> tuple[RootTrajectoryParams, dict]:
    source, target = np.asarray(source), np.asarray(target)
    if source.shape != target.shape or not np.isfinite(target).all():
        raise ValueError("Paired root trajectories must have matching finite [T,3] arrays")
    initial.apply(source)  # Shape validation before optimization.
    def residual(x):
        return (RootTrajectoryParams(x[:3], x[3:]).apply(source) - target).ravel()
    solved = least_squares(residual, np.r_[np.clip(initial.scale, .3, 2.), initial.offset],
                           bounds=(np.r_[np.full(3, .3), np.full(3, -np.inf)],
                                   np.r_[np.full(3, 2.), np.full(3, np.inf)]))
    rank = int(np.linalg.matrix_rank(solved.jac, tol=1e-7))
    return RootTrajectoryParams(solved.x[:3], solved.x[3:]), {
        "rmse_m": float(np.sqrt(np.mean(residual(solved.x)**2))), "rank": rank,
        "parameter_count": 6, "success": bool(solved.success),
        "warning": None if rank == 6 else "Root scale/offset are not independently observable on every axis in this motion.",
    }


@dataclass
class HumanMotion:
    names: tuple[str, ...]
    points: np.ndarray  # [T,J,3], world coordinates in robot axes, meters.
    fps: float
    root_rotations: np.ndarray | None = None  # [T,3,3], body -> world.
    source_frames: np.ndarray | None = None

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        if self.points.ndim != 3 or self.points.shape[1:] != (len(self.names), 3) or not len(self.points) or not np.isfinite(self.points).all():
            raise ValueError("Motion points must be nonempty finite [T,J,3]")
        if len(set(self.names)) != len(self.names) or "pelvis" not in self.names:
            raise ValueError("Motion requires unique joint_names including pelvis")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Motion fps must be positive")
        if self.source_frames is None:
            self.source_frames = np.arange(len(self.points))
        if self.root_rotations is None:
            # Infer upright heading only; explicit quaternions retain full roll/pitch/yaw.
            for name in ("left_hip", "right_hip"):
                if name not in self.names:
                    raise ValueError("Heading inference requires both hips or root_quat_xyzw")
            lateral = self.points[:, self.names.index("left_hip")] - self.points[:, self.names.index("right_hip")]
            norm = np.linalg.norm(lateral[:, :2], axis=1)
            if np.any(norm < 1e-6):
                raise ValueError("Degenerate hip heading; supply explicit root_quat_xyzw")
            yaw = np.arctan2(-lateral[:, 0], lateral[:, 1])
            self.root_rotations = Rotation.from_euler("z", yaw[:, None]).as_matrix()
        self.root_rotations = np.asarray(self.root_rotations)
        if self.root_rotations.shape != (len(self.points), 3, 3) or not np.isfinite(self.root_rotations).all():
            raise ValueError("root_rotations must be finite [T,3,3]")
        if not np.allclose(self.root_rotations.transpose(0, 2, 1) @ self.root_rotations, np.eye(3), atol=1e-6) or not np.allclose(np.linalg.det(self.root_rotations), 1., atol=1e-6):
            raise ValueError("root_rotations must be proper rotation matrices")

    @property
    def roots(self) -> np.ndarray:
        return self.points[:, self.names.index("pelvis")]

    @property
    def local_points(self) -> np.ndarray:
        return np.einsum("tjk,tkl->tjl", self.points - self.roots[:, None], self.root_rotations)

    def frames(self) -> list[dict[str, np.ndarray]]:
        return [dict(zip(self.names, frame, strict=True)) for frame in self.local_points]

    def calibrated(self, params: CalibrationParams, root: RootTrajectoryParams) -> tuple[np.ndarray, np.ndarray]:
        local = np.asarray([[result[n] for n in self.names]
                            for frame in self.frames() for result in [calibrate_human_points(frame, params)]])
        roots = root.apply(self.roots)
        world = local @ self.root_rotations.transpose(0, 2, 1) + roots[:, None]
        return local, world


def load_human_motion(path: Path, *, axes=("z", "x", "y"), fps: float | None = None,
                      model_path: Path | None = None, gender="neutral",
                      start=0, stop: int | None = None, stride=1) -> HumanMotion:
    if start < 0 or stride < 1 or (stop is not None and stop <= start):
        raise ValueError("Require 0 <= start < stop and stride >= 1")
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            data = dict(data)
    else:
        data = _read_pickle(path)
    if not isinstance(data, Mapping):
        raise ValueError("Motion input must be a mapping")
    rate = fps if fps is not None else data.get("fps", data.get("mocap_framerate"))
    if rate is None:
        raise ValueError("Motion fps missing: supply --fps or an fps field")
    points = next((_array(data[key]) for key in ("joints", "J", "keypoints", "joint_positions", "positions") if key in data), None)
    if points is None:
        poses = next((_array(data[key]) for key in ("poses", "pose", "body_pose") if key in data), None)
        if poses is None or poses.ndim != 2 or model_path is None:
            raise ValueError("Motion needs joints [T,J,3], or pose parameters [T,D] and --smpl-model")
        indices = np.arange(len(poses))[start:stop:stride]
        # Load the licensed body model once for the whole sequence.
        import smplx
        model = smplx.create(str(model_path), model_type="smpl", gender=gender, batch_size=1)
        points = np.asarray([_joints_from_smplx(data, model_path, int(i), gender, model=model)[0] for i in indices])
    else:
        if points.ndim != 3:
            raise ValueError("Motion joints must have shape [T,J,3]")
        indices = np.arange(len(points))[start:stop:stride]
        points = points[indices]
    if len(indices) == 0:
        raise ValueError("Frame selection is empty")
    names = _names_from_data(data, points.shape[1])
    # Reuse the static adapter's axis validation, including signed permutations.
    HumanSkeleton(names, points[0]).transformed(axes)
    basis = np.stack([_axis_vector(a) for a in axes])
    rotations = None
    if "root_quat_xyzw" not in data:
        orient = data.get("global_orient")
        if orient is None:
            pose = data.get("poses", data.get("pose"))
            if pose is not None and np.asarray(pose).ndim == 2:
                orient = np.asarray(pose)[:, :3]
        if orient is not None:
            orient = np.asarray(orient, dtype=float)
            if orient.ndim == 1:
                orient = np.repeat(orient[None], int(indices[-1]) + 1, axis=0)
            if orient.shape[-1] != 3 or not np.isfinite(orient).all():
                raise ValueError("global_orient must contain finite rotation vectors")
            rotations = basis @ Rotation.from_rotvec(orient[indices]).as_matrix() @ basis.T
    if "root_quat_xyzw" in data:
        quat = np.asarray(data["root_quat_xyzw"], dtype=float)[indices]
        if quat.shape != (len(indices), 4) or not np.isfinite(quat).all() or np.any(np.linalg.norm(quat, axis=1) < 1e-8):
            raise ValueError("root_quat_xyzw must be finite nonzero [T,4]")
        rotations = basis @ Rotation.from_quat(quat).as_matrix() @ basis.T
    return HumanMotion(names, points @ basis.T, float(rate) / stride, rotations, indices)


def reference_robot_frames(path: Path, robot, motion: HumanMotion, *, source_fps: float) -> tuple[list[dict], np.ndarray | None]:
    """Load synchronized robot q_t and evaluate URDF FK in the configured base."""
    with np.load(path, allow_pickle=False) as data:
        names = tuple(str(n) for n in data["joint_names"])
        q = np.asarray(data["joint_positions"], dtype=float)
        if len(names) != len(set(names)) or set(names) != set(robot.actuated_joint_names):
            raise ValueError("Reference joint_names must match URDF actuated joints exactly")
        if q.ndim != 2 or q.shape[1] != len(names) or not np.isfinite(q).all():
            raise ValueError("Reference joint_positions must be finite [T,D]")
        if "fps" not in data or not np.isclose(float(data["fps"]), source_fps):
            raise ValueError("Human/reference fps must match; resample and synchronize before fitting")
        if int(motion.source_frames[-1]) >= len(q):
            raise ValueError("Reference motion is shorter than selected human frames")
        roots = np.asarray(data["root_positions"], dtype=float) if "root_positions" in data else None
        if roots is not None:
            if roots.shape != (len(q), 3) or not np.isfinite(roots).all():
                raise ValueError("Reference root_positions must be finite [T,3]")
            roots = roots[motion.source_frames]
        if "base_link" in data and str(data["base_link"]) != robot.base_link:
            raise ValueError("Reference base_link does not match config")
        rotations = np.repeat(np.eye(3)[None], len(q), axis=0)
        if "root_quat_xyzw" in data:
            quats = np.asarray(data["root_quat_xyzw"], dtype=float)
            if quats.shape != (len(q), 4) or not np.isfinite(quats).all() or np.any(np.linalg.norm(quats, axis=1) < 1e-8):
                raise ValueError("root_quat_xyzw must be finite nonzero [T,4]")
            rotations = Rotation.from_quat(quats).as_matrix()
        frames = []
        for index, row in enumerate(q[motion.source_frames]):
            values = dict(zip(names, row, strict=True))
            for name, (lo, hi) in robot.joint_limits.items():
                if not lo - 1e-6 <= values[name] <= hi + 1e-6:
                    raise ValueError(f"Reference joint {name} violates URDF limits")
            robot.update(values)
            frame = robot.keypoints()
            frames.append(frame)
            if roots is not None:
                roots[index] += rotations[motion.source_frames[index]] @ frame["pelvis"]
    return frames, roots
