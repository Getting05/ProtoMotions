from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares


BONE_SCALES = ("torso", "upper_arm", "forearm", "thigh", "shank")
SIDE_SCALES = tuple(f"{side}_{bone}" for side in ("left", "right") for bone in BONE_SCALES[1:])
OFFSET_JOINTS = ("shoulder", "elbow", "hip")


@dataclass
class CalibrationParams:
    upper_scale: np.ndarray
    lower_scale: np.ndarray
    shoulder_offset: float = 0.0
    elbow_offset: float = 0.0
    mode: str = "legacy"
    bone_scales: dict[str, float] = field(default_factory=dict)
    joint_offsets: dict[str, np.ndarray] = field(default_factory=dict)
    asymmetric: bool = False

    @classmethod
    def defaults(cls, mode: str = "legacy") -> "CalibrationParams":
        return cls(np.ones(3), np.ones(3), mode=mode)

    @classmethod
    def from_mapping(cls, data: Mapping | None) -> "CalibrationParams":
        data = dict(data or {})
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown calibration fields: {sorted(unknown)}")
        result = cls(**{"upper_scale": np.ones(3), "lower_scale": np.ones(3), **data})
        result.validate()
        return result

    def validate(self) -> None:
        if self.mode not in ("legacy", "bone"):
            raise ValueError("calibration.mode must be legacy or bone")
        if not isinstance(self.asymmetric, bool):
            raise ValueError("asymmetric must be a boolean")
        for name in ("upper_scale", "lower_scale"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all() or np.any(value <= 0):
                raise ValueError(f"{name} must contain three finite positive scales")
        for name in ("shoulder_offset", "elbow_offset"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if set(self.bone_scales) - set(BONE_SCALES + SIDE_SCALES):
            raise ValueError("Unknown bone_scales key")
        for value in self.bone_scales.values():
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Bone scales must be finite and positive")
        if set(self.joint_offsets) - set(OFFSET_JOINTS):
            raise ValueError("joint_offsets supports shoulder, elbow and hip")
        for value in self.joint_offsets.values():
            value = np.asarray(value, dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("Joint offsets must contain three finite coordinates")

    def serializable(self) -> dict:
        self.validate()
        return {
            "mode": self.mode,
            "upper_scale": [float(x) for x in self.upper_scale],
            "lower_scale": [float(x) for x in self.lower_scale],
            "shoulder_offset": float(self.shoulder_offset),
            "elbow_offset": float(self.elbow_offset),
            "bone_scales": {k: float(v) for k, v in self.bone_scales.items()},
            "joint_offsets": {k: [float(x) for x in v] for k, v in self.joint_offsets.items()},
            "asymmetric": self.asymmetric,
        }

    def parameter_specs(self) -> list[tuple[str, float, float, float]]:
        if self.mode == "legacy":
            return [(f"{part}_{axis}", .3, 2., .01) for part in ("upper", "lower") for axis in "xyz"] + [
                (f"{joint}_offset_m", -.2, .2, .005) for joint in OFFSET_JOINTS[:2]]
        scales = ("torso",) + SIDE_SCALES if self.asymmetric else BONE_SCALES
        return [(key, .3, 2., .01) for key in scales] + [
            (f"{joint}_{axis}_m", -.2, .2, .005) for joint in OFFSET_JOINTS for axis in "xyz"]

    def vector(self) -> np.ndarray:
        if self.mode == "legacy":
            return np.r_[self.upper_scale, self.lower_scale, self.shoulder_offset, self.elbow_offset]
        scales = ("torso",) + SIDE_SCALES if self.asymmetric else BONE_SCALES
        return np.array([self.bone_scales.get(k, self.bone_scales.get(k.split("_", 1)[-1], 1.)) for k in scales] +
                        [x for k in OFFSET_JOINTS for x in self.joint_offsets.get(k, np.zeros(3))])

    def with_vector(self, values: np.ndarray) -> "CalibrationParams":
        data = self.serializable()
        if self.mode == "legacy":
            data.update(upper_scale=values[:3], lower_scale=values[3:6],
                        shoulder_offset=float(values[6]), elbow_offset=float(values[7]))
        else:
            scales = ("torso",) + SIDE_SCALES if self.asymmetric else BONE_SCALES
            data["bone_scales"].update(dict(zip(scales, values[:len(scales)], strict=True)))
            data["joint_offsets"] = {k: values[len(scales)+3*i:len(scales)+3*i+3] for i, k in enumerate(OFFSET_JOINTS)}
        return self.from_mapping(data)


UPPER = {
    "spine1", "spine2", "spine3", "neck", "head", "left_collar",
    "right_collar", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
}


def calibrate_human_points(
    points: Mapping[str, np.ndarray], params: CalibrationParams
) -> dict[str, np.ndarray]:
    params.validate()
    if params.mode == "bone":
        return _calibrate_bones(points, params)
    root = np.asarray(points["pelvis"])
    output: dict[str, np.ndarray] = {}
    for name, point in points.items():
        scale = params.upper_scale if name in UPPER else params.lower_scale
        value = root + (np.asarray(point) - root) * scale
        side = 1.0 if name.startswith("left_") else -1.0 if name.startswith("right_") else 0.0
        if "shoulder" in name:
            value = value + np.array([0.0, side * params.shoulder_offset, 0.0])
        if "elbow" in name:
            value = value + np.array([0.0, side * params.elbow_offset, 0.0])
        output[name] = value
    return output


def _calibrate_bones(points: Mapping[str, np.ndarray], params: CalibrationParams) -> dict[str, np.ndarray]:
    from .human import SMPL_EDGES
    parents = {child: parent for parent, child in SMPL_EDGES}
    output = {"pelvis": np.asarray(points["pelvis"], dtype=float).copy()}

    def visit(name: str) -> np.ndarray:
        if name in output:
            return output[name]
        if name not in parents or parents[name] not in points:
            raise ValueError(f"Bone mode requires SMPL parent for {name}: {parents.get(name)}")
        parent = parents[name]
        side, _, joint = name.partition("_")
        group = {"elbow": "upper_arm", "wrist": "forearm", "knee": "thigh", "ankle": "shank"}.get(joint)
        if name in ("spine1", "spine2", "spine3", "neck"):
            group = "torso"
        scale = params.bone_scales.get(group, 1.)
        if params.asymmetric and group and group != "torso":
            scale = params.bone_scales.get(f"{side}_{group}", scale)
        value = visit(parent) + scale * (np.asarray(points[name]) - np.asarray(points[parent]))
        if joint in OFFSET_JOINTS:
            offset = np.asarray(params.joint_offsets.get(joint, np.zeros(3)), dtype=float)
            value = value + offset * np.array([1., 1. if side == "left" else -1., 1.])
        output[name] = value
        return value

    for name in points:
        visit(name)
    return {name: output[name] for name in points}


def fit_calibration(
    human: Mapping[str, np.ndarray],
    robot: Mapping[str, np.ndarray],
    names: Sequence[str],
    *,
    initial: CalibrationParams | None = None,
    scale_regularization: float = 0.08,
    offset_regularization: float = 0.02,
) -> tuple[CalibrationParams, dict]:
    """Fit one paired pose; see fit_calibration_frames for synchronized motion."""
    return fit_calibration_frames([human], [robot], names, initial=initial,
                                  scale_regularization=scale_regularization,
                                  offset_regularization=offset_regularization)


def fit_calibration_frames(
    human: Sequence[Mapping[str, np.ndarray]],
    robot: Sequence[Mapping[str, np.ndarray]],
    names: Sequence[str],
    *,
    initial: CalibrationParams | None = None,
    scale_regularization: float = 0.08,
    offset_regularization: float = 0.02,
) -> tuple[CalibrationParams, dict]:
    """Fit the active mode; inactive parameters are preserved, never optimized."""
    if len(human) == 0 or len(human) != len(robot):
        raise ValueError("Human and robot must have the same nonzero frame count")
    initial = initial or CalibrationParams.defaults()
    initial.validate()
    if scale_regularization < 0 or offset_regularization < 0:
        raise ValueError("Regularization must be nonnegative")
    common = list(dict.fromkeys(name for name in names if all(name in h and name in r for h, r in zip(human, robot, strict=True))))
    if "pelvis" not in common or len(common) < 4:
        raise ValueError("Calibration requires pelvis and at least three other correspondences")
    for frames in (human, robot):
        for frame in frames:
            for point in frame.values():
                if np.asarray(point).shape != (3,) or not np.isfinite(point).all():
                    raise ValueError("Each frame must contain finite XYZ points")

    def unpack(x: np.ndarray) -> CalibrationParams:
        return initial.with_vector(x)

    def residual(x: np.ndarray, regularize: bool = True) -> np.ndarray:
        params = unpack(x)
        data = []
        for h, r in zip(human, robot, strict=True):
            pred = calibrate_human_points(h, params)
            data.extend(pred[n] - h["pelvis"] - (np.asarray(r[n]) - r["pelvis"]) for n in common)
        result = np.concatenate(data)
        if regularize:
            result = np.concatenate([
                result,
                np.sqrt(len(human)) * scale_regularization * (x[:scale_count] - 1.0),
                np.sqrt(len(human)) * offset_regularization * x[scale_count:],
            ])
        return result

    x0 = initial.vector()
    specs = initial.parameter_specs()
    scale_count = 6 if initial.mode == "legacy" else len(x0) - 9
    lower = np.array([s[1] for s in specs])
    upper = np.array([s[2] for s in specs])
    solved = least_squares(residual, np.clip(x0, lower, upper), bounds=(lower, upper))
    raw_jac = solved.jac[: len(human) * len(common) * 3]
    singular = np.linalg.svd(raw_jac, compute_uv=False)
    rank = int(np.linalg.matrix_rank(raw_jac, tol=1e-7))
    rmse = float(np.sqrt(np.mean(residual(solved.x, regularize=False) ** 2)))
    info = {
        "rmse_m": rmse,
        "rank": rank,
        "parameter_count": len(x0),
        "mode": initial.mode,
        "success": bool(solved.success),
        "message": str(solved.message),
        "condition": float(singular[0] / max(singular[-1], 1e-12)),
        "correspondence_count": len(common),
        "frame_count": len(human),
        "warning": None if rank == len(x0) else (
            "The selected poses do not independently constrain every parameter. "
            "Treat unconstrained values as initial suggestions and verify with more poses."
        ),
    }
    return unpack(solved.x), info


def _distance(points: Mapping[str, np.ndarray], a: str, b: str) -> float | None:
    if a not in points or b not in points:
        return None
    return float(np.linalg.norm(np.asarray(points[a]) - np.asarray(points[b])))


def geometry_metrics(points: Mapping[str, np.ndarray]) -> dict[str, float | None]:
    def average(pairs: list[tuple[str, str]]) -> float | None:
        values = [_distance(points, *pair) for pair in pairs]
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None

    shoulder_mid = None
    if "left_shoulder" in points and "right_shoulder" in points:
        shoulder_mid = (np.asarray(points["left_shoulder"]) + np.asarray(points["right_shoulder"])) / 2
    torso = None
    if shoulder_mid is not None and "pelvis" in points:
        torso = float(np.linalg.norm(shoulder_mid - np.asarray(points["pelvis"])))
    return {
        "shoulder_width": _distance(points, "left_shoulder", "right_shoulder"),
        "elbow_span": _distance(points, "left_elbow", "right_elbow"),
        "hip_width": _distance(points, "left_hip", "right_hip"),
        "torso_length": torso,
        "upper_arm": average([("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow")]),
        "forearm": average([("left_elbow", "left_wrist"), ("right_elbow", "right_wrist")]),
        "thigh": average([("left_hip", "left_knee"), ("right_hip", "right_knee")]),
        "shank": average([("left_knee", "left_ankle"), ("right_knee", "right_ankle")]),
    }

