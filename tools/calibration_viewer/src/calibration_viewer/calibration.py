from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares


@dataclass
class CalibrationParams:
    upper_scale: np.ndarray
    lower_scale: np.ndarray
    shoulder_offset: float = 0.0
    elbow_offset: float = 0.0

    @classmethod
    def defaults(cls) -> "CalibrationParams":
        return cls(np.ones(3), np.ones(3), 0.0, 0.0)

    def serializable(self) -> dict:
        value = asdict(self)
        value["upper_scale"] = [float(x) for x in self.upper_scale]
        value["lower_scale"] = [float(x) for x in self.lower_scale]
        value["shoulder_offset"] = float(self.shoulder_offset)
        value["elbow_offset"] = float(self.elbow_offset)
        return value


UPPER = {
    "spine1", "spine2", "spine3", "neck", "head", "left_collar",
    "right_collar", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
}


def calibrate_human_points(
    points: Mapping[str, np.ndarray], params: CalibrationParams
) -> dict[str, np.ndarray]:
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


def fit_calibration(
    human: Mapping[str, np.ndarray],
    robot: Mapping[str, np.ndarray],
    names: Sequence[str],
    *,
    initial: CalibrationParams | None = None,
    scale_regularization: float = 0.08,
    offset_regularization: float = 0.02,
) -> tuple[CalibrationParams, dict]:
    """Fit scales and lateral shoulder/elbow offsets with bounded least squares."""
    initial = initial or CalibrationParams.defaults()
    common = [name for name in names if name in human and name in robot]
    if "pelvis" not in common or len(common) < 4:
        raise ValueError("Calibration requires pelvis and at least three other correspondences")
    human_root = np.asarray(human["pelvis"])
    robot_root = np.asarray(robot["pelvis"])

    def unpack(x: np.ndarray) -> CalibrationParams:
        return CalibrationParams(x[:3], x[3:6], x[6], x[7])

    def residual(x: np.ndarray, regularize: bool = True) -> np.ndarray:
        pred = calibrate_human_points(human, unpack(x))
        data = [pred[n] - human_root - (np.asarray(robot[n]) - robot_root) for n in common]
        result = np.concatenate(data)
        if regularize:
            result = np.concatenate([
                result,
                scale_regularization * (x[:6] - 1.0),
                offset_regularization * x[6:8],
            ])
        return result

    x0 = np.r_[initial.upper_scale, initial.lower_scale,
               initial.shoulder_offset, initial.elbow_offset]
    solved = least_squares(
        residual, x0, bounds=(np.r_[np.full(6, 0.3), -0.20, -0.20],
                              np.r_[np.full(6, 2.0), 0.20, 0.20])
    )
    raw_jac = solved.jac[: len(common) * 3]
    singular = np.linalg.svd(raw_jac, compute_uv=False)
    rank = int(np.linalg.matrix_rank(raw_jac, tol=1e-7))
    rmse = float(np.sqrt(np.mean(residual(solved.x, regularize=False) ** 2)))
    info = {
        "rmse_m": rmse,
        "rank": rank,
        "parameter_count": 8,
        "condition": float(singular[0] / max(singular[-1], 1e-12)),
        "correspondence_count": len(common),
        "warning": None if rank == 8 else (
            "The canonical pose does not independently constrain every parameter. "
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

