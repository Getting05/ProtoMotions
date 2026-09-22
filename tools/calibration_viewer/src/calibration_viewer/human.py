from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SMPL_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee",
    "right_knee", "spine2", "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck", "left_collar", "right_collar",
    "head", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)

SMPL_EDGES = (
    ("pelvis", "left_hip"), ("pelvis", "right_hip"),
    ("pelvis", "spine1"), ("spine1", "spine2"), ("spine2", "spine3"),
    ("spine3", "neck"), ("neck", "head"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
    ("neck", "left_collar"), ("left_collar", "left_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand"),
    ("neck", "right_collar"), ("right_collar", "right_shoulder"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand"),
)


@dataclass(frozen=True)
class HumanSkeleton:
    names: tuple[str, ...]
    points: np.ndarray
    edges: tuple[tuple[str, str], ...] = SMPL_EDGES

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float64)
        if points.shape != (len(self.names), 3):
            raise ValueError(f"Expected ({len(self.names)}, 3) points, got {points.shape}")
        object.__setattr__(self, "points", points)

    @property
    def by_name(self) -> dict[str, np.ndarray]:
        return dict(zip(self.names, self.points, strict=True))

    def transformed(self, axes: Sequence[str]) -> "HumanSkeleton":
        """Return points in robot coordinates [forward, left, up]."""
        if len(axes) != 3:
            raise ValueError("axes must contain [forward, left, up]")
        basis = np.stack([_axis_vector(axis) for axis in axes], axis=0)
        if abs(np.linalg.det(basis)) < 0.5:
            raise ValueError(f"Coordinate axes are not independent: {axes}")
        return HumanSkeleton(self.names, self.points @ basis.T, self.edges)


def _axis_vector(value: str) -> np.ndarray:
    sign = -1.0 if value.startswith("-") else 1.0
    axis = value.removeprefix("-").lower()
    if axis not in "xyz" or len(axis) != 1:
        raise ValueError(f"Invalid signed axis {value!r}; use x, -x, y, -y, z, or -z")
    out = np.zeros(3)
    out["xyz".index(axis)] = sign
    return out


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        try:
            return pickle.load(stream, encoding="latin1")
        except TypeError:
            stream.seek(0)
            return pickle.load(stream)


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    if hasattr(value, "r"):
        value = value.r
    return np.asarray(value)


def _select_frame(points: np.ndarray, frame: int) -> np.ndarray:
    points = np.asarray(points)
    while points.ndim > 3 and points.shape[0] == 1:
        points = points[0]
    if points.ndim == 3:
        points = points[frame]
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"Joint array must end in [J, 3], got {points.shape}")
    return points


def _names_from_data(data: Mapping[str, Any], count: int) -> tuple[str, ...]:
    for key in ("joint_names", "joints_names", "keypoint_names", "body_names"):
        if key in data:
            names = tuple(str(x) for x in data[key])
            if len(names) == count:
                return names
    if count == len(SMPL_JOINT_NAMES):
        return SMPL_JOINT_NAMES
    raise ValueError(
        f"Found {count} joints but no joint_names. Standard SMPL has 24 joints; "
        "add joint_names to the pkl or pass a compatible file."
    )


def _joints_from_model(data: Mapping[str, Any]) -> np.ndarray:
    vertices = _array(data["v_template"]).astype(np.float64)
    shapedirs = data.get("shapedirs")
    betas = data.get("betas")
    if shapedirs is not None and betas is not None:
        dirs = _array(shapedirs).astype(np.float64)
        beta = _array(betas).reshape(-1).astype(np.float64)
        if dirs.ndim == 3:
            n = min(dirs.shape[-1], beta.size)
            vertices = vertices + np.tensordot(dirs[..., :n], beta[:n], axes=([-1], [0]))
        elif dirs.ndim == 2 and dirs.shape[0] == vertices.size:
            n = min(dirs.shape[-1], beta.size)
            vertices = vertices + (dirs[:, :n] @ beta[:n]).reshape(vertices.shape)
    regressor = _array(data["J_regressor"]).astype(np.float64)
    return regressor @ vertices


def _joints_from_smplx(
    data: Mapping[str, Any], model_path: Path, frame: int, gender: str
) -> np.ndarray:
    try:
        import smplx
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "This pkl contains SMPL pose parameters but no joints. Install the optional "
            "dependencies with `pip install 'calibration-viewer[smpl]'`."
        ) from exc
    model = smplx.create(str(model_path), model_type="smpl", gender=gender, batch_size=1)
    pose_value = data.get("poses", data.get("pose"))
    if pose_value is not None:
        poses = _array(pose_value)
        if poses.ndim == 2:
            poses = poses[frame]
        poses = poses.reshape(-1)
        global_orient, body_pose = poses[:3], poses[3:72]
    else:
        body_pose = _array(data["body_pose"])
        global_orient = _array(data.get("global_orient", np.zeros(3)))
        if body_pose.ndim == 2:
            body_pose = body_pose[frame]
        if global_orient.ndim == 2:
            global_orient = global_orient[frame]
        body_pose = body_pose.reshape(-1)[:69]
        global_orient = global_orient.reshape(-1)[:3]
    betas = _array(data.get("betas", np.zeros(10))).reshape(-1)[:10]
    transl = _array(data.get("trans", data.get("transl", np.zeros(3))))
    if transl.ndim == 2:
        transl = transl[frame]
    with torch.no_grad():
        output = model(
            global_orient=torch.as_tensor(global_orient, dtype=torch.float32)[None],
            body_pose=torch.as_tensor(body_pose, dtype=torch.float32)[None],
            betas=torch.as_tensor(betas, dtype=torch.float32)[None],
            transl=torch.as_tensor(transl, dtype=torch.float32)[None],
        )
    return output.joints[0, :24].cpu().numpy()


def load_smpl_pkl(
    path: str | Path,
    *,
    frame: int = 0,
    model_path: str | Path | None = None,
    gender: str = "neutral",
) -> HumanSkeleton:
    """Load precomputed joints, an SMPL model pkl, or an SMPL parameter pkl."""
    path = Path(path)
    data = _read_pickle(path)
    if not isinstance(data, Mapping):
        if hasattr(data, "__dict__"):
            data = vars(data)
        else:
            raise TypeError(f"Expected a mapping in {path}, got {type(data).__name__}")

    points = None
    for key in ("joints", "J", "keypoints", "joint_positions", "positions"):
        if key in data:
            candidate = _array(data[key])
            if candidate.shape[-1] == 3:
                points = _select_frame(candidate, frame)
                break
    if points is None and "v_template" in data and "J_regressor" in data:
        points = _joints_from_model(data)
    if points is None and any(k in data for k in ("poses", "pose", "body_pose")):
        if model_path is None:
            raise ValueError("SMPL parameter pkl requires --smpl-model PATH")
        points = _joints_from_smplx(data, Path(model_path), frame, gender)
    if points is None:
        raise ValueError(
            "Unsupported SMPL pkl. Expected joints/keypoints/positions, "
            "v_template + J_regressor, or pose parameters with --smpl-model."
        )
    names = _names_from_data(data, len(points))
    return HumanSkeleton(names, points)
