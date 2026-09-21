# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Retarget ProtoMotions G1 .motion files to Astro P2 (30 DOF).

This script is adapted from ProtoMotions' PyRoki whole-body retargeting solver,
but its source is *already retargeted G1 RobotState motion*, not SMPL/SOMA.

Pipeline:
    G1 .motion
      -> read G1 world-space body poses from RobotState
      -> build 15 semantic target points + 3 auxiliary points
      -> PyRoki/JAXLS optimization on Astro P2 URDF
      -> Astro P2 root pose + 30 joint angles
      -> optionally save NPZ
      -> or build and save a ProtoMotions Astro P2 .motion directly using P2 MJCF

Expected source robot:
    ProtoMotions G1 29-DOF, normally g1_bm_box_feet.xml.

Important design choices:
  * No SMPL scaling is applied. Source targets are already robot-space G1 poses.
  * G1 and Astro P2 joints with the same names are used as a soft joint reference.
  * G1's retargeting toe point is synthesized at +0.15 m from ankle-roll.
  * Astro P2 has no separate foot/toe link, so a virtual toe point is synthesized
    from ankle-roll at [0.11, 0.0, 0.0] m. Both semantic toes are at ankle
    height; actual collision geometry is used separately for ground clearance.
  * Astro P2's real fixed hand links are used as hand auxiliary targets.
  * Astro P2 head_joint is not driven by G1 and is softly kept near zero.

Example:
    python retarget_g1_motion_to_astro_p2.py \
      --motion-dir /data/.../g1_motions \
      --output-dir /data/.../astro_p2_motions \
      --astro-urdf-path /path/to/astro_p2/urdf/astro_p2_30dof_primitive_collision.urdf \
      --astro-mjcf-path /path/to/astro_p2/mjcf/astro_p2_30dof_primitive_collision.xml \
      --g1-mjcf-path protomotions/data/assets/mjcf/g1_bm_box_feet.xml \
      --output-format motion \
      --chunk-frames 300 \
      --chunk-overlap 10

For a first smoke test, add:
    --max-files 1 --output-format npz
"""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
import os
import sys
from pathlib import Path
from typing import Tuple, TypedDict

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import torch
import yourdfpy

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protomotions.components.pose_lib import (  # noqa: E402
    compute_cartesian_velocity,
    extract_kinematic_info,
    extract_transforms_from_qpos,
    fk_from_transforms_with_velocities,
)
from protomotions.simulator.base_simulator.simulator_state import (  # noqa: E402
    RobotState,
    StateConversion,
)
from protomotions.utils.rotations import quaternion_to_matrix  # noqa: E402


# -----------------------------------------------------------------------------
# Semantic retarget skeleton
# -----------------------------------------------------------------------------

ASTRO_LINK_NAMES = None
MAX_SOLVER_ITERS = 500
ASTRO_VELOCITY_LIMITS = None
GROUND_GEOMS = None
HUMAN_RETARGET_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_foot",
    "right_foot",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]
N_RETARGET = len(HUMAN_RETARGET_NAMES)
N_AUX = 3  # left hand, right hand, torso

# G1's official PyRoki retarget URDF uses a fixed foot keypoint +0.15m in ankle frame.
G1_TOE_LOCAL = np.array([0.15, 0.0, 0.0], dtype=np.float32)

# Virtual toe direction in the ankle frame (same height convention as G1).
# Capsule support, including its radius, is accounted for by the ground correction.
ASTRO_LEFT_TOE_LOCAL = jnp.array([0.11, 0.0, 0.0], dtype=jnp.float32)
ASTRO_RIGHT_TOE_LOCAL = jnp.array([0.11, 0.0, 0.0], dtype=jnp.float32)

DIRECT_PAIRS = [
    ("left_shoulder", "left_elbow", 1.0),
    ("right_shoulder", "right_elbow", 1.0),
    ("left_elbow", "left_wrist", 1.0),
    ("right_elbow", "right_wrist", 1.0),
    ("left_hip", "left_knee", 1.0),
    ("right_hip", "right_knee", 1.0),
    ("left_knee", "left_ankle", 1.0),
    ("right_knee", "right_ankle", 1.0),
    ("left_ankle", "left_foot", 1.0),
    ("right_ankle", "right_foot", 1.0),
]


def get_astro_retarget_indices() -> tuple[list[str], jnp.ndarray]:
    if ASTRO_LINK_NAMES is None:
        raise RuntimeError("ASTRO_LINK_NAMES not initialized")

    # left/right foot are placeholders pointing at ankle-roll; their actual positions
    # are replaced by virtual toe points in astro_retarget_positions().
    mapping = [
        ("pelvis", "pelvis"),
        ("left_hip", "left_hip_pitch_link"),
        ("right_hip", "right_hip_pitch_link"),
        ("left_knee", "left_knee_link"),
        ("right_knee", "right_knee_link"),
        ("left_ankle", "left_ankle_roll_link"),
        ("right_ankle", "right_ankle_roll_link"),
        ("left_foot", "left_ankle_roll_link"),
        ("right_foot", "right_ankle_roll_link"),
        ("left_shoulder", "left_shoulder_pitch_link"),
        ("right_shoulder", "right_shoulder_pitch_link"),
        ("left_elbow", "left_elbow_link"),
        ("right_elbow", "right_elbow_link"),
        ("left_wrist", "left_wrist_yaw_link"),
        ("right_wrist", "right_wrist_yaw_link"),
    ]

    names: list[str] = []
    indices: list[int] = []
    for semantic_name, astro_link_name in mapping:
        names.append(semantic_name)
        try:
            indices.append(ASTRO_LINK_NAMES.index(astro_link_name))
        except ValueError as exc:
            raise ValueError(
                f"Astro link '{astro_link_name}' is missing. Available links: {ASTRO_LINK_NAMES}"
            ) from exc
    return names, jnp.array(indices, dtype=jnp.int32)


human_retarget_names: list[str] | None = None
astro_joint_retarget_indices: jnp.ndarray | None = None


# -----------------------------------------------------------------------------
# Source G1 .motion loading
# -----------------------------------------------------------------------------


def _fps_from_motion_dict(data: dict, fallback: float) -> float:
    fps = data.get("fps", fallback)
    if torch.is_tensor(fps):
        fps = fps.detach().cpu().item()
    if isinstance(fps, np.ndarray):
        fps = fps.reshape(-1)[0]
    return float(fps)


def _body_index(body_names: list[str], name: str) -> int:
    try:
        return body_names.index(name)
    except ValueError as exc:
        raise ValueError(f"G1 body '{name}' not found. body_names={body_names}") from exc


def _transform_local_point(
    body_pos: np.ndarray, body_rot: np.ndarray, local_point: np.ndarray
) -> np.ndarray:
    """Transform one fixed local point for every frame.

    body_pos: [T, 3]
    body_rot: [T, 3, 3]
    local_point: [3]
    """
    return body_pos + np.einsum("tij,j->ti", body_rot, local_point)


def _smooth_contact(x: np.ndarray, window: int = 5) -> np.ndarray:
    x = x.astype(np.float32)
    if window <= 1:
        return x[:, None]
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(x, (window // 2, window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(x), None]


def _infer_contacts(
    left_toe: np.ndarray,
    right_toe: np.ndarray,
    fps: float,
    height_margin: float = 0.055,
    speed_threshold: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer soft foot contacts from source G1 toe height and speed."""
    both_z = np.concatenate([left_toe[:, 2], right_toe[:, 2]])
    ground_z = float(np.percentile(both_z, 5.0))

    def one(foot: np.ndarray) -> np.ndarray:
        vel = np.zeros(len(foot), dtype=np.float32)
        if len(foot) > 1:
            vel[1:] = np.linalg.norm(np.diff(foot, axis=0), axis=-1) * fps
            vel[0] = vel[1]
        c = (foot[:, 2] < ground_z + height_margin) & (vel < speed_threshold)
        return _smooth_contact(c.astype(np.float32), window=5)

    return one(left_toe), one(right_toe)


def load_g1_motion_targets(
    motion_path: Path,
    g1_kinematic_info,
    astro_joint_names: list[str],
    subsample_factor: int = 1,
    fallback_fps: float = 30.0,
) -> dict:
    """Load one G1 ProtoMotions .motion and build Astro-retarget targets.

    Returns full, unpadded arrays. Chunking/padding is handled later.
    """
    data = torch.load(motion_path, map_location="cpu", weights_only=False)
    if "length_starts" in data:
        raise ValueError(
            f"{motion_path} looks like a packaged motion library, not an individual .motion file"
        )

    motion = RobotState.from_dict(data, state_conversion=StateConversion.COMMON)
    fps = _fps_from_motion_dict(data, fallback_fps)

    body_names = list(g1_kinematic_info.body_names)
    if motion.rigid_body_pos is None or motion.rigid_body_rot is None:
        raise ValueError(f"{motion_path} has no rigid body pose data")
    if motion.rigid_body_pos.shape[1] != len(body_names):
        raise ValueError(
            f"G1 body-count mismatch for {motion_path}: motion has "
            f"{motion.rigid_body_pos.shape[1]} bodies but {len(body_names)} were parsed from "
            f"the supplied G1 MJCF. Make sure --g1-mjcf-path matches the source dataset."
        )

    body_pos = motion.rigid_body_pos.detach().cpu().float()  # [T, B, 3]
    body_quat_xyzw = motion.rigid_body_rot.detach().cpu().float()  # COMMON = xyzw
    body_rot = quaternion_to_matrix(body_quat_xyzw, w_last=True)

    body_pos_np = body_pos.numpy()
    body_rot_np = body_rot.numpy()

    # Semantic G1 bodies. G1 box-feet MJCF has no separate toe body, so toe is virtual.
    idx = {name: _body_index(body_names, name) for name in [
        "pelvis",
        "left_hip_pitch_link", "right_hip_pitch_link",
        "left_knee_link", "right_knee_link",
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_shoulder_pitch_link", "right_shoulder_pitch_link",
        "left_elbow_link", "right_elbow_link",
        "left_wrist_yaw_link", "right_wrist_yaw_link",
        "torso_link",
    ]}

    left_ankle_i = idx["left_ankle_roll_link"]
    right_ankle_i = idx["right_ankle_roll_link"]
    left_toe = _transform_local_point(
        body_pos_np[:, left_ankle_i], body_rot_np[:, left_ankle_i], G1_TOE_LOCAL
    )
    right_toe = _transform_local_point(
        body_pos_np[:, right_ankle_i], body_rot_np[:, right_ankle_i], G1_TOE_LOCAL
    )

    # Use the real G1 hand body when it exists. Fall back to wrist position otherwise.
    if "left_rubber_hand" in body_names:
        left_hand = body_pos_np[:, body_names.index("left_rubber_hand")]
    else:
        left_hand = body_pos_np[:, idx["left_wrist_yaw_link"]]
    if "right_rubber_hand" in body_names:
        right_hand = body_pos_np[:, body_names.index("right_rubber_hand")]
    else:
        right_hand = body_pos_np[:, idx["right_wrist_yaw_link"]]

    target_keypoints = np.stack(
        [
            body_pos_np[:, idx["pelvis"]],
            body_pos_np[:, idx["left_hip_pitch_link"]],
            body_pos_np[:, idx["right_hip_pitch_link"]],
            body_pos_np[:, idx["left_knee_link"]],
            body_pos_np[:, idx["right_knee_link"]],
            body_pos_np[:, left_ankle_i],
            body_pos_np[:, right_ankle_i],
            left_toe,
            right_toe,
            body_pos_np[:, idx["left_shoulder_pitch_link"]],
            body_pos_np[:, idx["right_shoulder_pitch_link"]],
            body_pos_np[:, idx["left_elbow_link"]],
            body_pos_np[:, idx["right_elbow_link"]],
            body_pos_np[:, idx["left_wrist_yaw_link"]],
            body_pos_np[:, idx["right_wrist_yaw_link"]],
            left_hand,
            right_hand,
            (0.5 * (body_pos_np[:, idx["left_shoulder_pitch_link"]] +
                    body_pos_np[:, idx["right_shoulder_pitch_link"]]) +
             np.einsum("tij,j->ti", body_rot_np[:, idx["torso_link"]],
                       np.array([0.15, 0., 0.], dtype=np.float32))),
        ],
        axis=1,
    ).astype(np.float32)  # [T, 18, 3]

    # Solver only uses target_orientations[:, 0] for root initialization; populate all
    # positions anyway so the representation stays compatible with the original script.
    T = target_keypoints.shape[0]
    target_orientations = np.tile(np.eye(3, dtype=np.float32), (T, N_RETARGET + N_AUX, 1, 1))
    target_orientations[:, 0] = body_rot_np[:, idx["pelvis"]]

    # Preserve supplied contacts, including all-airborne clips; infer only if absent.
    use_stored_contacts = False
    if motion.rigid_body_contacts is not None:
        contacts = motion.rigid_body_contacts.detach().cpu().numpy().astype(bool)
        if contacts.shape[:2] == body_pos_np.shape[:2]:
            left_contact = _smooth_contact(contacts[:, left_ankle_i].astype(np.float32))
            right_contact = _smooth_contact(contacts[:, right_ankle_i].astype(np.float32))
            use_stored_contacts = True
    if not use_stored_contacts:
        left_contact, right_contact = _infer_contacts(left_toe, right_toe, fps)

    # Soft joint reference: copy all shared G1 joint names into Astro P2 order.
    if motion.dof_pos is None:
        raise ValueError(f"{motion_path} has no dof_pos")
    g1_dof = motion.dof_pos.detach().cpu().numpy().astype(np.float32)
    g1_dof_names = list(g1_kinematic_info.dof_names)
    if g1_dof.shape[1] != len(g1_dof_names):
        raise ValueError(
            f"G1 DOF-count mismatch: motion has {g1_dof.shape[1]}, MJCF has {len(g1_dof_names)}"
        )

    astro_ref = np.zeros((T, len(astro_joint_names)), dtype=np.float32)
    g1_name_to_i = {n: i for i, n in enumerate(g1_dof_names)}
    for j, name in enumerate(astro_joint_names):
        if name in g1_name_to_i:
            astro_ref[:, j] = g1_dof[:, g1_name_to_i[name]]
        # head_joint is absent in G1 and remains zero.

    sf = max(int(subsample_factor), 1)
    if sf > 1:
        target_keypoints = target_keypoints[::sf]
        target_orientations = target_orientations[::sf]
        left_contact = left_contact[::sf]
        right_contact = right_contact[::sf]
        astro_ref = astro_ref[::sf]
        fps = fps / sf

    return {
        "target_keypoints": target_keypoints,
        "target_orientations": target_orientations,
        "left_contact": left_contact.astype(np.float32),
        "right_contact": right_contact.astype(np.float32),
        "astro_joint_reference": astro_ref,
        "fps": float(fps),
        "used_stored_contacts": use_stored_contacts,
    }


# -----------------------------------------------------------------------------
# Costs and solver
# -----------------------------------------------------------------------------


class RetargetingWeights(TypedDict):
    local_alignment: float
    global_alignment: float
    root_smoothness: float
    joint_smoothness: float
    self_collision: float
    joint_rest_penalty: float
    joint_vel_limit: float
    foot_contact: float
    foot_tilt: float
    joint_reference: float


def astro_retarget_positions(
    T_world_link: jaxlie.SE3,
    astro_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Return the 15 semantic Astro points, replacing foot placeholders by virtual toes."""
    assert human_retarget_names is not None
    assert ASTRO_LINK_NAMES is not None

    pos = T_world_link.translation()
    rot = T_world_link.rotation().as_matrix()
    out = pos[astro_indices]

    left_ankle_link_i = ASTRO_LINK_NAMES.index("left_ankle_roll_link")
    right_ankle_link_i = ASTRO_LINK_NAMES.index("right_ankle_roll_link")
    left_foot_sem_i = human_retarget_names.index("left_foot")
    right_foot_sem_i = human_retarget_names.index("right_foot")

    left_toe = pos[left_ankle_link_i] + rot[left_ankle_link_i] @ ASTRO_LEFT_TOE_LOCAL
    right_toe = pos[right_ankle_link_i] + rot[right_ankle_link_i] @ ASTRO_RIGHT_TOE_LOCAL
    out = out.at[left_foot_sem_i].set(left_toe)
    out = out.at[right_foot_sem_i].set(right_toe)
    return out


@jaxls.Cost.create_factory
def joint_vel_limit_cost(
    var_values: jaxls.VarValues,
    var_joints_curr: jaxls.Var[jnp.ndarray],
    var_joints_prev: jaxls.Var[jnp.ndarray],
    max_vel: float,
    dt: float,
    weight: float,
) -> jax.Array:
    joints_curr = var_values[var_joints_curr]
    joints_prev = var_values[var_joints_prev]
    joint_vel = (joints_curr - joints_prev) / dt
    excess_vel = jnp.maximum(jnp.abs(joint_vel) - max_vel, 0.0)
    return excess_vel.flatten() * weight


@jaxls.Cost.create_factory
def joint_reference_cost(
    var_values: jaxls.VarValues,
    var_joints: jaxls.Var[jnp.ndarray],
    reference: jnp.ndarray,
    weight: float,
) -> jax.Array:
    """Softly keep Astro near the corresponding G1 joint motion.

    This is intentionally soft: P2 geometry and waist-chain order differ from G1,
    so Cartesian alignment is allowed to override the copied joint angles.
    """
    return (var_values[var_joints] - reference).flatten() * weight


@jaxls.Cost.create_factory
def foot_contact_cost(
    var_values: jaxls.VarValues,
    var_Ts_world_root_curr: jaxls.SE3Var,
    var_Ts_world_root_prev: jaxls.SE3Var,
    var_robot_cfg_curr: jaxls.Var[jnp.ndarray],
    var_robot_cfg_prev: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    weight: float,
) -> jax.Array:
    assert ASTRO_LINK_NAMES is not None

    T_world_root_curr = var_values[var_Ts_world_root_curr]
    T_world_root_prev = var_values[var_Ts_world_root_prev]
    robot_cfg_curr = var_values[var_robot_cfg_curr]
    robot_cfg_prev = var_values[var_robot_cfg_prev]

    T_world_link_curr = T_world_root_curr @ jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg_curr, unroll_fk=True))
    T_world_link_prev = T_world_root_prev @ jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg_prev, unroll_fk=True))

    pos_c = T_world_link_curr.translation()
    pos_p = T_world_link_prev.translation()
    rot_c = T_world_link_curr.rotation().as_matrix()
    rot_p = T_world_link_prev.rotation().as_matrix()

    li = ASTRO_LINK_NAMES.index("left_ankle_roll_link")
    ri = ASTRO_LINK_NAMES.index("right_ankle_roll_link")

    left_ankle_c = pos_c[li]
    right_ankle_c = pos_c[ri]
    left_ankle_p = pos_p[li]
    right_ankle_p = pos_p[ri]

    left_toe_c = left_ankle_c + rot_c[li] @ ASTRO_LEFT_TOE_LOCAL
    right_toe_c = right_ankle_c + rot_c[ri] @ ASTRO_RIGHT_TOE_LOCAL
    left_toe_p = left_ankle_p + rot_p[li] @ ASTRO_LEFT_TOE_LOCAL
    right_toe_p = right_ankle_p + rot_p[ri] @ ASTRO_RIGHT_TOE_LOCAL

    lc = left_foot_contact[0]
    rc = right_foot_contact[0]

    # Frame-to-frame displacement; original PyRoki implementation uses displacement,
    # not divided by dt, for this contact pinning term.
    left_ankle_vel_cost = lc * (left_ankle_c - left_ankle_p)
    right_ankle_vel_cost = rc * (right_ankle_c - right_ankle_p)
    left_toe_vel_cost = lc * (left_toe_c - left_toe_p)
    right_toe_vel_cost = rc * (right_toe_c - right_toe_p)

    # For a flat foot both semantic points lie at ankle-frame height.
    expected_z_delta = -ASTRO_LEFT_TOE_LOCAL[2]
    left_z_cost = lc * ((left_ankle_c[2] - left_toe_c[2]) - expected_z_delta)
    right_z_cost = rc * ((right_ankle_c[2] - right_toe_c[2]) - expected_z_delta)

    return jnp.concatenate(
        [
            left_ankle_vel_cost.flatten(),
            right_ankle_vel_cost.flatten(),
            left_toe_vel_cost.flatten(),
            right_toe_vel_cost.flatten(),
            jnp.array([left_z_cost]),
            jnp.array([right_z_cost]),
        ]
    ) * weight


@jaxls.Cost.create_factory
def foot_tilt_cost(
    var_values: jaxls.VarValues,
    var_Ts_world_root: jaxls.SE3Var,
    var_robot_cfg: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    weight: float,
) -> jax.Array:
    assert ASTRO_LINK_NAMES is not None
    T_world_root = var_values[var_Ts_world_root]
    robot_cfg = var_values[var_robot_cfg]
    T_world_link = T_world_root @ jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg, unroll_fk=True))
    rot = T_world_link.rotation().as_matrix()

    li = ASTRO_LINK_NAMES.index("left_ankle_roll_link")
    ri = ASTRO_LINK_NAMES.index("right_ankle_roll_link")
    left_tilt = left_foot_contact[0] * (rot[li, 2, 2] - 1.0)
    right_tilt = right_foot_contact[0] * (rot[ri, 2, 2] - 1.0)
    return jnp.array([left_tilt, right_tilt]) * weight


@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision | None,
    target_keypoints: jnp.ndarray,
    target_orientations: jnp.ndarray,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    astro_joint_reference: jnp.ndarray,
    astro_indices: jnp.ndarray,
    astro_retarget_mask: jnp.ndarray,
    weights: RetargetingWeights,
    input_fps: float = 30.0,
) -> Tuple[jaxlie.SE3, jnp.ndarray]:
    assert human_retarget_names is not None
    assert ASTRO_LINK_NAMES is not None

    n_retarget = len(astro_indices)
    timesteps = target_keypoints.shape[0]

    joints_to_move_less = jnp.array(
        [
            robot.joints.actuated_names.index(name)
            for name in [
                "waist_roll_joint",
                "right_wrist_pitch_joint",
                "left_wrist_pitch_joint",
                "head_joint",
            ]
            if name in robot.joints.actuated_names
        ],
        dtype=jnp.int32,
    )

    class SimplifiedJointsScaleVarAstro(
        jaxls.Var[jax.Array], default_factory=lambda: jnp.ones((len(DIRECT_PAIRS),))
    ):
        pass

    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    var_Ts_world_root = jaxls.SE3Var(jnp.arange(timesteps))
    var_joints_scale = SimplifiedJointsScaleVarAstro(jnp.zeros(1, dtype=jnp.int32))

    root_init_values = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_matrix(target_orientations[:, 0]), target_keypoints[:, 0])

    costs: list[jaxls.Cost] = []

    @jaxls.Cost.create_factory
    def retargeting_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_joints_scale: SimplifiedJointsScaleVarAstro,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        robot_cfg = var_values[var_robot_cfg]
        T_world_link = var_values[var_Ts_world_root] @ jaxlie.SE3(
            robot.forward_kinematics(cfg=robot_cfg, unroll_fk=True)
        )
        target_pos = keypoints[:N_RETARGET]
        robot_pos = astro_retarget_positions(T_world_link, astro_indices)

        # The original 15x15 residual matrix has only 20 nonzero directed
        # edges. Evaluate exactly those edges, retaining both directions.
        pairs = [(human_retarget_names.index(a), human_retarget_names.index(b), w)
                 for a,b,w in DIRECT_PAIRS]
        ii = jnp.array([i for a,b,w in pairs for i in (a,b)])
        jj = jnp.array([i for a,b,w in pairs for i in (b,a)])
        edge_weight = jnp.array([w for a,b,w in pairs for _ in range(2)])
        dt = target_pos[ii] - target_pos[jj]
        dr = robot_pos[ii] - robot_pos[jj]
        scale = jnp.repeat(var_values[var_joints_scale], 2)[:,None]
        residual_pos = (dt - dr*scale) * edge_weight[:,None]
        dt_norm = dt / jnp.linalg.norm(dt+1e-6,axis=-1,keepdims=True)
        dr_norm = dr / jnp.linalg.norm(dr+1e-6,axis=-1,keepdims=True)
        residual_ang = (1-(dt_norm*dr_norm).sum(axis=-1))*edge_weight
        return jnp.concatenate([residual_pos.flatten(),residual_ang]) * weights["local_alignment"]

    @jaxls.Cost.create_factory
    def scale_regularization(
        var_values: jaxls.VarValues,
        var_joints_scale: SimplifiedJointsScaleVarAstro,
    ) -> jax.Array:
        s = var_values[var_joints_scale]
        # Original full-matrix penalty repeats once per frame and counts both
        # directions. sqrt(2*T) preserves that strength for one shared scale vector.
        return jnp.concatenate([(s-1.0).flatten(), jnp.clip(-s,min=0).flatten()*100.0]) * jnp.sqrt(2.0*timesteps)

    @jaxls.Cost.create_factory
    def pc_alignment_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        T_world_link = var_values[var_Ts_world_root] @ jaxlie.SE3(
            robot.forward_kinematics(cfg=var_values[var_robot_cfg], unroll_fk=True)
        )
        link_pos = astro_retarget_positions(T_world_link, astro_indices)

        left_hand = T_world_link.translation()[ASTRO_LINK_NAMES.index("left_hand_link")]
        right_hand = T_world_link.translation()[ASTRO_LINK_NAMES.index("right_hand_link")]
        torso = (0.5 * (T_world_link.translation()[ASTRO_LINK_NAMES.index("left_shoulder_pitch_link")] +
                        T_world_link.translation()[ASTRO_LINK_NAMES.index("right_shoulder_pitch_link")]) +
                 T_world_link.rotation().as_matrix()[ASTRO_LINK_NAMES.index("waist_roll_link")] @ jnp.array([0.15, 0., 0.]))
        link_pos_with_aux = jnp.concatenate(
            [link_pos, left_hand[None], right_hand[None], torso[None]], axis=0
        )

        keypoint_pos = keypoints

        # Downweight both hand auxiliary points and elbows, following the intent of
        # ProtoMotions' original solver, but without G1-specific hand offsets.
        for i in (-3, -2, -7, -6):
            keypoint_pos = keypoint_pos.at[i].set(keypoint_pos[i] / 4.0)
            link_pos_with_aux = link_pos_with_aux.at[i].set(link_pos_with_aux[i] / 4.0)

        return (link_pos_with_aux - keypoint_pos).flatten() * weights["global_alignment"]

    @jaxls.Cost.create_factory
    def root_smoothness(
        var_values: jaxls.VarValues,
        curr: jaxls.SE3Var,
        prev: jaxls.SE3Var,
    ) -> jax.Array:
        return (var_values[curr].inverse() @ var_values[prev]).log().flatten() * weights[
            "root_smoothness"
        ]

    costs.extend(
        [
            retargeting_cost(var_Ts_world_root, var_joints, var_joints_scale, target_keypoints),
            scale_regularization(var_joints_scale),
            pk.costs.limit_cost(jax.tree.map(lambda x: x[None], robot), var_joints, 100.0),
            pc_alignment_cost(var_Ts_world_root, var_joints, target_keypoints),
            joint_reference_cost(
                var_joints,
                astro_joint_reference,
                weights["joint_reference"],
            ),
        ]
    )

    if timesteps > 1:
        costs.extend(
            [
                pk.costs.smoothness_cost(
                    robot.joint_var_cls(jnp.arange(1, timesteps)),
                    robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                    weights["joint_smoothness"],
                ),
                root_smoothness(
                    jaxls.SE3Var(jnp.arange(1, timesteps)),
                    jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
                ),
                joint_vel_limit_cost(
                    robot.joint_var_cls(jnp.arange(1, timesteps)),
                    robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                    ASTRO_VELOCITY_LIMITS[None],
                    1.0 / input_fps,
                    weights["joint_vel_limit"],
                ),
            ]
        )

    # Rest cost: light globally, stronger on wrists/waist/head.
    rest_weights = jnp.full(var_joints.default_factory().shape, 0.02)
    if joints_to_move_less.size > 0:
        rest_weights = rest_weights.at[joints_to_move_less].set(weights["joint_rest_penalty"])
    costs.append(
        pk.costs.rest_cost(
            var_joints,
            var_joints.default_factory()[None],
            rest_weights[None],
        )
    )

    # Batch factors; a Python factor per frame makes compilation prohibitively slow.
    if timesteps > 1:
        costs.append(foot_contact_cost(
            jaxls.SE3Var(jnp.arange(1, timesteps)), jaxls.SE3Var(jnp.arange(timesteps-1)),
            robot.joint_var_cls(jnp.arange(1, timesteps)), robot.joint_var_cls(jnp.arange(timesteps-1)),
            jax.tree.map(lambda x: x[None], robot), left_foot_contact[1:], right_foot_contact[1:], weights["foot_contact"]))
    costs.append(foot_tilt_cost(
        var_Ts_world_root, var_joints, jax.tree.map(lambda x: x[None], robot),
        left_foot_contact, right_foot_contact, weights["foot_tilt"]))

    solution, summary = (
        jaxls.LeastSquaresProblem(costs, [var_joints, var_Ts_world_root, var_joints_scale])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make(
                [
                    var_joints.with_value(jnp.clip(astro_joint_reference, robot.joints.lower_limits, robot.joints.upper_limits)),
                    var_Ts_world_root.with_value(root_init_values),
                    var_joints_scale,
                ]
            ),
            termination=jaxls.TerminationConfig(max_iterations=MAX_SOLVER_ITERS),
            verbose=False, return_summary=True,
        )
    )
    return solution[var_Ts_world_root], solution[var_joints], summary


# -----------------------------------------------------------------------------
# Chunking / output
# -----------------------------------------------------------------------------


def _pad_last(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) >= target_len:
        return x[:target_len]
    if len(x) == 0:
        raise ValueError("cannot pad an empty motion")
    pad = np.repeat(x[-1:], target_len - len(x), axis=0)
    return np.concatenate([x, pad], axis=0)


def solve_full_motion(
    robot: pk.Robot,
    arrays: dict,
    astro_indices: jnp.ndarray,
    astro_mask: jnp.ndarray,
    weights: RetargetingWeights,
    chunk_frames: int,
    chunk_overlap: int,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    keypoints = arrays["target_keypoints"]
    orientations = arrays["target_orientations"]
    lc = arrays["left_contact"]
    rc = arrays["right_contact"]
    ref = arrays["astro_joint_reference"]
    fps = arrays["fps"]
    T = len(keypoints)

    if T < 2:
        raise ValueError("motion must contain at least 2 frames")

    if chunk_frames <= 0 or chunk_frames >= T:
        chunks = [(0, T, chunk_frames if chunk_frames > 0 else T)]
    else:
        if chunk_overlap < 0 or chunk_overlap >= chunk_frames:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_frames")
        step = chunk_frames - chunk_overlap
        chunks = []
        start = 0
        while start < T:
            end = min(T, start + chunk_frames)
            chunks.append((start, end, chunk_frames))
            if end == T:
                break
            start += step

    roots_out: list[np.ndarray] = []
    joints_out: list[np.ndarray] = []

    for ci, (start, end, padded_len) in enumerate(chunks):
        valid = end - start
        print(f"    chunk {ci+1}/{len(chunks)}: frames [{start}:{end}) valid={valid}")

        kp = _pad_last(keypoints[start:end], padded_len)
        ori = _pad_last(orientations[start:end], padded_len)
        lcc = _pad_last(lc[start:end], padded_len)
        rcc = _pad_last(rc[start:end], padded_len)
        rr = _pad_last(ref[start:end], padded_len)

        solve_start = time.monotonic()
        Ts, q, summary = solve_retargeting(
            robot=robot,
            robot_coll=None,
            target_keypoints=jnp.asarray(kp),
            target_orientations=jnp.asarray(ori),
            left_foot_contact=jnp.asarray(lcc),
            right_foot_contact=jnp.asarray(rcc),
            astro_joint_reference=jnp.asarray(rr),
            astro_indices=astro_indices,
            astro_retarget_mask=astro_mask,
            weights=weights,
            input_fps=float(fps),
        )

        history = np.asarray(summary.cost_history)
        iterations = int(summary.iterations)
        final_cost = float(history[min(iterations, len(history)-1)])
        if not np.isfinite(final_cost) or final_cost > float(history[0]) * 1.001 + 1e-5:
            raise ValueError("Optimization did not produce a finite non-increasing cost")
        print(f"      iterations={iterations} cost={float(history[0]):.4f}->{final_cost:.4f} solve_seconds={time.monotonic()-solve_start:.3f}", flush=True)
        root_wxyz_xyz = np.asarray(Ts.wxyz_xyz[:valid])
        joint_np = np.asarray(q[:valid])

        # Cross-fade in the overlap, including shortest-path quaternion slerp.
        # Unlike dropping the overlap, both endpoints remain continuous.
        if ci > 0 and chunk_overlap:
            from scipy.spatial.transform import Rotation, Slerp
            n = min(chunk_overlap, len(joint_np), len(joints_out[-1]))
            a = (np.arange(n, dtype=np.float32) + 1) / (n + 1)
            old_r = roots_out[-1][-n:].copy()
            new_r = root_wxyz_xyz[:n].copy()
            blends = np.empty_like(old_r[:, :4])
            for k in range(n):
                qs = np.stack([old_r[k, :4], new_r[k, :4]])[:, [1,2,3,0]]
                blends[k] = Slerp([0.,1.], Rotation.from_quat(qs))([a[k]]).as_quat()[0, [3,0,1,2]]
            roots_out[-1][-n:, :4] = blends
            roots_out[-1][-n:, 4:] = old_r[:, 4:] * (1-a[:,None]) + new_r[:,4:] * a[:,None]
            joints_out[-1][-n:] = joints_out[-1][-n:] * (1-a[:,None]) + joint_np[:n] * a[:,None]
        drop = chunk_overlap if ci > 0 else 0
        roots_out.append(root_wxyz_xyz[drop:].copy())
        joints_out.append(joint_np[drop:].copy())

    roots = np.concatenate(roots_out, axis=0)[:T]
    joints = np.concatenate(joints_out, axis=0)[:T]
    if len(roots) != T:
        raise RuntimeError(f"stitching error: expected {T} frames, got {len(roots)}")
    if not np.isfinite(roots).all() or not np.isfinite(joints).all():
        raise ValueError("Nonfinite optimization result")
    # The optimizer's soft limits may leave small violations; enforce the URDF bounds.
    joints = np.clip(joints, np.asarray(robot.joints.lower_limits), np.asarray(robot.joints.upper_limits))
    return roots, joints


def save_npz(
    outpath: Path,
    root_wxyz_xyz: np.ndarray,
    joints_solver_order: np.ndarray,
    fps: float,
    solver_joint_names: list[str],
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outpath,
        base_frame_pos=root_wxyz_xyz[:, 4:],
        base_frame_wxyz=root_wxyz_xyz[:, :4],
        joint_angles=joints_solver_order,
        joint_names=np.asarray(solver_joint_names),
        fps=np.float32(fps),
    )


def save_proto_motion(
    outpath: Path,
    root_wxyz_xyz: np.ndarray,
    joints_solver_order: np.ndarray,
    fps: float,
    solver_joint_names: list[str],
    p2_kinematic_info,
    left_contact: np.ndarray,
    right_contact: np.ndarray,
    fix_height: bool,
) -> None:
    """Build a standard ProtoMotions RobotState dict and torch.save it as .motion."""
    p2_names = list(p2_kinematic_info.dof_names)
    missing = [n for n in p2_names if n not in solver_joint_names]
    if missing:
        raise ValueError(f"P2 MJCF has DOFs missing from PyRoki URDF: {missing}")
    reorder = [solver_joint_names.index(n) for n in p2_names]
    q_mjcf = joints_solver_order[:, reorder]

    root_pos = torch.from_numpy(root_wxyz_xyz[:, 4:]).float()
    root_wxyz = torch.from_numpy(root_wxyz_xyz[:, :4]).float()
    joint_angles = torch.from_numpy(q_mjcf).float()
    qpos = torch.cat([root_pos, root_wxyz, joint_angles], dim=-1)

    root_pos_fk, joint_rot_mats = extract_transforms_from_qpos(p2_kinematic_info, qpos)
    motion = fk_from_transforms_with_velocities(
        kinematic_info=p2_kinematic_info,
        root_pos=root_pos_fk,
        joint_rot_mats=joint_rot_mats,
        fps=float(fps),
        compute_velocities=True,
        velocity_max_horizon=3,
    )
    motion.dof_pos = joint_angles
    motion.dof_vel = compute_cartesian_velocity(
        batched_robot_pos=joint_angles.unsqueeze(1), fps=float(fps)
    ).squeeze(1)

    # Transfer soft source contact labels to Astro ankle-roll bodies.
    contacts = torch.zeros(
        motion.rigid_body_pos.shape[0],
        motion.rigid_body_pos.shape[1],
        dtype=torch.bool,
    )
    body_names = list(p2_kinematic_info.body_names)
    li = body_names.index("left_ankle_roll_link")
    ri = body_names.index("right_ankle_roll_link")
    contacts[:, li] = torch.from_numpy(left_contact[:, 0] > 0.5)
    contacts[:, ri] = torch.from_numpy(right_contact[:, 0] > 0.5)
    motion.rigid_body_contacts = contacts

    if fix_height:
        # A single translation for the entire motion preserves jumps and velocities.
        # Use primitive collision support, not link origins, to avoid foot penetration.
        shift = max(0.0, 0.002 - minimum_collision_height(motion, p2_kinematic_info))
        motion.rigid_body_pos[:, :, 2] += shift

    # Same compatibility choice as ProtoMotions' PyRoki converter.
    motion.local_rigid_body_rot = None

    outpath.parent.mkdir(parents=True, exist_ok=True)
    for key in ("rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel", "dof_pos", "dof_vel"):
        if not torch.isfinite(getattr(motion, key)).all():
            raise ValueError(f"Invalid output: {key}")
    tmp = outpath.with_suffix(".partial")
    torch.save(motion.to_dict(), tmp)
    os.replace(tmp, outpath)


def load_collision_support(path):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(path))
    result = []
    for i in range(model.ngeom):
        if not (model.geom_contype[i] or model.geom_conaffinity[i]): continue
        result.append((model.body(int(model.geom_bodyid[i])).name,
                       int(model.geom_type[i]), model.geom_pos[i].copy(),
                       model.geom_quat[i].copy(), model.geom_size[i].copy()))
    return result


def minimum_collision_height(motion, kin):
    from scipy.spatial.transform import Rotation
    pos = motion.rigid_body_pos.numpy()
    rot = Rotation.from_quat(motion.rigid_body_rot.numpy().reshape(-1,4)).as_matrix().reshape(*pos.shape[:2],3,3)
    minimum = np.inf
    for name, typ, gp, gq, sz in GROUND_GEOMS:
        i = kin.body_names.index(name)
        center = pos[:, i] + np.einsum('tij,j->ti', rot[:,i], gp)
        axes = rot[:,i] @ Rotation.from_quat(gq[[1,2,3,0]]).as_matrix()
        if typ == 2: support = sz[0]  # sphere
        elif typ == 3: support = sz[0] + sz[1]*np.abs(axes[:,2,2])  # capsule
        elif typ == 5: support = sz[0]*np.sqrt(np.maximum(0.,1.-axes[:,2,2]**2)) + sz[1]*np.abs(axes[:,2,2])
        elif typ == 6: support = np.abs(axes[:,2,:]) @ sz
        else: raise ValueError(f'Unsupported collision type {typ}')
        minimum = min(minimum, float(np.min(center[:,2]-support)))
    return minimum


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Retarget ProtoMotions G1 .motion dataset to Astro P2"
    )
    p.add_argument("--motion-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--g1-mjcf-path",
        type=Path,
        default=Path("protomotions/data/assets/mjcf/g1_bm_box_feet.xml"),
    )
    p.add_argument("--astro-urdf-path", type=Path, required=True)
    p.add_argument("--astro-mjcf-path", type=Path, required=True)
    p.add_argument("--astro-mesh-dir", type=Path, default=None)
    p.add_argument("--output-format", choices=["motion", "npz"], default="motion")
    p.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--max-files", type=int, default=0, help="0 means all files")
    p.add_argument("--subsample-factor", type=int, default=1)
    p.add_argument("--fallback-fps", type=float, default=30.0)
    p.add_argument("--chunk-frames", type=int, default=300)
    p.add_argument("--chunk-overlap", type=int, default=10)
    p.add_argument("--max-iterations", type=int, default=500)
    p.add_argument("--fix-height", action="store_true")
    p.add_argument("--file-list", type=Path)
    p.add_argument("--manifest", type=Path)

    # Solver weights. Defaults are conservative for G1 -> P2 robot-to-robot transfer.
    p.add_argument("--w-local", type=float, default=1.0)
    p.add_argument("--w-global", type=float, default=4.0)
    p.add_argument("--w-root-smooth", type=float, default=1.0)
    p.add_argument("--w-joint-smooth", type=float, default=4.0)
    p.add_argument("--w-rest", type=float, default=0.5)
    p.add_argument("--w-vel-limit", type=float, default=50.0)
    p.add_argument("--w-foot-contact", type=float, default=30.0)
    p.add_argument("--w-foot-tilt", type=float, default=1.0)
    p.add_argument("--w-joint-reference", type=float, default=0.35)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    torch.set_num_threads(1)
    print("JAX backend:", jax.default_backend(), jax.devices(), flush=True)
    jax.config.update("jax_compilation_cache_dir", str(Path.home()/".cache/astro_p2_retarget_jax"))

    for path_name in ("motion_dir", "g1_mjcf_path", "astro_urdf_path", "astro_mjcf_path"):
        path = getattr(args, path_name.replace("_", "-") if False else path_name)
        if not path.exists():
            raise FileNotFoundError(f"{path_name}: {path}")

    # Load G1 body/DOF ordering for interpreting source .motion tensors.
    g1_kin = extract_kinematic_info(str(args.g1_mjcf_path))
    print(f"G1 source: bodies={g1_kin.num_bodies}, dofs={g1_kin.num_dofs}")

    # Load Astro URDF for PyRoki optimization.
    mesh_dir = args.astro_mesh_dir
    if mesh_dir is None:
        mesh_dir = args.astro_urdf_path.parent.parent / "meshes"
    urdf = yourdfpy.URDF.load(str(args.astro_urdf_path), mesh_dir=str(mesh_dir), load_meshes=False, build_scene_graph=True)
    robot = pk.Robot.from_urdf(urdf)

    global ASTRO_LINK_NAMES, human_retarget_names, astro_joint_retarget_indices, MAX_SOLVER_ITERS, ASTRO_VELOCITY_LIMITS, GROUND_GEOMS
    ASTRO_LINK_NAMES = list(robot.links.names)
    MAX_SOLVER_ITERS = int(args.max_iterations)
    human_retarget_names, astro_joint_retarget_indices = get_astro_retarget_indices()
    solver_joint_names = list(robot.joints.actuated_names)
    limits = {j.get("name"): float(j.find("limit").get("velocity"))
              for j in ET.parse(args.astro_urdf_path).findall("joint") if j.get("type") != "fixed"}
    ASTRO_VELOCITY_LIMITS = jnp.array([limits[n] for n in solver_joint_names])
    GROUND_GEOMS = load_collision_support(args.astro_mjcf_path)

    print(f"Astro PyRoki: links={len(ASTRO_LINK_NAMES)}, actuated_dofs={len(solver_joint_names)}")
    print("Astro joint order:")
    for i, n in enumerate(solver_joint_names):
        print(f"  {i:02d}: {n}")
    if len(solver_joint_names) != 30:
        raise ValueError(
            f"Expected Astro P2 to expose 30 actuated joints, got {len(solver_joint_names)}"
        )

    # P2 MJCF is only needed for direct .motion output, but validate it up front.
    p2_kin = extract_kinematic_info(str(args.astro_mjcf_path))
    print(f"Astro MJCF: bodies={p2_kin.num_bodies}, dofs={p2_kin.num_dofs}")
    if p2_kin.num_dofs != 30:
        raise ValueError(f"Expected 30 P2 MJCF DOFs, got {p2_kin.num_dofs}")

    n = len(astro_joint_retarget_indices)
    astro_mask = jnp.zeros((n, n), dtype=jnp.float32)
    for a, b, w in DIRECT_PAIRS:
        ia = human_retarget_names.index(a)
        ib = human_retarget_names.index(b)
        astro_mask = astro_mask.at[ia, ib].set(w)
        astro_mask = astro_mask.at[ib, ia].set(w)

    weights = RetargetingWeights(
        local_alignment=args.w_local,
        global_alignment=args.w_global,
        root_smoothness=args.w_root_smooth,
        joint_smoothness=args.w_joint_smooth,
        self_collision=0.0,
        joint_rest_penalty=args.w_rest,
        joint_vel_limit=args.w_vel_limit,
        foot_contact=args.w_foot_contact,
        foot_tilt=args.w_foot_tilt,
        joint_reference=args.w_joint_reference,
    )

    pattern = "**/*.motion" if args.recursive else "*.motion"
    motion_files = ([args.motion_dir / name for name in json.loads(args.file_list.read_text())]
                    if args.file_list else sorted(args.motion_dir.glob(pattern)))
    if args.max_files > 0:
        motion_files = motion_files[: args.max_files]
    if not motion_files:
        raise FileNotFoundError(f"No .motion files found in {args.motion_dir}")

    print(f"Found {len(motion_files)} G1 motions")
    failures: list[tuple[Path, str]] = []

    for k, motion_path in enumerate(motion_files, 1):
        rel = motion_path.relative_to(args.motion_dir)
        suffix = ".motion" if args.output_format == "motion" else ".npz"
        outpath = (args.output_dir / rel).with_suffix(suffix)
        if args.skip_existing and outpath.exists():
            print(f"[{k}/{len(motion_files)}] skip: {rel}")
            continue

        print(f"[{k}/{len(motion_files)}] {rel}")
        started = time.monotonic()
        try:
            arrays = load_g1_motion_targets(
                motion_path=motion_path,
                g1_kinematic_info=g1_kin,
                astro_joint_names=solver_joint_names,
                subsample_factor=args.subsample_factor,
                fallback_fps=args.fallback_fps,
            )
            print(
                f"    frames={len(arrays['target_keypoints'])}, fps={arrays['fps']:.3f}, "
                f"contacts={'stored' if arrays['used_stored_contacts'] else 'inferred'}"
            )

            roots, joints = solve_full_motion(
                robot=robot,
                arrays=arrays,
                astro_indices=astro_joint_retarget_indices,
                astro_mask=astro_mask,
                weights=weights,
                chunk_frames=args.chunk_frames,
                chunk_overlap=args.chunk_overlap,
                max_iterations=args.max_iterations,
            )

            if args.output_format == "npz":
                save_npz(outpath, roots, joints, arrays["fps"], solver_joint_names)
            else:
                save_proto_motion(
                    outpath=outpath,
                    root_wxyz_xyz=roots,
                    joints_solver_order=joints,
                    fps=arrays["fps"],
                    solver_joint_names=solver_joint_names,
                    p2_kinematic_info=p2_kin,
                    left_contact=arrays["left_contact"],
                    right_contact=arrays["right_contact"],
                    fix_height=args.fix_height,
                )
            record = dict(file=str(rel), frames=len(roots), fps=arrays["fps"], seconds=time.monotonic()-started,
                          status="ok", output=str(outpath), stored_contacts=arrays["used_stored_contacts"])
            if args.manifest:
                with args.manifest.open("a") as f: f.write(json.dumps(record)+"\n")
            print(f"    saved -> {outpath} ({record['seconds']:.2f}s)", flush=True)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            import traceback
            traceback.print_exc()
            failures.append((motion_path, repr(exc)))
            if args.manifest:
                with args.manifest.open("a") as f: f.write(json.dumps(dict(file=str(rel), status="error", error=repr(exc)))+"\n")

    print("\nDone.")
    print(f"success={len(motion_files) - len(failures)}, failures={len(failures)}")
    if failures:
        print("Failed files:")
        for p, err in failures:
            print(f"  {p}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
