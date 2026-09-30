"""User-confirmed P2 URDF neutral pose, shared by control and PyRoki.

No Torch/JAX dependencies: usable in either isolated Python environment.
These are joint positions, NOT physical limits or the calibration T-pose.
"""

P2_DEFAULT_DOF_POS = {
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "waist_roll_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "head_joint": 0.0,
}


def p2_default_joint_positions(joint_names):
    """Resolve by NAME, never assume PyRoki and MJCF traversal orders agree."""
    names = list(joint_names)
    if len(names) != len(set(names)) or set(names) != set(P2_DEFAULT_DOF_POS):
        raise ValueError("P2 model must have exactly the 30 known actuated joints")
    return [P2_DEFAULT_DOF_POS[name] for name in names]
