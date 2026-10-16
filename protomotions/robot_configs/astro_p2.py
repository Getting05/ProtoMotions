"""Astro P2 (30 DOF), primitive collisions.

Limits: supplied P2 URDF. PD/armature: user-supplied P2 motor parameters
(2026-10-16), with the legacy head actuator retained separately.
The source MJCF merges torso into waist_roll_link and hands into wrists.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from protomotions.robot_configs.base import RobotConfig, RobotAssetConfig, ControlConfig, SimulatorParams
from protomotions.components.pose_lib import ControlInfo
from protomotions.simulator.isaacgym.config import IsaacGymSimParams, IsaacGymPhysXParams
from protomotions.simulator.isaaclab.config import IsaacLabSimParams, IsaacLabPhysXParams
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.robot_configs.astro_p2_pose import P2_DEFAULT_DOF_POS

# User-supplied P2 joint-side motor inertias include gearbox reduction.
ARMATURE_3907_36 = 0.002387232
ARMATURE_5016_25 = 0.0087362778817
ARMATURE_8514_25 = 0.04416342683946
DAMPING_RATIO = 2.0
NATURAL_FREQ_39 = 12.0 * 2.0 * math.pi
NATURAL_FREQ_50 = 10.0 * 2.0 * math.pi
NATURAL_FREQ_85 = 10.0 * 2.0 * math.pi
STIFFNESS_3907_36 = ARMATURE_3907_36 * NATURAL_FREQ_39**2
STIFFNESS_5016_25 = ARMATURE_5016_25 * NATURAL_FREQ_50**2
STIFFNESS_8514_25 = ARMATURE_8514_25 * NATURAL_FREQ_85**2
DAMPING_3907_36 = 2.0 * DAMPING_RATIO * ARMATURE_3907_36 * NATURAL_FREQ_39
DAMPING_5016_25 = 2.0 * DAMPING_RATIO * ARMATURE_5016_25 * NATURAL_FREQ_50
DAMPING_8514_25 = 2.0 * DAMPING_RATIO * ARMATURE_8514_25 * NATURAL_FREQ_85


def _motor_parameters(armature, stiffness, damping, effort_limit, velocity_limit, *, motors=1):
    return dict(
        stiffness=motors * stiffness, damping=motors * damping,
        armature=motors * armature, friction=0.0,
        effort_limit=effort_limit, velocity_limit=velocity_limit,
    )


# Keep the 30-DOF model's semantic order. The reference has no head actuator;
# retain that joint's previous tuning for existing motion/checkpoint compatibility.
CONTROL_PARAMETERS = {
    'left_hip_pitch_joint': _motor_parameters(
        ARMATURE_8514_25, STIFFNESS_8514_25, DAMPING_8514_25,
        130.0, 23.04),
    'left_hip_roll_joint': _motor_parameters(
        ARMATURE_8514_25, STIFFNESS_8514_25, DAMPING_8514_25,
        130.0, 23.04),
    'left_hip_yaw_joint': _motor_parameters(
        ARMATURE_8514_25, STIFFNESS_8514_25, DAMPING_8514_25,
        130.0, 23.04),
    'left_knee_joint': _motor_parameters(
        ARMATURE_8514_25, STIFFNESS_8514_25, DAMPING_8514_25,
        130.0, 23.04),
    'left_ankle_pitch_joint': _motor_parameters(
        ARMATURE_5016_25, STIFFNESS_5016_25, DAMPING_5016_25,
        60.0, 26.18, motors=2),
    'left_ankle_roll_joint': _motor_parameters(
        ARMATURE_5016_25, STIFFNESS_5016_25, DAMPING_5016_25,
        45.0, 26.18, motors=2),
    'right_hip_pitch_joint': _motor_parameters(
        ARMATURE_8514_25, STIFFNESS_8514_25, DAMPING_8514_25,
        130.0, 23.04),
    'right_hip_roll_joint': _motor_parameters(
        ARMATURE_8514_25, STIFFNESS_8514_25, DAMPING_8514_25,
        130.0, 23.04),
    'right_hip_yaw_joint': _motor_parameters(
        ARMATURE_8514_25, STIFFNESS_8514_25, DAMPING_8514_25,
        130.0, 23.04),
    'right_knee_joint': _motor_parameters(
        ARMATURE_8514_25, STIFFNESS_8514_25, DAMPING_8514_25,
        130.0, 23.04),
    'right_ankle_pitch_joint': _motor_parameters(
        ARMATURE_5016_25, STIFFNESS_5016_25, DAMPING_5016_25,
        60.0, 26.18, motors=2),
    'right_ankle_roll_joint': _motor_parameters(
        ARMATURE_5016_25, STIFFNESS_5016_25, DAMPING_5016_25,
        45.0, 26.18, motors=2),
    'waist_yaw_joint': _motor_parameters(
        ARMATURE_8514_25, STIFFNESS_8514_25, DAMPING_8514_25,
        130.0, 13.04),
    'waist_pitch_joint': _motor_parameters(
        ARMATURE_5016_25, STIFFNESS_5016_25, DAMPING_5016_25,
        60.0, 26.18, motors=2),
    'waist_roll_joint': _motor_parameters(
        ARMATURE_5016_25, STIFFNESS_5016_25, DAMPING_5016_25,
        50.0, 26.18, motors=2),
    'left_shoulder_pitch_joint': _motor_parameters(
        ARMATURE_5016_25, STIFFNESS_5016_25, DAMPING_5016_25,
        30.0, 26.18),
    'left_shoulder_roll_joint': _motor_parameters(
        ARMATURE_5016_25, STIFFNESS_5016_25, DAMPING_5016_25,
        30.0, 26.18),
    'left_shoulder_yaw_joint': _motor_parameters(
        ARMATURE_5016_25, STIFFNESS_5016_25, DAMPING_5016_25,
        30.0, 26.18),
    'left_elbow_joint': _motor_parameters(
        ARMATURE_5016_25, STIFFNESS_5016_25, DAMPING_5016_25,
        30.0, 26.18),
    'left_wrist_roll_joint': _motor_parameters(
        ARMATURE_5016_25, STIFFNESS_5016_25, DAMPING_5016_25,
        30.0, 26.18),
    'left_wrist_pitch_joint': _motor_parameters(
        ARMATURE_3907_36, STIFFNESS_3907_36, DAMPING_3907_36,
        10.0, 20.94),
    'left_wrist_yaw_joint': _motor_parameters(
        ARMATURE_3907_36, STIFFNESS_3907_36, DAMPING_3907_36,
        10.0, 20.94),
    'right_shoulder_pitch_joint': _motor_parameters(
        ARMATURE_5016_25, STIFFNESS_5016_25, DAMPING_5016_25,
        30.0, 26.18),
    'right_shoulder_roll_joint': _motor_parameters(
        ARMATURE_5016_25, STIFFNESS_5016_25, DAMPING_5016_25,
        30.0, 26.18),
    'right_shoulder_yaw_joint': _motor_parameters(
        ARMATURE_5016_25, STIFFNESS_5016_25, DAMPING_5016_25,
        30.0, 26.18),
    'right_elbow_joint': _motor_parameters(
        ARMATURE_5016_25, STIFFNESS_5016_25, DAMPING_5016_25,
        30.0, 26.18),
    'right_wrist_roll_joint': _motor_parameters(
        ARMATURE_5016_25, STIFFNESS_5016_25, DAMPING_5016_25,
        30.0, 26.18),
    'right_wrist_pitch_joint': _motor_parameters(
        ARMATURE_3907_36, STIFFNESS_3907_36, DAMPING_3907_36,
        10.0, 20.94),
    'right_wrist_yaw_joint': _motor_parameters(
        ARMATURE_3907_36, STIFFNESS_3907_36, DAMPING_3907_36,
        10.0, 20.94),
    'head_joint': {'stiffness': 9.423498282160118, 'damping': 0.8998777996942602, 'armature': 0.002387, 'friction': 0.0, 'effort_limit': 10.0, 'velocity_limit': 20.94},
}

@dataclass
class AstroP2RobotConfig(RobotConfig):
    semantic_forward_axis_xy: Tuple[float, float] = (1.0, 0.0)
    asset: RobotAssetConfig = field(default_factory=lambda: RobotAssetConfig(
        asset_file_name="astro_p2/mjcf/astro_p2_protomotions.xml",
        replace_cylinder_with_capsule=True, thickness=0.001,
        max_angular_velocity=1000., max_linear_velocity=1000.,
        density=0.001, angular_damping=0., linear_damping=0.,
    ))
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(default_factory=lambda: {
        "all_left_foot_bodies": ["left_ankle_roll_link"],
        "all_right_foot_bodies": ["right_ankle_roll_link"],
        "all_left_hand_bodies": ["left_wrist_yaw_link"],
        "all_right_hand_bodies": ["right_wrist_yaw_link"],
        "head_body_name": ["head_link"],
        "torso_body_name": ["waist_roll_link"],
    })
    anchor_body_name: str = "waist_roll_link"
    trackable_bodies_subset: List[str] = field(default_factory=lambda: [
        "waist_roll_link", "head_link", "left_ankle_roll_link",
        "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
    ])
    default_root_height: float = 0.645  # User-supplied P2 initial root position.
    default_dof_pos: Dict[str, float] = field(default_factory=lambda: P2_DEFAULT_DOF_POS.copy())
    control: ControlConfig = field(default_factory=lambda: ControlConfig(
        pd_action_center="default_pose",
        override_control_info={name: ControlInfo(**values) for name, values in CONTROL_PARAMETERS.items()}
    ))
    simulation_params: SimulatorParams = field(default_factory=lambda: SimulatorParams(
        isaacgym=IsaacGymSimParams(fps=100, decimation=2, substeps=2,
            physx=IsaacGymPhysXParams(num_position_iterations=8, num_velocity_iterations=4, max_depenetration_velocity=1)),
        isaaclab=IsaacLabSimParams(fps=200, decimation=4,
            physx=IsaacLabPhysXParams(num_position_iterations=8, num_velocity_iterations=4, max_depenetration_velocity=1)),
        genesis=GenesisSimParams(fps=100, decimation=2, substeps=2),
        newton=NewtonSimParams(fps=200, decimation=4),
    ))
