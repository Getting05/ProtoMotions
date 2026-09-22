from pathlib import Path

import pytest

from calibration_viewer.config import load_config


CONFIG_DIR = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize(
    ("filename", "model", "pelvis", "left_hip", "left_shoulder"),
    [
        (
            "unitree_g1.yaml",
            "unitree_g1_29dof",
            "pelvis_contour_link",
            "left_hip_pitch_link",
            "left_shoulder_pitch_link",
        ),
        (
            "unitree_h1_2.yaml",
            "unitree_h1_2_27dof",
            "pelvis",
            "left_hip_yaw_link",
            "left_shoulder_roll_link",
        ),
    ],
)
def test_unitree_presets(filename, model, pelvis, left_hip, left_shoulder):
    robot = load_config(CONFIG_DIR / filename)["robot"]

    assert robot["model"] == model
    assert robot["keypoint_links"]["pelvis"] == pelvis
    assert robot["keypoint_links"]["left_hip"] == left_hip
    assert robot["keypoint_links"]["left_shoulder"] == left_shoulder
    assert robot["t_pose_joint_positions"]["left_elbow_joint"] == pytest.approx(1.5708)
    assert robot["t_pose_joint_positions"]["right_elbow_joint"] == pytest.approx(1.5708)

