import numpy as np

from calibration_viewer.calibration import CalibrationParams, calibrate_human_points, fit_calibration


def test_fit_recovers_known_parameters():
    human = {
        "pelvis": np.array([0.0, 0.0, 0.0]),
        "left_hip": np.array([0.0, 0.12, 0.0]), "right_hip": np.array([0.0, -0.12, 0.0]),
        "left_knee": np.array([0.03, 0.12, -0.45]), "right_knee": np.array([0.03, -0.12, -0.45]),
        "left_ankle": np.array([0.04, 0.12, -0.90]), "right_ankle": np.array([0.04, -0.12, -0.90]),
        "left_shoulder": np.array([0.01, 0.23, 0.55]), "right_shoulder": np.array([0.01, -0.23, 0.55]),
        "left_elbow": np.array([0.02, 0.52, 0.53]), "right_elbow": np.array([0.02, -0.52, 0.53]),
        "left_wrist": np.array([0.06, 0.80, 0.49]), "right_wrist": np.array([0.06, -0.80, 0.49]),
    }
    expected = CalibrationParams(np.array([0.91, 0.88, 0.80]), np.array([0.93, 0.90, 0.85]), 0.02, 0.045)
    robot = calibrate_human_points(human, expected)
    solved, info = fit_calibration(human, robot, list(human), scale_regularization=1e-8, offset_regularization=1e-8)
    assert info["rmse_m"] < 1e-5
    assert np.allclose(solved.upper_scale, expected.upper_scale, atol=2e-4)
    assert np.allclose(solved.lower_scale, expected.lower_scale, atol=2e-4)
    assert abs(solved.shoulder_offset - expected.shoulder_offset) < 2e-4
    assert abs(solved.elbow_offset - expected.elbow_offset) < 2e-4

