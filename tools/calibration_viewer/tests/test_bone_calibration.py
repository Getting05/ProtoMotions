import numpy as np
import pytest

from calibration_viewer.calibration import CalibrationParams, calibrate_human_points, fit_calibration
from calibration_viewer.config import load_config, save_config
from calibration_viewer.human import SMPL_EDGES


@pytest.fixture
def skeleton():
    # Non-axis-aligned full skeleton makes accidental Cartesian scaling observable.
    points = {"pelvis": np.array([.3, -.4, 1.2])}
    remaining = list(SMPL_EDGES)
    rng = np.random.default_rng(42)
    while remaining:
        for parent, child in remaining[:]:
            if parent in points:
                points[child] = points[parent] + rng.normal(size=3) * .2
                remaining.remove((parent, child))
    return points


def test_identity_and_input_order(skeleton):
    params = CalibrationParams.defaults("bone")
    result = calibrate_human_points(dict(reversed(list(skeleton.items()))), params)
    for name in skeleton:
        np.testing.assert_allclose(result[name], skeleton[name])


def test_upper_arm_only_preserves_other_lengths(skeleton):
    params = CalibrationParams.defaults("bone")
    params.bone_scales = {"upper_arm": .6}
    result = calibrate_human_points(skeleton, params)
    for parent, child in SMPL_EDGES:
        factor = .6 if child.endswith("_elbow") else 1.
        np.testing.assert_allclose(result[child] - result[parent], factor * (skeleton[child] - skeleton[parent]))
    np.testing.assert_allclose(result["neck"], skeleton["neck"])


def test_torso_moves_shoulders_without_scaling_arms(skeleton):
    params = CalibrationParams.defaults("bone")
    params.bone_scales = {"torso": .8}
    result = calibrate_human_points(skeleton, params)
    for side in ("left", "right"):
        np.testing.assert_allclose(result[f"{side}_wrist"] - result[f"{side}_shoulder"],
                                   skeleton[f"{side}_wrist"] - skeleton[f"{side}_shoulder"])
    assert not np.allclose(result["neck"], skeleton["neck"])


def test_offsets_propagate_and_mirror_only_y(skeleton):
    params = CalibrationParams.defaults("bone")
    params.joint_offsets = {"shoulder": [.02, .03, .01], "hip": [-.01, .02, -.03], "elbow": [.01, -.02, .04]}
    result = calibrate_human_points(skeleton, params)
    for side, sign in (("left", 1), ("right", -1)):
        mirror = np.array([1, sign, 1])
        for joint in ("hip", "knee", "ankle", "foot"):
            np.testing.assert_allclose(result[f"{side}_{joint}"] - skeleton[f"{side}_{joint}"], np.array(params.joint_offsets["hip"]) * mirror)
        for joint in ("elbow", "wrist", "hand"):
            expected = (np.array(params.joint_offsets["shoulder"]) + params.joint_offsets["elbow"]) * mirror
            np.testing.assert_allclose(result[f"{side}_{joint}"] - skeleton[f"{side}_{joint}"], expected)


def test_asymmetric_and_inactive_legacy_parameters(skeleton):
    params = CalibrationParams.defaults("bone")
    params.asymmetric = True
    params.bone_scales = {"upper_arm": .9, "left_upper_arm": .5}
    params.upper_scale = [2, 2, 2]
    params.shoulder_offset = .1
    result = calibrate_human_points(skeleton, params)
    for side, factor in (("left", .5), ("right", .9)):
        np.testing.assert_allclose(result[f"{side}_elbow"] - result[f"{side}_shoulder"], factor * (skeleton[f"{side}_elbow"] - skeleton[f"{side}_shoulder"]))


@pytest.mark.parametrize("asymmetric", [False, True])
def test_fit_known_bone_geometry(skeleton, asymmetric):
    params = CalibrationParams.defaults("bone")
    params.asymmetric = asymmetric
    x = params.vector()
    count = len(x) - 9
    x[:count] = np.linspace(.7, 1.2, count)
    x[count:] = np.linspace(-.03, .03, 9)
    expected = params.with_vector(x)
    target = calibrate_human_points(skeleton, expected)
    fit, info = fit_calibration(skeleton, target, list(skeleton), initial=params,
                                scale_regularization=1e-9, offset_regularization=1e-9)
    assert info["success"]
    assert info["parameter_count"] == len(x)
    assert info["rmse_m"] < 1e-6
    np.testing.assert_allclose(fit.vector(), x, atol=1e-5)


def test_yaml_roundtrip(tmp_path):
    params = CalibrationParams.defaults("bone")
    params.asymmetric = True
    params.bone_scales = {"torso": .9, "left_thigh": 1.1}
    params.joint_offsets = {"hip": np.array([.01, .02, .03])}
    path = tmp_path / "result.yaml"
    save_config(path, {"format_version": 2, "robot": {"base_link": "base", "keypoint_links": {}}, "calibration": params.serializable()})
    loaded = CalibrationParams.from_mapping(load_config(path)["calibration"])
    np.testing.assert_allclose(params.vector(), loaded.vector())
    assert loaded.mode == "bone"
    assert CalibrationParams.from_mapping({"shoulder_offset": .04}).mode == "legacy"


def test_missing_parent_is_explicit(skeleton):
    del skeleton["left_collar"]
    with pytest.raises(ValueError, match="left_collar"):
        calibrate_human_points(skeleton, CalibrationParams.defaults("bone"))


@pytest.mark.parametrize("data", [{"mode": "typo"}, {"bone_scales": {"thigh": -1}},
                                  {"joint_offsets": {"hip": [1, 2]}}, {"upper_scale": [1, float("nan"), 1]}])
def test_bad_config(data):
    with pytest.raises(ValueError):
        CalibrationParams.from_mapping(data)


def test_collar_branches_from_spine3(skeleton):
    # The neck is a sibling of the collars in SMPL, not an arm ancestor.
    params = CalibrationParams.defaults("bone")
    params.bone_scales = {"torso": .7}
    before = calibrate_human_points(skeleton, params)
    skeleton["neck"] = skeleton["neck"] + [0, 0, .2]
    after = calibrate_human_points(skeleton, params)
    np.testing.assert_allclose(before["left_wrist"], after["left_wrist"])
    np.testing.assert_allclose(before["right_wrist"], after["right_wrist"])


def test_switching_symmetry_preserves_inactive_scales():
    params = CalibrationParams.defaults("bone")
    params.bone_scales = {"upper_arm": .9, "left_upper_arm": .7}
    params = params.with_vector(params.vector())
    params.asymmetric = True
    assert params.vector()[1] == .7
    assert params.vector()[5] == .9
