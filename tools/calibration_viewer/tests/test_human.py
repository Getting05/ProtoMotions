import pickle

import numpy as np

from calibration_viewer.human import SMPL_JOINT_NAMES, load_smpl_pkl


def test_load_precomputed_joints(tmp_path):
    path = tmp_path / "sample.pkl"
    joints = np.arange(24 * 3, dtype=float).reshape(24, 3)
    with path.open("wb") as stream:
        pickle.dump({"joints": joints}, stream)
    loaded = load_smpl_pkl(path)
    assert loaded.names == SMPL_JOINT_NAMES
    assert np.array_equal(loaded.points, joints)


def test_load_model_regressor(tmp_path):
    path = tmp_path / "model.pkl"
    vertices = np.arange(30, dtype=float).reshape(10, 3)
    regressor = np.zeros((24, 10)); regressor[:, 0] = 1.0
    with path.open("wb") as stream:
        pickle.dump({"v_template": vertices, "J_regressor": regressor}, stream)
    loaded = load_smpl_pkl(path)
    assert loaded.points.shape == (24, 3)
    assert np.allclose(loaded.points, vertices[0])

