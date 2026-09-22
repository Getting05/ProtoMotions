import pickle

import numpy as np

from calibration_viewer.human import HumanSkeleton, SMPL_JOINT_NAMES, load_smpl_pkl


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
    faces = np.array([[0, 1, 2], [2, 3, 4]])
    with path.open("wb") as stream:
        pickle.dump({"v_template": vertices, "J_regressor": regressor, "f": faces}, stream)
    loaded = load_smpl_pkl(path)
    assert loaded.points.shape == (24, 3)
    assert np.allclose(loaded.points, vertices[0])
    assert np.array_equal(loaded.vertices, vertices)
    assert np.array_equal(loaded.faces, faces)


def test_coordinate_transform_includes_mesh():
    points = np.zeros((24, 3))
    vertices = np.array([[1.0, 2.0, 3.0], [-1.0, 4.0, 2.0]])
    faces = np.array([[0, 1, 1]])
    human = HumanSkeleton(SMPL_JOINT_NAMES, points, vertices=vertices, faces=faces)

    transformed = human.transformed(["z", "x", "y"])

    assert np.array_equal(transformed.vertices, vertices[:, [2, 0, 1]])
    assert np.array_equal(transformed.faces, faces)
