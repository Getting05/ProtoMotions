from pathlib import Path
import importlib.util

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration_viewer.calibration import CalibrationParams, calibrate_human_points, fit_calibration_frames
from calibration_viewer.config import load_config
from calibration_viewer.human import SMPL_EDGES
from calibration_viewer.motion import HumanMotion, RootTrajectoryParams, fit_root_trajectory, load_human_motion, reference_robot_frames
from calibration_viewer.robot import RobotModel


@pytest.fixture
def demo(tmp_path):
    spec = importlib.util.spec_from_file_location('demo', Path(__file__).parents[1]/'examples/generate_demo.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.generate(tmp_path, frames=6)
    cfg = load_config(tmp_path/'config.yaml')
    robot = RobotModel.load(tmp_path/'robot.urdf', base_link='pelvis', keypoint_links=cfg['robot']['keypoint_links'])
    motion = load_human_motion(tmp_path/'human.npz', axes=['x','y','z'])
    return tmp_path, robot, motion


def test_root_scale_offsets_and_recovery():
    t = np.linspace(0, 1, 10)
    source = np.column_stack([2+t, 3+t**2, 1+.1*np.sin(t*3)])
    params = RootTrajectoryParams(np.array([.8, 1.2, .9]), np.array([.1,-.2,.05]))
    target = params.apply(source)
    np.testing.assert_allclose(target[0], [2.1,2.8,.95])
    np.testing.assert_allclose(np.diff(target, axis=0), np.diff(source, axis=0)*params.scale)
    solved, info = fit_root_trajectory(source, target, RootTrajectoryParams())
    np.testing.assert_allclose(solved.scale, params.scale, atol=1e-6)
    np.testing.assert_allclose(solved.offset, params.offset, atol=1e-6)
    assert info['rank'] == 6


def test_root_static_rank_warning():
    _, info = fit_root_trajectory(np.ones((5,3)), np.ones((5,3)), RootTrajectoryParams())
    assert info['rank'] < 6 and info['warning']


def test_world_heading_and_offsets_follow_turn(demo):
    _, _, motion = demo
    params = CalibrationParams.defaults('bone')
    params.joint_offsets = {'shoulder': [.1,0,0]}
    local, world = motion.calibrated(params, RootTrajectoryParams())
    index = motion.names.index('left_shoulder')
    np.testing.assert_allclose(world[:,index] - motion.points[:,index], motion.root_rotations @ [.1,0,0], atol=1e-9)
    np.testing.assert_allclose(local[:,0], 0, atol=1e-9)


def test_motion_selection_and_reference_fk(demo):
    path, robot, motion = demo
    selected = load_human_motion(path/'human.npz', axes=['x','y','z'], start=1, stop=6, stride=2)
    assert selected.fps == 15 and selected.source_frames.tolist() == [1,3,5]
    reference, roots = reference_robot_frames(path/'reference.npz', robot, selected, source_fps=30)
    np.testing.assert_allclose(roots, selected.roots)
    np.testing.assert_allclose([[f[n] for n in selected.names] for f in reference], selected.local_points, atol=1e-8)
    with pytest.raises(ValueError, match='fps'):
        reference_robot_frames(path/'reference.npz', robot, motion, source_fps=25)


def test_multiframe_recovers_bones_and_offsets(demo):
    _, robot, motion = demo
    params = CalibrationParams.defaults('bone')
    params.bone_scales = dict(torso=.8, upper_arm=.9, forearm=1.1, thigh=.85, shank=.95)
    params.joint_offsets = {'shoulder':[.02,.01,-.01], 'elbow':[.01,.005,.015], 'hip':[-.01,.02,-.015]}
    frames = motion.frames()
    targets = [calibrate_human_points(f, params) for f in frames]
    fit, info = fit_calibration_frames(frames, targets, list(robot.keypoint_links), initial=CalibrationParams.defaults('bone'),
                                      scale_regularization=1e-9, offset_regularization=1e-9)
    assert info['frame_count'] == 6 and info['rank'] == 14
    np.testing.assert_allclose(fit.vector(), params.vector(), atol=2e-5)


def test_multi_frame_rejects_mismatch(demo):
    _, robot, motion = demo
    with pytest.raises(ValueError, match='frame count'):
        fit_calibration_frames(motion.frames(), motion.frames()[:1], list(robot.keypoint_links))


def test_motion_requires_frame_rate_and_finite_data(tmp_path):
    path=tmp_path/'motion.npz'
    np.savez(path, joints=np.zeros((2,24,3)))
    with pytest.raises(ValueError, match='fps'):
        load_human_motion(path)
    with pytest.raises(ValueError, match='Degenerate'):
        load_human_motion(path, fps=30)
    np.savez(path, joints=np.full((2,24,3), np.nan), fps=30)
    with pytest.raises(ValueError, match='finite'):
        load_human_motion(path)


def test_pyroki_retargets_reachable_motion(demo):
    pytest.importorskip('pyroki')
    from calibration_viewer.retarget import retarget_motion, save_motion
    path, robot, motion = demo
    root = RootTrajectoryParams(np.array([.8,1.1,.95]), np.array([0.,0.,.02]))
    output, info = retarget_motion(motion, robot, CalibrationParams.defaults('bone'), root,
                                   rest_weight=0., smoothness_weight=0., max_iterations=60)
    assert info['backend'] == 'pyroki+jaxls'
    assert info['position_rmse_m'] < 1e-4
    assert info['frames_above_threshold'] == 0
    assert np.isfinite(output['joint_positions']).all()
    assert np.max(np.abs(output['joint_positions'])) <= 1.4
    np.testing.assert_allclose(output['pelvis_positions'], root.apply(motion.roots))
    save_motion(path/'result.npz', output)
    with np.load(path/'result.npz', allow_pickle=False) as saved:
        assert saved['joint_positions'].shape == (6,8)
        assert saved['root_quat_xyzw'].shape == (6,4)


def test_pyroki_unreachable_targets_report_error(demo):
    pytest.importorskip('pyroki')
    from calibration_viewer.retarget import retarget_motion
    _, robot, motion = demo
    motion = HumanMotion(motion.names, motion.points[:1], motion.fps, motion.root_rotations[:1])
    params = CalibrationParams.defaults('bone')
    params.bone_scales = {'upper_arm':2., 'forearm':2.}
    output, info = retarget_motion(motion, robot, params, RootTrajectoryParams(), max_iterations=30)
    assert info['frames_above_threshold'] == 1
    assert info['max_keypoint_error_m'] > .1
    assert np.isfinite(output['joint_positions']).all()
    assert np.max(np.abs(output['joint_positions'])) <= 1.4
