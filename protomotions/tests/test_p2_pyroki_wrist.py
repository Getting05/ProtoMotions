"""P2 wrist regression at the original PyRoki layer only."""
import runpy
from pathlib import Path
import numpy as np
import pytest
import yourdfpy
import pyroki as pk

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def module():
    return runpy.run_path(str(ROOT/'pyroki/batch_retarget_to_astro_p2_from_keypoints.py'))


@pytest.mark.parametrize('side,offset', [('left',[-.0149,.084,-.0082]),('right',[-.0103,-.0846,-.0061])])
def test_aux_uses_anatomical_direction_and_p2_length(module,side,offset):
    p=np.array([[.3,.4,.7],[1.,2.,3.]])
    r=np.repeat(np.eye(3)[None],2,axis=0)
    before=p.copy()
    result=module['p2_smpl_hand_aux'](p,r,side)
    expected=np.array(offset)/np.linalg.norm(offset)*.11
    np.testing.assert_allclose(result-p,np.broadcast_to(expected,p.shape))
    np.testing.assert_array_equal(p,before)


def test_aux_is_rotation_equivariant(module):
    from scipy.spatial.transform import Rotation
    r=Rotation.from_euler('xyz',[.2,.3,-.5]).as_matrix()
    f=module['p2_smpl_hand_aux']
    np.testing.assert_allclose(f(np.zeros(3),r,'left'),r@f(np.zeros(3),np.eye(3),'left'))


def test_loader_only_changes_hand_aux_after_existing_scaling(module,tmp_path):
    p=np.arange(2*18*3,dtype=float).reshape(2,18,3)*.01
    r=np.broadcast_to(np.eye(3),(2,18,3,3)).copy()
    path=tmp_path/'source.npy'
    np.save(path,dict(positions=p,orientations=r,left_foot_contacts=np.ones((2,2)),right_foot_contacts=np.ones((2,2)),fps=30.))
    out,_,_,_,n,_=module['load_motion_data'](str(path),'smpl',1,2,30.)
    assert n==2
    root=p[:,:1]*[.9,.9,.85]
    expected=(p-p[:,:1])*np.array([.9,.9,.8])+root
    expected[:,:9]=(p[:,:9]-p[:,:1])*[.9,.9,.85]+root
    keep=[i for i in range(18) if i not in (15,16)]
    np.testing.assert_allclose(out[:,keep],expected[:,keep])
    for wrist,aux,side in [(13,15,'left'),(14,16,'right')]:
        np.testing.assert_allclose(out[:,aux],module['p2_smpl_hand_aux'](out[:,wrist],r[:,wrist],side))
    np.testing.assert_array_equal(np.load(path,allow_pickle=True).item()['positions'],p)


def test_native_collision_model_ignores_placeholders_but_keeps_wrist_hip(module):
    urdf=yourdfpy.URDF.load(ROOT/'protomotions/data/assets/astro_p2/urdf/astro_p2_retarget.urdf',load_meshes=False)
    model=module['build_p2_collision_model'](urdf)
    assert isinstance(model,pk.collision.RobotCollision)
    pairs={frozenset((model.link_names[i],model.link_names[j])) for i,j in zip(model.active_idx_i,model.active_idx_j)}
    assert frozenset(('right_wrist_pitch_link','right_hip_roll_link')) in pairs
    assert frozenset(('left_wrist_yaw_link','right_wrist_yaw_link')) in pairs
    for pair in pairs:
        assert all(urdf.link_map[n].collisions for n in pair)
    robot=pk.Robot.from_urdf(urdf)
    distances=np.asarray(model.compute_self_collision_distance(robot,robot.joint_var_cls.default_factory()))
    assert np.isfinite(distances).all()
    assert distances.min()>-.001


def test_no_second_ik_stage():
    text=(ROOT/'scripts/retarget_amass_to_robot.sh').read_text()
    assert 'refine_astro_p2_arms' not in text
    assert not (ROOT/'data/scripts/refine_astro_p2_arms.py').exists()
