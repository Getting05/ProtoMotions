"""Regression tests for v2's sampling, temporal state and physical objectives."""
import tempfile
from pathlib import Path
import numpy as np
import mujoco
from .sampling_v2 import primitive_samples_deterministic
from .geometry import CollisionModel,primitive_sdf
from .planes_v2 import patch_area,stable_intervals,horizontal_foot_windows
from types import SimpleNamespace
from .scene_v2 import contact_track,SweptSurface
from .reconstruct import safe_box,make_box
from scipy.spatial.transform import Rotation
from .wrench_v2 import WrenchScorer,EffortScorer
from .extract_motion import Motion


def sorted_rows(x):return sorted(map(tuple,np.round(x,8)))


def test_symmetric_quadrature_has_no_seed_or_order_dependence():
    kinds=[(mujoco.mjtGeom.mjGEOM_BOX,[.1,.07,.03]),
           (mujoco.mjtGeom.mjGEOM_SPHERE,[.1,0,0]),
           (mujoco.mjtGeom.mjGEOM_CAPSULE,[.07,.12,0]),
           (mujoco.mjtGeom.mjGEOM_CYLINDER,[.07,.12,0])]
    for kind,size in kinds:
        p,n=primitive_samples_deterministic(kind,size,96)
        np.testing.assert_allclose(primitive_sdf(p,kind,np.array(size)),0.,atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(n,axis=1),1.,atol=1e-9)
        original=sorted_rows(np.column_stack((p,n)))
        for axis in range(3):
            reflected=np.column_stack((p,n));reflected[:,[axis,axis+3]]*=-1
            assert sorted_rows(reflected)==original


def test_patch_area_does_not_grow_with_duplicate_samples():
    p=np.array([[0,0,1],[.1,0,1],[0,.2,1],[.1,.2,1]])
    assert abs(patch_area(p,[0,0,1])-.02)<1e-10
    assert patch_area(np.repeat(p,20,axis=0),[0,0,1])==patch_area(p,[0,0,1])


def test_swept_broad_phase_matches_signed_distance():
    rng=np.random.default_rng(82)
    points=rng.normal(size=(10000,3));sweep=SweptSurface(points)
    for _ in range(40):
        box=make_box(rng.uniform(-3,3,3),Rotation.random(random_state=rng).as_matrix(),
                     rng.uniform(.01,.8,3),'test',1.)
        assert sweep.safe(box,.006)==safe_box(points,box,.006)


def test_contact_hysteresis_extends_rolling_support_until_separation():
    d=np.array([.04,.02,.006,.004,.005,.008,.013,.022,.032,.04])
    v=np.array([-.3,-.2,-.1,0,.05,.1,.15,.2,.3,.4])
    active=contact_track(d,v,np.full(10,.002),1/30)
    np.testing.assert_array_equal(active,[False,False,True,True,True,True,True,True,False,False])


def test_contact_tracks_never_bridge_a_real_gap():
    d=np.array([.003]*3+[.05]*2+[.003]*3)
    a=contact_track(d,np.zeros(8),np.full(8,.002),1/30)
    np.testing.assert_array_equal(a,[True,True,True,False,False,True,True,True])


def test_isolated_proximity_is_not_a_persistent_contact():
    d=np.array([.1,.003,.1,.1,.003,.003,.1])
    assert not contact_track(d,np.zeros(7),np.full(7,.002),1/30).any()


def test_stable_plane_interval_does_not_merge_different_heights():
    z=np.r_[np.full(15,.2),np.linspace(.2,.5,8),np.full(15,.5)]
    intervals,_=stable_intervals(z,1/30)
    assert len(intervals)>=2
    assert all(np.ptp(z[a:b])<=.022+1e-12 for a,b in intervals)


def test_rolling_patch_survives_changing_lowest_sample():
    model=SimpleNamespace(body_names=['left_ankle_roll_link'],surface=SimpleNamespace(body=np.zeros(4,int)))
    motion=SimpleNamespace(dt=1/30)
    p=np.tile([[0,0,.2],[.1,0,.2],[0,.1,.2],[.1,.1,.2]],(18,1,1))
    p[:,3,2]=.19+.009*np.sin(np.arange(18)*1.3)
    n=np.tile([0.,0.,-1.],(18,4,1))
    windows=horizontal_foot_windows([],model,motion,p,n)
    assert windows[0]==[(0,18)]


def test_root_wrench_obeys_unilateral_normal():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'robot.xml'
        path.write_text('<mujoco><worldbody><body name="root"><freejoint/><geom type="sphere" size=".1" mass="2"/></body></worldbody></mujoco>')
        model=CollisionModel(path,96,sampling='symmetric')
        motion=Motion(0,'test',1/30,np.tile([[[0.,0.,.5]]],(10,1,1)),
                      np.tile([0.,0.,0.,1.],(10,1,1)),np.zeros((10,0)))
        score=WrenchScorer(model,motion)
        no_contact=score.score([[] for _ in range(10)])
        ground=score.score([[([0.,0.,0.],[0.,0.,1.])] for _ in range(10)])
        ceiling=score.score([[([0.,0.,1.],[0.,0.,-1.])] for _ in range(10)])
        np.testing.assert_allclose(no_contact,1.,atol=1e-10)
        assert ground.max()<1e-10
        np.testing.assert_allclose(ceiling,1.,atol=1e-10)


def test_articulated_support_distinguishes_same_wrench_on_different_links():
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'arm.xml'
        path.write_text('''<mujoco><worldbody><body name="root"><freejoint/>
        <geom type="sphere" size=".1" mass="5"/><body name="arm">
        <joint name="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 1 0 0" size=".03" mass="2"/>
        </body></body></worldbody><actuator><motor joint="hinge" forcelimited="true" forcerange="-20 20"/></actuator></mujoco>''')
        model=CollisionModel(path,96,sampling='symmetric')
        motion=Motion(0,'test',1/30,np.tile([[[0.,0.,.5],[0.,0.,.5]]],(10,1,1)),
            np.tile([0.,0.,0.,1.],(10,2,1)),np.zeros((10,1)))
        scorer=EffortScorer(model,motion)
        # Identical world points/normals give the same root wrench, but only
        # support transmitted through the arm can relieve its hinge torque.
        base=[([0.,0.,0.],[0.,0.,1.],0)]
        root=scorer.score([base+[([1.,0.,.5],[0.,0.,1.],0)] for _ in range(10)])
        arm=scorer.score([base+[([1.,0.,.5],[0.,0.,1.],1)] for _ in range(10)])
        assert arm.mean()<root.mean()*.1
