"""Reference overlay geometry, physics isolation, pose ordering and alignment."""

from types import SimpleNamespace as NS

import pytest
import torch

from protomotions.utils.reference_video import ReferenceRobot, reference_pose


def test_reference_pose_uses_motion_time_and_environment_alignment():
    ids = torch.tensor([3])
    times = torch.tensor([0.25])
    positions = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    rotation = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]])

    def get_motion(motion_ids, motion_times):
        assert motion_ids is ids and motion_times is times
        return NS(rigid_body_pos=positions, rigid_body_rot=rotation)

    env = NS(
        motion_lib=NS(get_motion_state=get_motion),
        motion_manager=NS(motion_ids=ids, motion_times=times),
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=lambda p: (
            torch.ones_like(p) * 10
        ),
    )
    p, q = reference_pose(env)
    assert torch.equal(p, positions[0] + 10)
    assert torch.equal(q, rotation[0])
    assert positions[0, 0, 0] == 1  # library data never mutated


def test_reference_clones_visible_shapes_without_physics_or_duplicate_links():
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/World/Robot/hip")
    parent.AddTranslateOp().Set(Gf.Vec3d(1, 0, 0))
    UsdPhysics.RigidBodyAPI.Apply(parent.GetPrim())
    visual = UsdGeom.Cube.Define(stage, "/World/Robot/hip/visual")
    visual.AddTranslateOp().Set(Gf.Vec3d(0.1, 0.2, 0.3))
    visual.CreateSizeAttr(0.25)
    child = UsdGeom.Xform.Define(stage, "/World/Robot/hip/knee")
    UsdPhysics.RigidBodyAPI.Apply(child.GetPrim())
    UsdGeom.Sphere.Define(stage, "/World/Robot/hip/knee/visual")
    hidden = UsdGeom.Cube.Define(stage, "/World/Robot/hip/collision")
    UsdPhysics.CollisionAPI.Apply(hidden.GetPrim())
    hidden.MakeInvisible()
    overlay = ReferenceRobot.__new__(ReferenceRobot)
    overlay.path = "/World/PolicyVideoReference"
    overlay.stage = stage
    overlay._build(stage, [parent.GetPath(), child.GetPath()], [1, 0])
    assert overlay.shape_count == 2
    overlay.set_pose([[10, 20, 30], [40, 50, 60]], [[0, 0, 0, 1], [0, 0, 0, 1]])
    cache = UsdGeom.XformCache()
    target = stage.GetPrimAtPath(overlay.path + "/body_0/shape_0")
    actual = cache.GetLocalToWorldTransform(target).ExtractTranslation()
    assert list(actual) == pytest.approx([40.1, 50.2, 60.3])
    shader = UsdShade.Shader(
        stage.GetPrimAtPath(overlay.path + "/BlueReference/Surface")
    )
    assert shader.GetInput("opacity").Get() == pytest.approx(1.0)
    assert list(shader.GetInput("diffuseColor").Get()) == pytest.approx(
        [0.03, 0.25, 1.0]
    )
    for prim in Usd.PrimRange(stage.GetPrimAtPath(overlay.path)):
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert not prim.HasAPI(UsdPhysics.CollisionAPI)
        assert not prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        if prim.IsA(UsdGeom.Gprim):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            assert str(material.GetPath()) == overlay.path + "/BlueReference"
    assert parent.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    overlay.close()
    assert not stage.GetPrimAtPath(overlay.path)


def test_reference_blending_keeps_policy_and_background_visible():
    import numpy as np

    from protomotions.utils.policy_video import blend_reference

    base = np.array([[[200, 200, 200], [20, 30, 40]]], dtype=np.uint8)
    overlay = np.array([[[0, 0, 255], [20, 30, 40]]], dtype=np.uint8)
    result = blend_reference(base, overlay, 0.4)
    assert result.tolist() == [[[120, 120, 222], [20, 30, 40]]]
    assert base[0, 0].tolist() == [200, 200, 200]


def test_camera_composites_same_instant_and_restores_reference_visibility():
    from unittest.mock import Mock

    import numpy as np

    from protomotions.utils.policy_video import FollowCamera

    camera = FollowCamera.__new__(FollowCamera)
    base = np.full((2, 2, 3), 200, dtype=np.uint8)
    overlay = np.full((2, 2, 3), 100, dtype=np.uint8)
    camera._frame = Mock(side_effect=[base, base, overlay, overlay])
    reference = Mock()
    assert np.all(camera.frame(reference) == 160)
    assert [c.args for c in reference.set_visible.call_args_list] == [(False,), (True,)]
    camera._frame = Mock(side_effect=RuntimeError("render failed"))
    with pytest.raises(RuntimeError, match="render failed"):
        camera.frame(reference)
    reference.set_visible.assert_called_with(True)
