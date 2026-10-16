"""CPU regressions: geometry, uncertainty, provenance, ragged contacts and pairing.

python -m unittest scripts.whole_body_scene.test_pipeline -v
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
from .geometry import CollisionModel,primitive_sdf,primitive_samples,tangent_basis
from .contacts import InferenceConfig,detect_dynamic,detect_static
from .extract_motion import Motion
from .reconstruct import make_box,box_distance,fit_cluster
from .validation import validate_geometry
from protomotions.components.motion_scene_contact import validate_binding,file_sha256,MotionSceneContact
from protomotions.envs.obs.whole_body_contact_obs import whole_body_contact_plan
from protomotions.envs.rewards.whole_body_contact_reward import whole_body_contact_reward,unexpected_contact_penalty


class GeometryTests(unittest.TestCase):
    def test_signed_distances_and_normals(self):
        rng=np.random.default_rng(4)
        for kind,size in [(mujoco.mjtGeom.mjGEOM_BOX,[.1,.2,.3]),
                          (mujoco.mjtGeom.mjGEOM_SPHERE,[.1,0,0]),
                          (mujoco.mjtGeom.mjGEOM_CAPSULE,[.1,.2,0]),
                          (mujoco.mjtGeom.mjGEOM_CYLINDER,[.1,.2,0])]:
            size=np.array(size)
            p,n=primitive_samples(kind,size,40,rng)
            np.testing.assert_allclose(primitive_sdf(p,kind,size),0,atol=1e-8)
            np.testing.assert_allclose(np.linalg.norm(n,axis=1),1,atol=1e-8)
            self.assertLess(primitive_sdf(np.zeros((1,3)),kind,size)[0],0)

    def test_box_rotation_and_interior(self):
        r=Rotation.from_euler("z",45,degrees=True).as_matrix()
        box=make_box([1,2,3],r,[2,.1,.1],"wall",.5)
        self.assertLess(box_distance(np.array([[1,2,3]]),box)[0],0)
        self.assertGreater(box_distance(np.array([[2,2,3]]),box)[0],0)
        np.testing.assert_allclose(tangent_basis([1,2,3]).T@tangent_basis([1,2,3]),np.eye(3),atol=1e-8)

    def test_visuals_excluded_all_collision_bodies_sampled(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"robot.xml"
            p.write_text('''<mujoco><worldbody><body name="body"><freejoint/>
                <geom type="box" size=".1 .1 .1" mass="1"/>
                <geom type="sphere" size="5" contype="0" conaffinity="0"/>
                <body name="hand" pos=".3 0 0"><joint axis="0 1 0"/>
                <geom type="capsule" size=".03 .1" mass="1"/></body>
                </body></worldbody></mujoco>''')
            c=CollisionModel(p,12)
            self.assertEqual(c.model.ngeom,2)
            self.assertEqual(set(c.surface.body),{0,1})
            self.assertEqual(c.body_names,["body","hand"])

    def test_exact_replay_catches_interior_penetration(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"robot.xml"
            p.write_text('<mujoco><worldbody><body name="body"><freejoint/><geom name="b" type="box" size=".1 .1 .1" mass="1"/></body></worldbody></mujoco>')
            c=CollisionModel(p,8)
            motion=Motion(0,"test",.03,np.tile([[[0.,0.,1.]]],(4,1,1)),
                          np.tile([[[0.,0.,0.,1.]]],(4,1,1)),np.zeros((4,0)))
            scene={"objects":[{"object_id":0,"primitives":[make_box([0,0,1],np.eye(3),[.01,.01,.01],"platform",.5)]}],"rejected_objects":[]}
            report=validate_geometry(c,motion,scene,[],InferenceConfig())
            self.assertEqual(report["rejected_object_ids"],[0])
            self.assertEqual(scene["objects"],[])

    def test_obstructed_support_rejected(self):
        events=[{"point":[x,y,1.],"normal":[0,0,1.],"confidence":.8}
                for x,y in [(-.1,-.1),(-.1,.1),(.1,-.1),(.1,.1)]]
        # Robot sweep passes through the proposed slab itself.
        points=np.array([[[0,0,.99]]])
        box,reason=fit_cluster(list(range(4)),events,points,InferenceConfig())
        self.assertIsNone(box)
        self.assertEqual(reason,"support_intersects_robot_sweep")

    def test_floor_transform_is_one_constant_translation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"robot.xml"
            p.write_text('<mujoco><worldbody><body name="body"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
            model=CollisionModel(p,8)
            pos=np.array([[[0,0,z]] for z in [.5,.2,-.1,-.3]],dtype=float)
            motion=Motion(0,"downstairs",.03,pos.copy(),np.tile([0,0,0,1.],(4,1,1)),np.zeros((4,0)))
            points,normals,rots=model.world_samples(motion)
            offset=model.floor_offset(motion,rots,.008)
            np.testing.assert_allclose(offset,[0,0,.401])
            moved=motion.pos+offset
            np.testing.assert_allclose(np.diff(moved,axis=0),np.diff(pos,axis=0))

    def test_small_overlap_refines_geometry_without_moving_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"robot.xml"
            p.write_text('<mujoco><worldbody><body name="body"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
            model=CollisionModel(p,8)
            motion=Motion(0,"support",.03,np.tile([[[0.,0.,1.]]],(4,1,1)),
                          np.tile([0,0,0,1.],(4,1,1)),np.zeros((4,0)))
            obj={"object_id":0,"kind":"platform","surface_point":[0,0,.92],"surface_normal":[0,0,1],
                 "primitives":[make_box([0,0,.46],np.eye(3),[1,1,.92],"platform",.8)]}
            scene={"objects":[obj],"rejected_objects":[]}
            report=validate_geometry(model,motion,scene,[],InferenceConfig())
            self.assertFalse(report["rejected_object_ids"])
            self.assertLessEqual(obj["exact_max_penetration_m"],.008)
            self.assertGreater(obj["normal_retreat_m"],0)
            self.assertLessEqual(obj["normal_retreat_m"],.025)
            np.testing.assert_allclose(motion.pos[:,:,2],1.)


class SupervisionTests(unittest.TestCase):
    def test_stationary_hand_does_not_invent_a_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"hand.xml"
            path.write_text('<mujoco><worldbody><body name="wrist"><freejoint/><geom type="sphere" size=".05" mass="1"/></body></worldbody></mujoco>')
            model=CollisionModel(path,12)
            motion=Motion(0,"idle",.03,np.tile([[[0.,0.,1.]]],(30,1,1)),
                          np.tile([0,0,0,1.],(30,1,1)),np.zeros((30,0)))
            points,normals,rotations=model.world_samples(motion)
            # The hand's own lowest point is not evidence of world-ground contact.
            events,_=detect_static(model,motion,points,normals,rotations,InferenceConfig())
            self.assertTrue(events)
            self.assertTrue(all(e["status"]=="unknown" for e in events))

    def test_static_reset_never_writes_velocity(self):
        from protomotions.simulator.base_simulator.scene_state import write_scene_object_state
        from unittest.mock import Mock
        obj=Mock(); states=torch.randn(3,13);ids=torch.tensor([1,4,7])
        write_scene_object_state(obj,states,ids,torch.tensor([True,False,True]))
        pose,pose_ids=obj.write_root_pose_to_sim.call_args.args
        velocity,velocity_ids=obj.write_root_velocity_to_sim.call_args.args
        torch.testing.assert_close(pose,states[:,:7]);torch.testing.assert_close(pose_ids,ids)
        torch.testing.assert_close(velocity,states[1:2,7:]);self.assertEqual(velocity_ids.tolist(),[4])
        obj.reset_mock()
        write_scene_object_state(obj,states,ids,torch.ones(3,dtype=torch.bool))
        obj.write_root_velocity_to_sim.assert_not_called()

    def test_unknown_not_negative(self):
        state=torch.tensor([[-1,0,1]])
        confidence=torch.ones((1,3))
        obs=whole_body_contact_plan(state[:,None],confidence[:,None])
        torch.testing.assert_close(obs,torch.tensor([[0.,0.,0.,0.,1.,1.,1.,1.,1.]]))
        penalty=unexpected_contact_penalty(state,confidence,torch.tensor([[100.,0.,0.]]))
        self.assertEqual(penalty.item(),0)

    def test_force_on_wrong_object_is_not_rewarded(self):
        state=torch.tensor([[1,1,-1]])
        confidence=torch.ones((1,3))
        forces=torch.ones((1,3))
        matches=torch.tensor([[False,True,True]])
        self.assertAlmostEqual(whole_body_contact_reward(state,confidence,forces,matches).item(),.5)

    def test_no_positive_evidence_has_zero_reward(self):
        z=torch.zeros((2,3)); s=torch.full((2,3),-1)
        self.assertTrue(torch.equal(whole_body_contact_reward(s,z,z,z),torch.zeros(2)))

    def test_dynamic_rigidity_does_not_invent_object(self):
        n=50
        pos=np.zeros((n,2,3));pos[:,0,1]=-.2;pos[:,1,1]=.2;pos[:,:,0]=np.arange(n)[:,None]*.01
        motion=Motion(0,"carrying",.03,pos,np.tile([0,0,0,1.],(n,2,1)),np.zeros((n,0)))
        model=SimpleNamespace(body_names=["left_wrist_yaw_link","right_wrist_yaw_link"],collision_bodies=[0,1])
        result=detect_dynamic(model,motion,InferenceConfig())
        self.assertTrue(result["hypotheses"])
        self.assertEqual(result["status"],"underdetermined")
        self.assertIsNone(result["hypotheses"][0]["geometry"])


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.scene=self.root/"scene.pt";self.scene.write_bytes(b"scene")
        self.motion=self.root/"motion.pt";self.motion.write_bytes(b"motion")
        self.manifest=self.root/"manifest.jsonl"
        self.record={"motion_local_id":0,"motion_key":"walk","num_frames":2,"dt":.1,"contact_file":"contact.npz"}
        self.manifest.write_text(json.dumps(self.record)+"\n")
        self.lib=SimpleNamespace(motion_file=str(self.motion),motion_files=["walk"],motion_num_frames=torch.tensor([2]),
                                 motion_dt=torch.tensor([.1]),num_motions=lambda:1)
        self.binding={"includes_needs_review":False,"manifest":str(self.manifest),
                      "scene_sha256":file_sha256(self.scene),"manifest_sha256":file_sha256(self.manifest),
                      "motion_file_sha256":file_sha256(self.motion),"motion_ids":[0]}
        Path(str(self.scene)+".binding.json").write_text(json.dumps(self.binding))

    def tearDown(self):self.tmp.cleanup()

    def test_matching_binding(self):
        self.assertEqual(validate_binding(self.scene,self.lib)["motion_ids"],[0])

    def test_modified_motion_rejected(self):
        self.motion.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError,"hash mismatch"):validate_binding(self.scene,self.lib)

    def test_motion_key_mismatch_rejected(self):
        self.lib.motion_files=["other"]
        with self.assertRaisesRegex(ValueError,"key/frame-count"):validate_binding(self.scene,self.lib)

    def test_multi_object_patches_and_clip_boundaries(self):
        np.savez(self.root/"contact.npz",body_names=["pelvis"],motion_key="walk",
                 link_contact_state=np.array([[1],[-1]]),link_confidence=np.array([[.8],[0.]],np.float32),
                 patch_offsets=[0,2,2],patch_object_ids=[4,5],patch_confidence=[.7,.8],
                 patch_local_points=[[0,0,0],[1,0,0]],patch_points=[[0,0,1],[1,0,1]])
        ref=MotionSceneContact(self.manifest,self.lib,[0],["pelvis"],"cpu",2)
        q=ref.query(torch.tensor([0,0]),torch.tensor([-1.,9.]))
        self.assertEqual(q["state"].tolist(),[[1],[-1]])
        self.assertEqual((q["patch_weight"][0]>0).sum().item(),2)
        with self.assertRaisesRegex(ValueError,"Too many"):
            MotionSceneContact(self.manifest,self.lib,[0],["pelvis"],"cpu",1)

    def test_packing_preserves_source_id_and_applies_offset(self):
        from .pack import pack
        original=torch.zeros((8,1,3))
        torch.save({"gts":original,"grs":torch.tensor([0.,0.,0.,1.]).expand(8,1,4).clone(),
                    "dps":torch.zeros((8,0)),"motion_num_frames":torch.tensor([4,4]),
                    "length_starts":torch.tensor([0,4]),"motion_dt":torch.tensor([.1,.1]),
                    "motion_files":("first","second")},self.motion)
        scene={"objects":[{"primitives":[{"translation":[0,0,.5],"rotation":[0,0,0,1],"size":[1,1,1]}]}]}
        (self.root/"scene.json").write_text(json.dumps(scene))
        record={"motion_local_id":1,"motion_key":"second","num_frames":4,"dt":.1,
                "scene_file":"scene.json","motion_shard":str(self.motion),"motion_file_sha256":file_sha256(self.motion),
                "asset_sha256":"testasset","training_eligible":True,"status":"geometry_validated",
                "scene_origin_offset":[0,0,1.2]}
        self.manifest.write_text(json.dumps(record)+"\n")
        binding=pack(self.manifest,self.root/"packed.pt")
        self.assertEqual(binding["motion_ids"],[0]);self.assertEqual(binding["source_motion_ids"],[1])
        packed=torch.load(binding["motion_shard"],weights_only=False)
        self.assertEqual(packed["motion_files"],("second",))
        torch.testing.assert_close(packed["gts"][:,:,2],torch.full((4,1),1.2))
        unchanged=torch.load(self.motion,weights_only=False)
        torch.testing.assert_close(unchanged["gts"],original)

    def test_ground_uses_existing_terrain_without_duplicate_collider(self):
        from .pack import pack
        self.test_packing_preserves_source_id_and_applies_offset()
        scene=json.loads((self.root/"scene.json").read_text())
        scene["objects"][0]["primitives"][0]["role"]="ground"
        (self.root/"scene.json").write_text(json.dumps(scene))
        binding=pack(self.manifest,self.root/"ground.pt")
        data=torch.load(self.root/"ground.pt",weights_only=False)
        objects=data["original_scenes"][0]["objects"]
        self.assertEqual(len(objects),1)
        self.assertLessEqual(np.asarray(objects[0]["translation"]).reshape(-1,3)[0,2],-100)
        self.assertEqual(binding["ground_collider"],"existing_simulator_terrain_z0")

    def test_resume_rejects_changed_contact_artifact(self):
        from .run import process_one
        from .extract_motion import MotionArchive
        self.test_packing_preserves_source_id_and_applies_offset()
        archive=MotionArchive(self.motion); motion=archive.get(1)
        token=f"000001_{motion.digest()[:12]}"
        record={"config_sha256":"cfg","motion_file_sha256":"source"}
        for key in ("scene_file","contact_file","graph_file","dynamic_object_file","quality_file","events_file"):
            record[key]=key+".artifact"
            (self.root/record[key]).write_bytes(b"original")
        record["scene_sha256"]=file_sha256(self.root/record["scene_file"])
        record["contact_sha256"]=file_sha256(self.root/record["contact_file"])
        (self.root/"records").mkdir()
        (self.root/"records"/f"{token}.json").write_text(json.dumps(record))
        (self.root/record["contact_file"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"Corrupted resumed artifact"):
            process_one(archive,None,1,self.root,None,"source","cfg",0)


if __name__=="__main__":unittest.main()
