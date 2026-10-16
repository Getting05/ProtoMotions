"""Key-based exclusion regressions across sampling, resume, pairing and packing."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch

from protomotions.components.motion_exclusions import (
    MotionExclusions, allowed_motion_ids, apply_motion_exclusions,
)
from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.envs.motion_manager.motion_manager import MotionManager
from protomotions.envs.motion_manager.config import MotionManagerConfig

BAD = "230323/stairs_climbing_down_loop_R_104__A301.pkl"
MIRROR = "230323/stairs_climbing_jog_down_stop_R_002__A300_M.pkl"


class MotionExclusionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "motions.pt"
        self.keys = ["safe/walk.pkl", "/old/root/"+BAD, "safe/stand.pkl", "/new/root/"+MIRROR]
        frames, bodies = 16, 1
        self.data = dict(
            motion_files=tuple(self.keys), motion_lengths=torch.ones(4)*.3,
            motion_num_frames=torch.ones(4,dtype=torch.long)*4,
            motion_dt=torch.ones(4)*.1, length_starts=torch.arange(4)*4,
            motion_weights=torch.ones(4),gts=torch.arange(frames*3).reshape(frames,bodies,3).float(),
            grs=torch.tensor([0.,0.,0.,1.]).expand(frames,bodies,4).clone(),
            gvs=torch.zeros(frames,bodies,3),gavs=torch.zeros(frames,bodies,3),
            dps=torch.zeros(frames,0),dvs=torch.zeros(frames,0),contacts=torch.ones(frames,bodies,dtype=torch.bool),
        )
        torch.save(self.data,self.path)
        self.lib = MotionLib(MotionLibConfig(motion_file=str(self.path)),"cpu")

    def tearDown(self): self.tmp.cleanup()

    def manager(self, **kwargs):
        return MotionManager(MotionManagerConfig(init_start_prob=0.,**kwargs),2,.02,torch.device("cpu"),self.lib)

    def test_root_changes_and_mirrors_match_but_neighbors_do_not(self):
        p=MotionExclusions()
        self.assertTrue(p.matches("/another/root/"+BAD))
        self.assertTrue(p.matches("C:\\data\\"+MIRROR.replace("/","\\")))
        self.assertFalse(p.matches("230324/"+Path(BAD).name))
        self.assertFalse(p.matches(BAD.replace("A301","A302")))

    def test_load_preserves_ids_frames_and_blocks_direct_query(self):
        self.assertEqual(self.lib.num_motions(),4)
        self.assertEqual(self.lib.motion_files,tuple(self.keys))
        self.assertEqual(allowed_motion_ids(self.lib).tolist(),[0,2])
        self.assertEqual(self.lib.motion_weights.tolist(),[1.,0.,1.,0.])
        torch.testing.assert_close(self.lib.gts,self.data["gts"])
        with self.assertRaisesRegex(ValueError,"Excluded motion"):
            self.lib.get_motion_state_exact_frame(torch.tensor([1]),torch.tensor([0]))
        result=self.lib.get_motion_state_exact_frame(torch.tensor([2]),torch.tensor([0]))
        torch.testing.assert_close(result.rigid_body_pos,self.data["gts"][8:9])

    def test_weight_updates_and_old_checkpoint_cannot_restore_excluded_ids(self):
        manager=self.manager()
        manager.update_sampling_weights(torch.tensor([1.,1e8,1.,1e8]))
        manager.load_state_dict({"motion_file_name":str(self.path),"motion_weights":torch.ones(4)})
        self.assertEqual(manager.motion_weights.tolist(),[1.,0.,1.,0.])
        self.assertEqual(len(manager.get_state_dict()["motion_exclusion_report"]["excluded"]),2)
        self.assertTrue(set(manager.sample_n_motion_ids(10000).tolist()) <= {0,2})
        with self.assertRaisesRegex(ValueError,"Excluded motion"):
            manager.sample_motions(torch.tensor([0]),torch.tensor([1]))

    def test_deterministic_subset_and_fixed_scene_cannot_bypass(self):
        self.assertEqual(self.manager(subset_method="first").available_motion_ids.tolist(),[0,2])
        with self.assertRaisesRegex(ValueError,"Excluded motion"):
            self.manager(subset_method=[0,1])
        with self.assertRaisesRegex(ValueError,"Excluded motion"):
            MotionManager(MotionManagerConfig(),2,.02,torch.device("cpu"),self.lib,torch.tensor([1,2]))

    def test_repack_and_old_config_recompute_key_policy(self):
        self.lib.motion_files=(self.keys[1],self.keys[0],self.keys[2],self.keys[3])
        self.lib.config=SimpleNamespace()  # Old checkpoint has no new field.
        apply_motion_exclusions(self.lib)
        self.assertEqual(allowed_motion_ids(self.lib).tolist(),[1,2])
        manager=self.manager()
        self.assertEqual(manager.motion_ids.tolist(),[1,1])

    def test_custom_policy_is_additive_and_missing_policy_fails_closed(self):
        extra=self.root/"extra.json"
        extra.write_text(json.dumps({"schema_version":1,"exclusions":[
            {"id":"extra","motion_key_suffix":"safe/walk.pkl","reason":"test"}]}))
        lib=MotionLib(MotionLibConfig(motion_file=str(self.path),motion_exclusion_files=[str(extra)]),"cpu")
        self.assertEqual(allowed_motion_ids(lib).tolist(),[2])
        with self.assertRaises(FileNotFoundError):MotionExclusions([str(self.root/"missing.json")])

    def test_all_excluded_fails_instead_of_flat_fallback(self):
        data=dict(self.data,motion_files=(BAD,BAD,MIRROR,MIRROR))
        torch.save(data,self.path)
        with self.assertRaisesRegex(ValueError,"Every motion"):
            MotionLib(MotionLibConfig(motion_file=str(self.path)),"cpu")

    def test_missing_source_keys_fails_closed(self):
        data=dict(self.data)
        del data["motion_files"]
        torch.save(data,self.path)
        with self.assertRaisesRegex(ValueError,"lacks source motion_files keys"):
            MotionLib(MotionLibConfig(motion_file=str(self.path)),"cpu")

    def test_reconstruction_and_review_pack_cannot_force_excluded_motion(self):
        from scripts.whole_body_scene.run import process_one
        from scripts.whole_body_scene.pack import pack
        with self.assertRaisesRegex(ValueError,"Excluded motion"):
            process_one(SimpleNamespace(keys=[BAD]),None,0,self.root,None,"source","cfg",0)
        manifest=self.root/"manifest.jsonl"
        manifest.write_text(json.dumps({"motion_key":BAD,"motion_local_id":0,
                                       "training_eligible":True,"status":"geometry_validated"})+"\n")
        with self.assertRaisesRegex(ValueError,"No eligible scenes"):
            pack(manifest,self.root/"scenes.pt",include_review=True)

    def test_filter_manifest_keeps_audit_and_is_idempotent(self):
        from scripts.motion_exclusions import filter_manifest
        manifest=self.root/"manifest.jsonl"
        rows=[{"motion_key":k,"motion_local_id":i,"training_eligible":True,"status":"geometry_validated"}
              for i,k in enumerate(self.keys)]
        manifest.write_text("".join(json.dumps(r)+"\n" for r in rows))
        report=filter_manifest(manifest,MotionExclusions())
        self.assertEqual(report["active_records"],2)
        self.assertEqual([r["motion_local_id"] for r in report["excluded"]],[1,3])
        self.assertTrue(list(self.root.glob("manifest.jsonl.before_exclusions.*")))
        again=filter_manifest(manifest,MotionExclusions())
        self.assertEqual(len(again["excluded"]),2)

    def test_add_command_can_expand_exact_exclusion_to_mirrors(self):
        from unittest.mock import patch
        from scripts.motion_exclusions import main
        custom=self.root/"added.json"
        args=["motion_exclusions", "add", "--file", str(custom),
              "--motion-file", str(self.path), "--ids", "0", "--reason", "Bad retarget"]
        with patch("sys.argv", [*args,"--exact-only"]): main()
        with patch("sys.argv", args): main()
        with patch("sys.argv", args): main()  # Repeating an addition is idempotent.
        rules=json.loads(custom.read_text())["exclusions"]
        self.assertEqual(len(rules),2)
        self.assertEqual(len({r["id"] for r in rules}),2)
        self.assertTrue(MotionExclusions([str(custom)]).matches("safe/walk_M.pkl"))

    def test_mimic_evaluation_enumerates_only_allowed_source_ids(self):
        from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator
        evaluator=SimpleNamespace(
            motion_lib=self.lib, motion_manager=self.manager(),
            agent=SimpleNamespace(current_epoch=0),
            device=torch.device("cpu"), num_envs=2,
            config=SimpleNamespace(eval_num_motions=None),
            fabric=SimpleNamespace(global_rank=0),
        )
        batches=MimicEvaluator._build_eval_batches(evaluator)
        self.assertEqual(torch.cat([ids for _,ids in batches]).tolist(),[0,2])


if __name__ == "__main__": unittest.main()
