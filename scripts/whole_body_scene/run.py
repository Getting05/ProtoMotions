"""Reconstruct a MotionLib shard without modifying its reference trajectories.

python -m scripts.whole_body_scene.run --motion-file ... --output ... --ids 0,6
"""
from __future__ import annotations
import argparse
from dataclasses import fields
from pathlib import Path
import hashlib
import json
import os
import time
import traceback
import numpy as np
import torch
from .geometry import CollisionModel, sha256
from .extract_motion import MotionArchive
from .contacts import InferenceConfig, detect_static, detect_dynamic
from .reconstruct import reconstruct_static, interaction_graph
from .validation import validate_geometry, build_labels, validate_support


def write_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def process_one(archive,collision,index,output,cfg,source_hash,config_hash,support_frames):
    start=time.monotonic(); motion=archive.get(index)
    digest=motion.digest()
    token=f"{index:06d}_{digest[:12]}"
    path=output/"records"/f"{token}.json"
    if path.exists():
        old=json.loads(path.read_text())
        if old["config_sha256"] != config_hash or old["motion_file_sha256"] != source_hash:
            raise ValueError("Output was produced with a different source/config; use a new output directory")
        for key in ("scene_file", "contact_file", "graph_file", "dynamic_object_file", "quality_file", "events_file"):
            artifact = output/old[key]
            if not artifact.is_file():
                raise ValueError(f"Incomplete resumed record: {artifact}")
        for key,hash_key in (("scene_file","scene_sha256"),("contact_file","contact_sha256")):
            if sha256(output/old[key]) != old[hash_key]:
                raise ValueError(f"Corrupted resumed artifact: {output/old[key]}")
        return old
    fk=collision.verify_fk(motion)
    print(f"motion {index}: FK checked; sampling all collision bodies",flush=True)
    points,normals,rotations=collision.world_samples(motion)
    origin_offset=collision.floor_offset(motion,rotations,cfg.penetration_tolerance)
    motion.pos+=origin_offset
    points+=origin_offset
    events,candidate_metrics=detect_static(collision,motion,points,normals,rotations,cfg)
    print(f"motion {index}: {len(events)} contact candidates; fitting geometry",flush=True)
    dynamic=detect_dynamic(collision,motion,cfg)
    scene=reconstruct_static(events,points,collision,cfg)
    print(f"motion {index}: {len(scene['objects'])} objects; exact collision replay",flush=True)
    geometry=validate_geometry(collision,motion,scene,events,cfg)
    labels,contact_metrics=build_labels(collision,motion,points,normals,scene,events,cfg)
    # Dynamic evidence is unknown supervision, even if empty-scene clearance
    # had suggested a negative. Accepted static positives remain independent.
    for h in dynamic["hypotheses"]:
        for b in h["body_ids"]:
            a,z=h["start"],h["end"]
            unknown=labels["link_contact_state"][a:z,b] != 1
            labels["link_contact_state"][a:z,b][unknown]=-1
            labels["link_confidence"][a:z,b][unknown]=0
    support=validate_support(collision,motion,labels,cfg,support_frames)
    reasons=[]
    if not geometry["ground_pass"]: reasons.append("reference_penetrates_simulator_ground")
    if not scene["objects"]: reasons.append("no_explained_static_support")
    if dynamic["hypotheses"]: reasons.append("dynamic_object_hypothesis_requires_review")
    if support["force_residual_p95_over_weight"] > .3: reasons.append("large_support_force_residual")
    if support["moment_residual_p95_normalized"] > .2: reasons.append("large_support_moment_residual")
    if support["joint_residual_p95_over_limit"] > .2: reasons.append("large_joint_torque_residual")
    if support["solver_failures"]: reasons.append("support_solver_did_not_converge")
    if geometry["self_collision_frames"]: reasons.append("reference_self_collision")
    status="geometry_validated" if geometry["ground_pass"] and scene["objects"] else "rejected"
    if status != "rejected" and reasons: status="needs_review"
    quality={"status":status,"reasons":reasons,"fk_max_error_m":fk,"candidate_inference":candidate_metrics,
             "geometry":geometry,"contacts":contact_metrics,"support":support,
             "isaaclab_closed_loop":{"status":"not_run"},"runtime_seconds":time.monotonic()-start}
    paths={"scene_file":f"static_scenes/{token}.json","contact_file":f"contacts/{token}.npz",
           "graph_file":f"graphs/{token}.json","dynamic_object_file":f"dynamic_objects/{token}.json",
           "quality_file":f"validation/{token}.json","events_file":f"events/{token}.json"}
    for folder in ("contacts","static_scenes","graphs","dynamic_objects","validation","events"):
        (output/folder).mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/paths["contact_file"],**labels)
    write_json(output/paths["scene_file"],scene)
    write_json(output/paths["graph_file"],interaction_graph(scene,events,collision,dynamic))
    write_json(output/paths["dynamic_object_file"],dynamic)
    write_json(output/paths["quality_file"],quality)
    write_json(output/paths["events_file"],events)
    record={"schema_version":1,"motion_key":motion.key,"motion_local_id":index,"robot":"astro_p2",
            "motion_shard":str(archive.path),"motion_file_sha256":source_hash,"motion_sha256":motion.digest(),
            "asset_sha256":collision.asset_sha256,"config_sha256":config_hash,
            "source_motion_sha256":digest,"scene_origin_offset":origin_offset.tolist(),
            "num_frames":len(motion.pos),"dt":motion.dt,"status":status,"reasons":reasons,
            "scene_type":"static_multi_contact","coordinate_frame":"source_motion_world_plus_scene_origin_offset",
            "quaternion_order":"xyzw","training_eligible":status=="geometry_validated",
            "physical_execution_validated":False,**paths}
    record["contact_sha256"]=sha256(output/paths["contact_file"])
    record["scene_sha256"]=sha256(output/paths["scene_file"])
    write_json(path,record)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion-file",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--model",default="protomotions/data/assets/astro_p2/mjcf/astro_p2_protomotions.xml")
    p.add_argument("--ids",help="Comma-separated source-shard local IDs")
    p.add_argument("--match",help="Case-insensitive filename selection, never used as contact evidence")
    p.add_argument("--limit",type=int)
    p.add_argument("--stride",type=int,default=1)
    p.add_argument("--offset",type=int,default=0)
    p.add_argument("--samples-per-geom",type=int,default=24)
    p.add_argument("--support-frames",type=int,default=32,help="0 checks all frames")
    p.add_argument("--config",help="JSON overrides of InferenceConfig")
    args=p.parse_args()
    if args.stride<1 or args.offset<0 or args.samples_per_geom<4 or args.support_frames<0:
        p.error("Invalid stride, offset, sample count or support frame count")
    torch.set_num_threads(1)
    cfg=InferenceConfig(samples_per_geom=args.samples_per_geom)
    if args.config:
        for k,v in json.loads(Path(args.config).read_text()).items():
            if k not in {f.name for f in fields(cfg)}: raise ValueError(f"Unknown config key {k}")
            setattr(cfg,k,v)
    output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=True)
    # Output lock prevents simultaneous manifest writers/resume corruption.
    import fcntl
    with open(output/".writer.lock","w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        archive=MotionArchive(args.motion_file)
        collision=CollisionModel(args.model,cfg.samples_per_geom,cfg.seed)
        signature={"inference":cfg.to_dict(),"support_frames":args.support_frames,
                   "asset_sha256":collision.asset_sha256,"schema_version":1,
                   "algorithm_sha256":hashlib.sha256("".join(sha256(Path(__file__).parent/name) for name in
                       ("geometry.py","extract_motion.py","contacts.py","reconstruct.py","validation.py","run.py")).encode()).hexdigest()}
        config_hash=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        source_hash=sha256(archive.path)
        signature.update(config_sha256=config_hash,motion_file_sha256=source_hash,motion_shard=str(archive.path))
        config_path=output/"config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != signature:
            raise ValueError("Refusing to mix different motion shards, assets or settings in one output")
        write_json(config_path,signature); write_json(output/"collision_model.json",collision.report())
        ids=list(map(int,args.ids.split(","))) if args.ids else list(range(args.offset,len(archive),args.stride))
        if args.match: ids=[i for i in ids if args.match.lower() in archive.keys[i].lower()]
        if args.limit is not None: ids=ids[:args.limit]
        if not ids: raise ValueError("No motions selected")
        failures=[]
        for index in ids:
            try:
                record=process_one(archive,collision,index,output,cfg,source_hash,config_hash,args.support_frames)
                print(json.dumps({"id":index,"status":record["status"],"reasons":record["reasons"]}),flush=True)
            except Exception as e:
                failure={"id":index,"error":str(e),"traceback":traceback.format_exc()}
                failures.append(failure); write_json(output/"failures"/f"{index:06d}.json",failure)
                print(json.dumps({"id":index,"error":str(e)}),flush=True)
        records=[json.loads(x.read_text()) for x in sorted((output/"records").glob("*.json"))] if (output/"records").exists() else []
        tmp=output/"manifest.jsonl.tmp"
        tmp.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records)); tmp.replace(output/"manifest.jsonl")
        summary={"selected":len(ids),"total_completed":len(records),"failures":failures,
                 "statuses":{s:sum(r["status"]==s for r in records) for s in ("geometry_validated","needs_review","rejected")},
                 "physical_execution_validated":False}
        write_json(output/"summary.json",summary); print(json.dumps(summary),flush=True)
        if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()
