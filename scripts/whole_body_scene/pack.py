"""Pack motion-paired SceneLib files, with binding metadata and safe padding."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .extract_motion import MotionArchive
from .geometry import sha256
from .run import write_json
from protomotions.components.motion_exclusions import MotionExclusions


def pack(manifest,output,include_review=False,exclude_files=None):
    from protomotions.components.scene_lib import Scene, SceneLib, BoxSceneObject, ObjectOptions
    manifest=Path(manifest).resolve(); root=manifest.parent
    records=[json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    policy=MotionExclusions(exclude_files)
    blocked=[r for r in records if policy.matches(r["motion_key"])]
    records=[r for r in records if not policy.matches(r["motion_key"])]
    records=[r for r in records if r["training_eligible"] or (include_review and r["status"] == "needs_review")]
    if not records:
        raise ValueError("No eligible scenes; inspect validation reports. Refusing a silent flat-ground fallback")
    sources={r["motion_file_sha256"] for r in records}
    assets={r["asset_sha256"] for r in records}
    if len(sources)!=1 or len(assets)!=1:
        raise ValueError("Each SceneLib shard must correspond to exactly one motion shard and one asset")
    output=Path(output).resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    archive=MotionArchive(records[0]["motion_shard"])
    if sha256(archive.path)!=next(iter(sources)):
        raise ValueError("Source motion shard changed after reconstruction")
    frame_fields=("gts","grs","gvs","gavs","dps","dvs","contacts","lrs","goal_states")
    packed={k:[] for k in frame_fields if k in archive.data and archive.data[k] is not None}
    packed_records=[]; counts=[]; dts=[]; keys=[]
    for new_id,r in enumerate(records):
        old_id=r["motion_local_id"]
        if not 0 <= old_id < len(archive) or archive.keys[old_id] != r["motion_key"]:
            raise ValueError(f"Source motion ID/key mismatch at {old_id}")
        policy.require_allowed(archive.keys[old_id])
        start=int(archive.data["length_starts"][old_id]); count=int(archive.data["motion_num_frames"][old_id])
        for k in packed:
            value=archive.data[k][start:start+count].clone()
            if k=="gts":
                value=(value.double()+torch.as_tensor(r.get("scene_origin_offset",[0,0,0]),dtype=torch.float64)).to(value.dtype)
            packed[k].append(value)
        counts.append(count); dts.append(r["dt"]); keys.append(r["motion_key"])
        record=dict(r,source_motion_local_id=old_id,source_motion_shard=r["motion_shard"],
                    source_motion_file_sha256=r["motion_file_sha256"],motion_local_id=new_id)
        # Keep artifact paths absolute so the packed manifest can live elsewhere.
        for key in ("scene_file","contact_file","graph_file","quality_file","events_file","dynamic_object_file"):
            if record.get(key): record[key]=str(root/record[key])
        packed_records.append(record)
    data={k:torch.cat(v,0) for k,v in packed.items()}
    data.update(motion_num_frames=torch.tensor(counts),motion_dt=torch.tensor(dts),
                motion_lengths=(torch.tensor(counts)-1)*torch.tensor(dts),
                motion_weights=torch.ones(len(records)),
                length_starts=torch.tensor([0]+np.cumsum(counts).tolist()[:-1]),motion_files=tuple(keys))
    motion_path=output.with_name(output.stem+".motions.pt")
    torch.save(data,motion_path)
    motion_hash=sha256(motion_path)
    packed_manifest=output.with_name(output.stem+".manifest.jsonl")
    for r in packed_records:
        r["motion_shard"]=str(motion_path); r["motion_file_sha256"]=motion_hash
    packed_manifest.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in packed_records))
    scenes=[]
    for new_id,r in enumerate(records):
        if r.get("scene_sha256") and sha256(root/r["scene_file"])!=r["scene_sha256"]:
            raise ValueError("Reconstructed scene changed after validation")
        scene=json.loads((root/r["scene_file"]).read_text())
        objects=[]
        for entity in scene["objects"]:
            for box in entity["primitives"]:
                # Isaac Lab already supplies the z=0 terrain collider. Adding
                # another coplanar ground box duplicates physical contacts.
                # Its inferred contact labels still refer to the real floor.
                if box.get("role")=="ground":
                    continue
                objects.append(BoxSceneObject(translation=box["translation"],rotation=box["rotation"],
                    width=box["size"][0],depth=box["size"][1],height=box["size"][2],
                    options=ObjectOptions(fix_base_link=True,static_friction=.7,dynamic_friction=.7)))
        if not scene["objects"]:
            raise ValueError(f"Motion {r['motion_local_id']} has no surviving geometry")
        scenes.append(Scene(objects=objects,humanoid_motion_id=new_id))
    slots=max(1,max(len(s.objects) for s in scenes))
    for s in scenes:
        while len(s.objects)<slots:
            # Fixed below the ground, beyond the complete robot's sampled region;
            # never a tiny collidable object left at the reference origin.
            s.objects.append(BoxSceneObject(translation=[0,0,-100-len(s.objects)],
                rotation=[0,0,0,1],width=.001,depth=.001,height=.001,
                options=ObjectOptions(fix_base_link=True)))
    output=Path(output).resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    SceneLib.save_scenes_to_file(scenes,str(output))
    binding={"schema_version":1,"manifest":str(packed_manifest),"manifest_sha256":sha256(packed_manifest),
             "scene_sha256":sha256(output),"motion_file_sha256":motion_hash,"source_motion_file_sha256":next(iter(sources)),
             "asset_sha256":next(iter(assets)),"motion_shard":str(motion_path),
             "motion_ids":list(range(len(records))),"source_motion_ids":[r["motion_local_id"] for r in records],"object_slots":slots,
             "includes_needs_review":include_review,"physics_validated":False,
             "ground_collider":"existing_simulator_terrain_z0",
             "exclusion_policy_files":policy.files,"excluded_source_motion_ids":[r["motion_local_id"] for r in blocked],
             "padding_z_max":-100,"strict_motion_pairing":True}
    write_json(str(output)+".binding.json",binding)
    return binding


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True); p.add_argument("--output",required=True)
    p.add_argument("--include-needs-review",action="store_true",help="Inspection only; training loader rejects these by default")
    p.add_argument("--exclude-file",action="append",default=[])
    args=p.parse_args(); print(json.dumps(pack(args.manifest,args.output,args.include_needs_review,args.exclude_file),indent=2))


if __name__=="__main__": main()
