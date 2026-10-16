"""Strict Motion–Scene–Contact bindings and unknown-aware reference queries.

The original ragged patches remain on disk. Training uses one strongest patch
per body/object, with a checked capacity; it never silently truncates objects.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from protomotions.components.motion_exclusions import MotionExclusions


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()


def validate_binding(scene_file,motion_lib,asset_file=None,allow_review=False):
    binding_path=Path(str(scene_file)+".binding.json")
    if not binding_path.is_file(): raise ValueError(f"Missing Motion–Scene binding: {binding_path}")
    binding=json.loads(binding_path.read_text())
    if binding["includes_needs_review"] and not allow_review:
        raise ValueError("Scene bundle contains needs_review motions; excluded from Stage 1 training")
    for path,key in ((scene_file,"scene_sha256"),(binding["manifest"],"manifest_sha256"),
                     (motion_lib.motion_file,"motion_file_sha256")):
        if file_sha256(path)!=binding[key]: raise ValueError(f"Motion–Scene binding hash mismatch: {path}")
    if asset_file is not None and file_sha256(asset_file)!=binding["asset_sha256"]:
        raise ValueError("P2 collision asset differs from reconstruction asset")
    ids=binding["motion_ids"]
    if not ids or len(ids)!=len(set(ids)) or min(ids)<0 or max(ids)>=motion_lib.num_motions():
        raise ValueError("Invalid paired motion IDs")
    records={r["motion_local_id"]:r for r in
             (json.loads(x) for x in Path(binding["manifest"]).read_text().splitlines() if x.strip())}
    policy=MotionExclusions(getattr(getattr(motion_lib,"config",None),"motion_exclusion_files",None))
    for i in ids:
        r=records[i]
        policy.require_allowed(r["motion_key"])
        if r["motion_key"]!=motion_lib.motion_files[i] or r["num_frames"]!=int(motion_lib.motion_num_frames[i]):
            raise ValueError(f"Motion ID/key/frame-count mismatch at {i}")
        if not np.isclose(r["dt"],float(motion_lib.motion_dt[i]),rtol=1e-6,atol=1e-9):
            raise ValueError(f"Motion timing mismatch at {i}")
    return binding


class MotionSceneContact:
    def __init__(self,manifest,motion_lib,motion_ids,body_names,device,max_objects_per_link=4):
        root=Path(manifest).parent
        records={r["motion_local_id"]:r for r in
                 (json.loads(x) for x in Path(manifest).read_text().splitlines() if x.strip())}
        B=len(body_names); K=max_objects_per_link
        self.start=torch.full((motion_lib.num_motions(),),-1,dtype=torch.long,device=device)
        self.length=torch.zeros_like(self.start)
        self.dt=motion_lib.motion_dt.to(device)
        arrays={k:[] for k in ("state","confidence","local_points","target_points","patch_weight")}
        offset=0
        for mid in sorted(set(map(int,motion_ids))):
            r=records[mid]
            if r.get("contact_sha256") and file_sha256(root/r["contact_file"])!=r["contact_sha256"]:
                raise ValueError("Contact artifact changed after reconstruction")
            with np.load(root/r["contact_file"],allow_pickle=False) as z:
                if list(z["body_names"])!=list(body_names): raise ValueError("Contact body-name order mismatch")
                if str(z["motion_key"])!=motion_lib.motion_files[mid]: raise ValueError("Contact motion key mismatch")
                state=z["link_contact_state"].copy(); T=len(state)
                if state.shape!=(int(motion_lib.motion_num_frames[mid]),B): raise ValueError("Contact shape mismatch")
                if not np.isin(state,[-1,0,1]).all(): raise ValueError("Contact states must be -1,0,1")
                local=np.zeros((T,B,K,3),np.float32); target=np.zeros_like(local)
                weight=np.zeros((T,B,K),np.float32)
                for row in range(T*B):
                    a,zend=z["patch_offsets"][row:row+2]
                    oids=z["patch_object_ids"][a:zend]
                    unique=np.unique(oids)
                    if len(unique)>K: raise ValueError("Too many simultaneous contact objects; raise explicit capacity")
                    t,b=divmod(row,B)
                    for slot,oid in enumerate(unique):
                        js=np.flatnonzero(oids==oid)+a
                        j=js[np.argmax(z["patch_confidence"][js])]
                        local[t,b,slot]=z["patch_local_points"][j]
                        target[t,b,slot]=z["patch_points"][j]
                        weight[t,b,slot]=z["patch_confidence"][j]
                for k,x in (("state",state),("confidence",z["link_confidence"].copy()),
                            ("local_points",local),("target_points",target),("patch_weight",weight)):
                    arrays[k].append(torch.as_tensor(x,device=device))
                self.start[mid]=offset; self.length[mid]=T; offset+=T
        self.arrays={k:torch.cat(v,0) for k,v in arrays.items()}

    def query(self,motion_ids,times):
        starts=self.start[motion_ids]
        if bool((starts<0).any()): raise ValueError("Motion sampled without its paired scene/contact labels")
        frame=torch.round(times/self.dt[motion_ids]).long()
        frame=torch.minimum(torch.maximum(frame,torch.zeros_like(frame)),self.length[motion_ids]-1)
        return {k:v[starts+frame] for k,v in self.arrays.items()}
