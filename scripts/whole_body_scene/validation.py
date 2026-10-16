"""Exact primitive distance checks and bounded inverse-dynamics feasibility."""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.optimize import lsq_linear
from .geometry import tangent_basis
from .reconstruct import box_distance


def validate_geometry(collision, motion, scene, events, cfg):
    root = ET.fromstring(collision.xml)
    world = root.find("worldbody")
    mapping = {}
    for obj in scene["objects"]:
        for j, box in enumerate(obj["primitives"]):
            name = f"wbc_object_{obj['object_id']}_{j}"
            q = np.array(box["rotation"])[[3,0,1,2]]
            ET.SubElement(world,"geom",name=name,type="box",size=" ".join(map(str,np.array(box["size"])/2)),
                          pos=" ".join(map(str,box["translation"])),quat=" ".join(map(str,q)))
            mapping[name] = obj["object_id"]
    ET.SubElement(world,"geom",name="wbc_base_ground",type="plane",size="0 0 .05",pos="0 0 0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    # The inserted world geoms change geom ordering. Resolve by original names.
    robot_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] > 0 and
                 (model.geom_contype[i] or model.geom_conaffinity[i])]
    object_ids = {mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,n):oid for n,oid in mapping.items()}
    floor = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"wbc_base_ground")
    object_ids[floor] = -1
    worst = {oid: 0. for oid in object_ids.values()}
    bad_counts = {oid: 0 for oid in object_ids.values()}
    self_depth, self_frames = 0., 0
    qpos = collision.qpos(motion)
    fromto = np.empty(6)
    refinements=[]
    for refinement_pass in range(3):
        worst = {oid: 0. for oid in object_ids.values()}
        bad_counts = {oid: 0 for oid in object_ids.values()}
        self_depth, self_frames = 0., 0
        # Exact geometry at every recorded frame, rather than only sampled surfaces.
        for t,q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_forward(model,data)
            # MuJoCo 3.8 mj_geomDistance returns zero for complete convex
            # containment (e.g. a small box inside a larger box). The collision
            # manifold still reports signed depth, so retain both measurements.
            manifold={}
            for contact in data.contact:
                key=tuple(sorted((int(contact.geom1),int(contact.geom2))))
                manifold[key]=max(manifold.get(key,0.),-float(contact.dist))
            depths_by_object={oid:0. for oid in object_ids.values()}
            for og,oid in object_ids.items():
                frame_depth = 0.
                for g in robot_ids:
                    dist = mujoco.mj_geomDistance(model,data,g,og,.05,fromto)
                    frame_depth = max(frame_depth,-float(dist),manifold.get(tuple(sorted((g,og))),0.))
                depths_by_object[oid]=max(depths_by_object[oid],frame_depth)
            for oid,frame_depth in depths_by_object.items():
                worst[oid] = max(worst[oid],frame_depth)
                bad_counts[oid] += int(frame_depth > cfg.penetration_tolerance)
            # Actual self contacts only; separate from reconstructed environment.
            depths = [-float(c.dist) for c in data.contact
                      if model.geom_bodyid[c.geom1] > 0 and model.geom_bodyid[c.geom2] > 0
                      and model.geom_bodyid[c.geom1] != model.geom_bodyid[c.geom2]]
            depth = max(depths,default=0.)
            self_depth = max(self_depth,depth)
            self_frames += int(depth > cfg.penetration_tolerance)
        changed=False
        if refinement_pass < 2:
            for obj in scene["objects"]:
                oid=obj["object_id"]; depth=worst[oid]
                used=obj.get("normal_retreat_m",0.)
                remaining=cfg.contact_tolerance-used
                if obj.get("kind")=="ground" or depth<=cfg.penetration_tolerance or depth>remaining+cfg.penetration_tolerance:
                    continue
                shift=min(remaining,depth-cfg.penetration_tolerance+.002)
                if shift<=0: continue
                normal=np.asarray(obj["surface_normal"])
                for j,box in enumerate(obj["primitives"]):
                    box["translation"]=(np.asarray(box["translation"])-normal*shift).tolist()
                    g=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f"wbc_object_{oid}_{j}")
                    model.geom_pos[g]=box["translation"]
                obj["surface_point"]=(np.asarray(obj["surface_point"])-normal*shift).tolist()
                obj["normal_retreat_m"]=used+shift
                refinements.append({"object_id":oid,"retreat_m":shift,"previous_penetration_m":depth})
                changed=True
        if not changed: break
    rejected = {oid for oid,v in worst.items() if oid >= 0 and v > cfg.penetration_tolerance}
    for obj in scene["objects"]:
        obj["exact_max_penetration_m"] = worst[obj["object_id"]]
        obj["status"] = "rejected" if obj["object_id"] in rejected else "geometry_validated"
    for e in events:
        if e["object_id"] in rejected:
            e["status"] = "rejected"; e["reason"] = "exact_primitive_penetration"
        elif e["object_id"] >= 0:
            e["status"] = "geometry_validated"
    scene["rejected_objects"] += [o for o in scene["objects"] if o["object_id"] in rejected]
    scene["objects"] = [o for o in scene["objects"] if o["object_id"] not in rejected]
    return {"backend": "mujoco_mj_geomDistance_all_primitives_all_frames", "frames_checked": len(qpos),
            "inter_frame_continuous_collision_checked": False,
            "max_penetration_m_by_object": {str(k):v for k,v in worst.items()},
            "penetrating_frames_by_object": {str(k):v for k,v in bad_counts.items()},
            "rejected_object_ids": sorted(rejected), "ground_max_penetration_m": worst[-1],
            "self_collision_max_penetration_m": self_depth, "self_collision_frames": self_frames,
            "ground_pass": worst[-1] <= cfg.penetration_tolerance,
            "physics_validation": "not_run", "contact_tolerance_refinements":refinements}


def build_labels(collision, motion, points, normals, scene, events, cfg):
    frames, bodies = motion.pos.shape[:2]
    state = np.full((frames,bodies),-1,np.int8)
    confidence = np.zeros((frames,bodies),np.float32)
    object_id = np.full((frames,bodies),-1,np.int32)
    contact_type = np.zeros((frames,bodies),np.int8)
    objects = {o["object_id"]:o for o in scene["objects"]}
    # No-contact evidence is relative to the reconstructed scene ONLY; uncertain
    # rejected events are masked again below, never relabeled as negatives.
    for b in collision.collision_bodies:
        ps = points[:,collision.surface.body == b]
        d = np.abs(ps[...,2]).min(axis=1)
        for o in scene["objects"]:
            for box in o["primitives"]:
                d = np.minimum(d,box_distance(ps,box).min(axis=1))
        free = d > cfg.contact_tolerance*1.5
        state[free,b] = 0; confidence[free,b] = .5
    patches = [[] for _ in range(frames*bodies)]
    for e in events:
        a,z,b = e["start"],e["end"],e["body_id"]
        if e["status"] != "geometry_validated":
            state[a:z,b] = -1; confidence[a:z,b] = 0
    distances, alignments = [], []
    for e in events:
        if e["status"] != "geometry_validated":
            continue
        obj = objects[e["object_id"]]
        a,z,b,s = e["start"],e["end"],e["body_id"],e["sample_id"]
        ps = points[a:z,s]
        ds = np.abs(box_distance(ps,obj["primitives"][0]))
        alignment = (-normals[a:z,s])@np.array(obj["surface_normal"])
        valid = (ds <= cfg.contact_tolerance) & (alignment >= cfg.normal_cos)
        for rel in np.flatnonzero(valid):
            t = a+int(rel)
            state[t,b] = 1
            confidence[t,b] = max(confidence[t,b],e["confidence"])
            prev = object_id[t,b]
            object_id[t,b] = e["object_id"] if prev in (-1,e["object_id"]) else -2
            contact_type[t,b] = e["contact_type"]
            patches[t*bodies+b].append((ps[rel],obj["surface_normal"],e["confidence"],
                                       e["object_id"],e["event_id"],collision.surface.points[s]))
            distances.append(float(ds[rel])); alignments.append(float(alignment[rel]))
    flat = [p for row in patches for p in row]
    offsets = np.r_[0,np.cumsum([len(row) for row in patches])].astype(np.int64)
    data = {"link_contact_state": state,"link_confidence":confidence,"scene_object_id":object_id,
            "contact_type":contact_type,"patch_offsets":offsets,
            "patch_points":np.asarray([p[0] for p in flat],np.float32).reshape(-1,3),
            "patch_normals":np.asarray([p[1] for p in flat],np.float32).reshape(-1,3),
            "patch_confidence":np.asarray([p[2] for p in flat],np.float32),
            "patch_object_ids":np.asarray([p[3] for p in flat],np.int32),
            "patch_event_ids":np.asarray([p[4] for p in flat],np.int32),
            "patch_local_points":np.asarray([p[5] for p in flat],np.float32).reshape(-1,3),
            "body_names":np.asarray(collision.body_names),"dt":np.array(motion.dt),
            "motion_key":np.array(motion.key),"motion_sha256":np.array(motion.digest())}
    metrics = {"desired_contact_distance_mean_m":float(np.mean(distances)) if distances else None,
               "normal_alignment_mean":float(np.mean(alignments)) if alignments else None,
               "known_fraction":float(np.mean(state >= 0)),"positive_link_frames":int((state == 1).sum()),
               "accepted_event_fraction":sum(e["status"] == "geometry_validated" for e in events)/max(1,len(events)),
               "body_part_contact_frames":{n:int((state[:,b] == 1).sum()) for b,n in enumerate(collision.body_names)},
               "manual_precision":None,"coverage_is_detector_evidence_not_ground_truth":True}
    return data,metrics


def validate_support(collision,motion,labels,cfg,max_frames=32):
    """Bounded least squares QP on M qdd + h = S tau + J^T f.

    Friction is an inscribed 4-ray pyramid. Joint torques obey the P2 config.
    Residuals score feasibility; this is not a tracking or grasp success test.
    """
    from protomotions.robot_configs.astro_p2 import CONTROL_PARAMETERS
    m,d = collision.model,collision.data
    qpos = collision.qpos(motion)
    qvel = np.zeros((len(qpos),m.nv))
    for t in range(1,len(qpos)):
        mujoco.mj_differentiatePos(m,qvel[t],motion.dt,qpos[t-1],qpos[t])
    qvel[0] = qvel[1]
    qacc = np.gradient(qvel,motion.dt,axis=0)
    limits = np.array([CONTROL_PARAMETERS[n]["effort_limit"] for n in collision.joint_names])
    count = min(max_frames,len(qpos)) if max_frames else len(qpos)
    ids = np.unique(np.linspace(0,len(qpos)-1,count).astype(int))
    metrics = []
    B = len(collision.body_names)
    total_weight = float(m.body_mass.sum()*9.81)
    for t in ids:
        d.qpos[:]=qpos[t]; d.qvel[:]=qvel[t]
        mujoco.mj_forward(m,d)
        mass=np.zeros((m.nv,m.nv)); mujoco.mj_fullM(m,mass,d.qM)
        target=mass@qacc[t]+d.qfrc_bias
        cols=[]
        for b in range(B):
            a,z=labels["patch_offsets"][t*B+b:t*B+b+2]
            # Up to four spatially distributed points per body capture support
            # moments without hundreds of redundant force variables.
            patch_ids=np.arange(a,z)
            if len(patch_ids)>4:
                p=labels["patch_points"][patch_ids]
                chosen=[0]
                for _ in range(3):
                    dist=np.linalg.norm(p[:,None]-p[chosen][None],axis=-1).min(axis=1)
                    chosen.append(int(np.argmax(dist)))
                patch_ids=patch_ids[np.unique(chosen)]
            for j in patch_ids:
                jac=np.zeros((3,m.nv)); jacr=np.zeros_like(jac)
                mujoco.mj_jac(m,d,jac,jacr,labels["patch_points"][j].astype(float),b+1)
                basis=tangent_basis(labels["patch_normals"][j])
                rays=np.array([[cfg.friction,0,1],[-cfg.friction,0,1],
                               [0,cfg.friction,1],[0,-cfg.friction,1]])@basis.T
                cols.extend(jac.T@rays.T[:,k] for k in range(4))
        act=np.zeros((m.nv,len(limits))); act[6:]=np.eye(len(limits))
        A=np.column_stack([act,*cols]) if cols else act
        lower=np.r_[-limits,np.zeros(len(cols))]
        upper=np.r_[limits,np.full(len(cols),total_weight*5)]
        fit=lsq_linear(A,target,bounds=(lower,upper),tol=1e-5,max_iter=100)
        residual=A@fit.x-target
        metrics.append({"frame":int(t),"force_residual_over_weight":float(np.linalg.norm(residual[:3])/total_weight),
                        "base_moment_residual_nm":float(np.linalg.norm(residual[3:6])),
                        "base_moment_residual_normalized":float(np.linalg.norm(residual[3:6])/(total_weight*.5)),
                        "joint_residual_over_effort_limit":float(np.max(np.abs(residual[6:])/limits)),
                        "generalized_residual_norm":float(np.linalg.norm(residual)),
                        "solver_success":bool(fit.success),"contact_rays":len(cols)})
    return {"status":"scored","method":"bounded_inverse_dynamics_friction_pyramid_QP",
            "friction_prior":cfg.friction,"frames_checked":len(metrics),"total_frames":len(qpos),
            "all_frames_checked":len(metrics)==len(qpos),
            "force_residual_p95_over_weight":float(np.percentile([x["force_residual_over_weight"] for x in metrics],95)),
            "moment_residual_p95_normalized":float(np.percentile([x["base_moment_residual_normalized"] for x in metrics],95)),
            "joint_residual_p95_over_limit":float(np.percentile([x["joint_residual_over_effort_limit"] for x in metrics],95)),
            "solver_failures":sum(not x["solver_success"] for x in metrics),
            "samples":metrics,"closed_loop_validated":False}
