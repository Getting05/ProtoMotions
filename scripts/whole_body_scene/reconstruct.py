"""Normal-aware clustering, finite 3D primitives and swept-volume constraints."""
import copy
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .geometry import unit, tangent_basis, primitive_sdf
import mujoco


def box_distance(points, box):
    r = Rotation.from_quat(box["rotation"]).as_matrix()
    return primitive_sdf((points-np.asarray(box["translation"]))@r,
                         mujoco.mjtGeom.mjGEOM_BOX, np.asarray(box["size"])/2)


def make_box(center, rotation, size, role, confidence):
    return {"type": "box", "translation": np.asarray(center).tolist(),
            "rotation": Rotation.from_matrix(rotation).as_quat().tolist(),
            "size": np.asarray(size).tolist(), "role": role, "confidence": float(confidence),
            "fixed": True}


def cluster_events(events, cfg):
    """DBSCAN-style core expansion with both spatial and oriented plane affinity."""
    valid = [i for i,e in enumerate(events) if e["status"] == "candidate"]
    if not valid:
        return []
    p = np.array([events[i]["point"] for i in valid])
    n = np.array([events[i]["normal"] for i in valid])
    tree = cKDTree(p)
    neighbors = []
    for i, pt in enumerate(p):
        ids = np.asarray(tree.query_ball_point(pt, cfg.cluster_radius), dtype=int)
        delta = p[ids]-pt
        good = (n[ids]@n[i] > cfg.normal_cos) & (np.abs(delta@n[i]) < cfg.plane_tolerance)
        good &= np.abs(np.sum(delta*n[ids], axis=1)) < cfg.plane_tolerance
        neighbors.append(ids[good])
    core = np.array([len(x) >= 3 for x in neighbors])
    visited = np.zeros(len(p), bool)
    clusters = []
    for i in range(len(p)):
        if visited[i] or not core[i]:
            continue
        queue, members = [i], []
        visited[i] = True
        while queue:
            j = queue.pop(); members.append(j)
            if core[j]:
                for k in neighbors[j]:
                    if not visited[k]:
                        visited[k] = True; queue.append(int(k))
        clusters.append([valid[j] for j in members])
    for j in np.flatnonzero(~visited):
        events[valid[j]]["status"] = "unknown"
        events[valid[j]]["reason"] = "insufficient_spatial_contact_evidence"
    return clusters


def split_empty_regions(ids, events, depth=0):
    p = np.array([events[i]["point"] for i in ids])
    n = unit(np.mean([events[i]["normal"] for i in ids], axis=0))
    basis = tangent_basis(n)
    xy = (p-p.mean(0))@basis[:, :2]
    _, _, v = np.linalg.svd(xy, full_matrices=False)
    xy = xy@v.T
    extent = np.ptp(xy, axis=0)
    area = max(np.prod(extent), .0064)
    occupied = len(np.unique(np.floor((xy-xy.min(0))/.08), axis=0))*.0064
    if depth >= 5 or len(ids) < 12 or max(extent) < .25 or occupied/area > .4:
        return [ids]
    order = np.argsort(xy[:, 0]); middle = len(order)//2
    return split_empty_regions([ids[i] for i in order[:middle]], events, depth+1) + \
           split_empty_regions([ids[i] for i in order[middle:]], events, depth+1)


def safe_box(points, box, tolerance):
    return float(np.min(box_distance(points, box))) >= -tolerance


def fit_cluster(ids, events, points, cfg):
    p = np.array([events[i]["point"] for i in ids])
    n = unit(np.mean([events[i]["normal"] for i in ids], axis=0))
    if n[2] > .90 and np.ptp(p[:, 2]) < cfg.plane_tolerance*2:
        n = np.array([0., 0., 1.])
    basis = tangent_basis(n)
    xy = (p-p.mean(0))@basis[:, :2]
    _, _, v = np.linalg.svd(xy, full_matrices=False)
    tangents = basis[:, :2]@v.T
    if np.linalg.det(np.column_stack((tangents, n))) < 0:
        tangents[:, 1] *= -1
    basis = np.column_stack((tangents, n))
    local = p@basis
    lo, hi = local.min(0), local.max(0)
    size = np.maximum(hi-lo, [.035, .035, cfg.min_thickness])
    surface = basis@np.r_[(lo[:2]+hi[:2])/2, np.median(local[:, 2])]
    plane_error = np.max(np.abs((p-surface)@n))
    if plane_error > cfg.plane_tolerance:
        return None, "nonplanar_patch"
    confidence = float(np.mean([events[i]["confidence"] for i in ids]))
    top = surface[2]
    floor = n[2] > .95 and abs(top) < cfg.contact_tolerance
    if floor:
        # World z=0 is the existing simulator ground. Preserve source coordinates.
        surface[2] = 0.
    thickness = max(cfg.min_thickness, top) if n[2] > .98 and top > .04 else cfg.min_thickness
    size[2] = thickness
    role = "ground" if floor else ("platform" if n[2] > .85 else "wall" if abs(n[2]) < .25 else "inclined_support")
    box = make_box(surface-n*thickness/2, basis, size, role, confidence)
    flat_points = points.reshape(-1, 3)
    if not safe_box(flat_points, box, cfg.penetration_tolerance):
        # A table may allow motion below its top: try a slab with checked legs.
        size[2] = cfg.min_thickness
        box = make_box(surface-n*size[2]/2, basis, size, role, confidence)
        if not safe_box(flat_points, box, cfg.penetration_tolerance):
            return None, "support_intersects_robot_sweep"
    # Binary search lateral expansion while preserving non-contact free space.
    low, high = 0., cfg.expansion
    for _ in range(7):
        margin = (low+high)/2
        trial = copy.deepcopy(box)
        trial["size"][:2] = (np.array(box["size"][:2])+2*margin).tolist()
        if safe_box(flat_points, trial, cfg.penetration_tolerance):
            low = margin
        else:
            high = margin
    box["size"][:2] = (np.array(box["size"][:2])+2*low).tolist()
    primitives = [box]
    # Any elevated slab/wall needs geometry reaching the ground. Four corner
    # columns are an explicitly low-confidence completion, not observed truth.
    corners = np.array([[sx,sy,sz] for sx in [-.5,.5] for sy in [-.5,.5] for sz in [-.5,.5]])
    corners = (corners*np.array(box["size"]))@basis.T+box["translation"]
    bottom = float(corners[:,2].min())
    if bottom > .025:
        low_corners = corners[np.argsort(corners[:,2])[:4]]
        legs = []
        for corner in low_corners:
            leg = make_box([corner[0],corner[1],corner[2]/2],np.eye(3),[.025,.025,corner[2]],"inferred_support", .15)
            if not safe_box(flat_points, leg, cfg.penetration_tolerance):
                return None, "no_collision_free_grounded_completion"
            legs.append(leg)
        primitives += legs
    return {"primitives": primitives, "surface_point": surface.tolist(), "surface_normal": n.tolist(),
            "event_ids": ids, "confidence": confidence,
            "structural_completion_is_prior": len(primitives)>1, "kind": role}, None


def reconstruct_static(events, points, model, cfg):
    objects, rejected = [], []
    for cluster in cluster_events(events, cfg):
        for ids in split_empty_regions(cluster, events):
            obj, reason = fit_cluster(ids, events, points, cfg)
            if obj is None:
                rejected.append({"event_ids": ids, "reason": reason})
                for i in ids:
                    events[i]["status"] = "rejected"; events[i]["reason"] = reason
                continue
            obj["object_id"] = len(objects)
            objects.append(obj)
            for i in ids:
                events[i]["object_id"] = obj["object_id"]
                events[i]["status"] = "geometric_candidate"
    # The simulator already has a z=0 ground plane. Represent it explicitly
    # when the motion provides near-ground evidence, rather than losing the
    # floor to DBSCAN splits or treating slow feet as the only support source.
    if np.min(points[...,2]) >= -cfg.penetration_tolerance and np.min(np.abs(points[...,2])) < cfg.contact_tolerance:
        from .contacts import runs
        lo=points.reshape(-1,3).min(0); hi=points.reshape(-1,3).max(0)
        box=make_box([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,-.03],np.eye(3),
                     [hi[0]-lo[0]+.4,hi[1]-lo[1]+.4,.06],"ground",.85)
        oid=len(objects); ids=[]
        # Use every collision body with near-ground surface evidence, including
        # knees, hands and torso. The plane prior is explicitly recorded.
        for s in range(points.shape[1]):
            close=np.abs(points[:,s,2]) < .012
            for a,z in runs(close):
                if z-a < 3: continue
                i=len(events); ids.append(i)
                point=points[a:z,s].mean(0); point[2]=0.
                events.append({"event_id":i,"body_id":int(model.surface.body[s]),"sample_id":s,
                               "start":int(a),"end":int(z),"point":point.tolist(),"normal":[0.,0.,1.],
                               "confidence":.8,"contact_type":1,"status":"geometric_candidate","object_id":oid,
                               "evidence":{"ground_plane_prior":True,"surface_proximity":True}})
        objects.append({"object_id":oid,"primitives":[box],"surface_point":[0.,0.,0.],
                        "surface_normal":[0.,0.,1.],"event_ids":ids,"confidence":.85,
                        "structural_completion_is_prior":False,"kind":"ground"})
    return {"coordinate_frame": "motion_world_z_up_meters", "quaternion_order": "xyzw",
            "objects": objects, "rejected_objects": rejected,
            "ground_z": 0.0, "dynamic_objects": []}


def interaction_graph(scene, events, model, dynamic=None):
    # Keep compound-object association explicit and uncertain: adjacency alone
    # cannot prove that a seat and a backrest belong to one physical chair.
    links = [{"id": i, "name": n, "has_collision": i in model.collision_bodies}
             for i,n in enumerate(model.body_names)]
    edges = [{k:e[k] for k in ("event_id","body_id","object_id","start","end","point","normal","confidence","contact_type","status")}
             for e in events]
    associations = []
    for a,obj in enumerate(scene["objects"]):
        for other in scene["objects"][a+1:]:
            if obj["kind"] == other["kind"]:
                continue
            if np.linalg.norm(np.array(obj["surface_point"])-other["surface_point"]) < .6:
                associations.append({"object_ids": [obj["object_id"],other["object_id"]],
                                     "relation": "possible_compound_support", "confidence": .25})
    dynamic_nodes=[]
    if dynamic:
        for index,h in enumerate(dynamic["hypotheses"]):
            oid=f"unknown_dynamic_{index}"
            dynamic_nodes.append({"object_id":oid,"status":"underdetermined","geometry":None,
                                  "reason":h["reason"],"rigidity_rmse_m":h["rigidity_rmse_m"]})
            for body in h["body_ids"]:
                edges.append({"body_id":body,"object_id":oid,"start":h["start"],"end":h["end"],
                              "confidence":h["confidence"],"contact_type":3,"status":"unknown"})
    return {"robot_nodes": links, "scene_nodes": scene["objects"]+dynamic_nodes,
            "contact_edges": edges, "entity_hypotheses": associations}
