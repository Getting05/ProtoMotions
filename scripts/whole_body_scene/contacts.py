"""Whole-body candidate evidence. Scores are heuristics, not calibrated probabilities."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial.transform import Rotation
from .geometry import unit


@dataclass
class InferenceConfig:
    static_scene_mode: str = "generic"
    samples_per_geom: int = 24
    speed_enter: float = 0.10
    speed_exit: float = 0.22
    normal_speed_enter: float = 0.055
    min_duration_s: float = 0.08
    deceleration: float = 0.65
    min_approach_speed: float = 0.08
    penetration_tolerance: float = 0.008
    contact_tolerance: float = 0.025
    cluster_radius: float = 0.14
    normal_cos: float = 0.88
    plane_tolerance: float = 0.018
    expansion: float = 0.04
    min_thickness: float = 0.035
    friction: float = 0.7
    seed: int = 20261016

    def to_dict(self):
        return asdict(self)


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def detect_static(model, motion, points, normals, rotations, cfg):
    t_count, s_count = points.shape[:2]
    body_v = np.gradient(gaussian_filter1d(motion.pos, 0.7, axis=0), motion.dt, axis=0)
    # World angular velocities: log(R[t+1] R[t]^T) / dt, not derivatives
    # of quaternion components. q and -q therefore give identical velocities.
    delta = rotations[1:] @ rotations[:-1].swapaxes(-1, -2)
    omega = Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec().reshape(t_count-1, -1, 3)/motion.dt
    omega = np.concatenate((omega[:1], omega))
    omega = gaussian_filter1d(omega, 0.7, axis=0)
    b = model.surface.body
    v = body_v[:, b] + np.cross(omega[:, b], points-motion.pos[:, b])
    vn = np.sum(v*normals, axis=-1)
    vt = np.linalg.norm(v-vn[..., None]*normals, axis=-1)
    speed = np.linalg.norm(v, axis=-1)
    an = np.gradient(vn, motion.dt, axis=0)
    # Robot outward normal points toward the environment: approach has vn > 0.
    decel = (an < -cfg.deceleration) & (vn > -0.03)
    recent = maximum_filter1d(decel.astype(np.uint8), size=max(3, int(.18/motion.dt)*2+1),
                              axis=0, origin=max(0, int(.18/motion.dt))) > 0
    ground_z = float(np.min(points[..., 2]))
    near_ground = (np.abs(points[..., 2]) < .025) & (normals[..., 2] < -.7)
    events = []
    for j in range(s_count):
        name = model.body_names[b[j]]
        # Slow pelvis/torso support need a longer dwell, rather than a sharp impact.
        slow_support = any(x in name for x in ("pelvis", "waist", "knee"))
        dwell = .22 if slow_support else cfg.min_duration_s
        low = (np.abs(vn[:, j]) < cfg.normal_speed_enter) & (speed[:, j] < cfg.speed_enter)
        sliding = (np.abs(vn[:, j]) < .035) & (vt[:, j] < .55) & recent[:, j]
        active = False
        starts = np.zeros(t_count, dtype=bool)
        anchor = None
        for t in range(t_count):
            enter = (low[t] and (recent[t, j] or near_ground[t, j] or slow_support or t == 0)) or sliding[t]
            if not active and enter:
                active, anchor = True, points[t, j].copy()
            if active:
                displacement = points[t, j]-anchor
                normal_drift = abs(float(displacement@normals[t, j]))
                if abs(vn[t, j]) > .13 or normal_drift > .04 or (speed[t, j] > cfg.speed_exit and not sliding[t]):
                    active = False
                else:
                    starts[t] = True
        for a, z in runs(starts):
            if (z-a)*motion.dt < dwell:
                continue
            ns = normals[a:z, j]
            normal = unit(np.mean(ns, axis=0))
            if np.min(ns@normal) < cfg.normal_cos:
                continue
            ps = points[a:z, j]
            # Split long sliding patches spatially; no single link-origin proxy.
            span = np.linalg.norm(np.ptp(ps, axis=0))
            pieces = max(1, int(np.ceil(span/.10)))
            for ids in np.array_split(np.arange(a, z), pieces):
                if len(ids) < 2:
                    continue
                p = np.mean(points[ids, j], axis=0)
                before = max(0, int(ids[0])-max(1, int(.25/motion.dt)))
                approach = float(np.max(vn[before:int(ids[-1])+1, j]))
                evidence = {"deceleration": bool(recent[ids, j].any()),
                            "max_approach_speed": approach,
                            "low_speed": float(np.mean(low[ids])),
                            "ground_proximity": bool(near_ground[ids, j].any()),
                            "duration_s": float(len(ids)*motion.dt),
                            "median_speed": float(np.median(speed[ids, j]))}
                confidence = min(.9, .25+.2*evidence["deceleration"]+.25*evidence["low_speed"]+
                                 .15*evidence["ground_proximity"]+.1*min(1, len(ids)*motion.dt/.3))
                # A body held still in free space is indistinguishable from
                # one resting on an unseen support. Keep its evidence UNKNOWN
                # unless a normal approach/stop or the explicit ground prior
                # supports a static contact. Low speed alone never spawns a box.
                supported = evidence["ground_proximity"] or (
                    evidence["deceleration"] and approach >= cfg.min_approach_speed)
                event = {"body_id": int(b[j]), "sample_id": j,
                               "start": int(ids[0]), "end": int(ids[-1]+1),
                               "point": p.tolist(), "normal": (-normal).tolist(),
                               "confidence": confidence,
                               "contact_type": 2 if evidence["median_speed"] > cfg.speed_enter else 1,
                               "evidence": evidence, "status": "candidate" if supported else "unknown", "object_id": -1}
                if not supported:
                    event["reason"] = "stationary_surface_without_observed_contact_transition"
                events.append(event)
    # Spatial/temporal deduplication keeps dense samples from multiplying evidence.
    keep = {}
    for e in events:
        key = (e["body_id"], e["start"]//4, e["end"]//4,
               *np.round(np.asarray(e["point"])/.018).astype(int),
               *np.round(np.asarray(e["normal"])*3).astype(int))
        if key not in keep or keep[key]["confidence"] < e["confidence"]:
            keep[key] = e
    events = list(keep.values())
    if events:
        distance = model.minimum_swept_distance(np.array([e["point"] for e in events]), motion, rotations)
        for i, (e, d) in enumerate(zip(events, distance)):
            e["event_id"] = i
            e["swept_clearance"] = float(d)
            if d < -cfg.penetration_tolerance:
                e["status"] = "rejected"
                e["reason"] = "whole_body_sweep_or_self_intersection"
    return events, {"candidate_events": len(events), "ground_lower_envelope": ground_z,
                    "candidate_bodies": sorted(set(e["body_id"] for e in events)),
                    "scores_are_calibrated_probabilities": False}


def detect_dynamic(model, motion, cfg):
    """Fit bimanual SE(3) hypotheses; never invent mass/shape from two wrists.

    Three non-collinear points derived from each *observed* wrist orientation
    fix a candidate rigid frame. The geometry and existence of a grasp remain
    unobserved, so even a low-residual fit is kept UNKNOWN and not simulated.
    """
    names = model.body_names
    candidates = [i for i,n in enumerate(names) if "wrist_yaw" in n and i in model.collision_bodies]
    if len(candidates) != 2:
        return {"status": "underdetermined", "reason": "no_two_collision_hand_bodies", "hypotheses": []}
    left, right = candidates
    rots = Rotation.from_quat(motion.rot.reshape(-1, 4)).as_matrix().reshape(*motion.rot.shape[:2], 3, 3)
    relative = np.einsum("tji,tj->ti", rots[:, left], motion.pos[:, right]-motion.pos[:, left])
    relrot = rots[:, left].swapaxes(-1,-2)@rots[:, right]
    speed = np.linalg.norm(np.gradient((motion.pos[:, left]+motion.pos[:, right])/2, motion.dt, axis=0), axis=1)
    vel = np.linalg.norm(np.gradient(relative, motion.dt, axis=0), axis=1)
    angular = np.r_[0., np.linalg.norm(Rotation.from_matrix(relrot[1:]@relrot[:-1].swapaxes(-1,-2)).as_rotvec(), axis=1)/motion.dt]
    width = np.linalg.norm(relative, axis=1)
    stable = (vel < .10) & (angular < .8) & (speed > .08) & (width > .12) & (width < .85)
    hypotheses = []
    for a,z in runs(stable):
        if (z-a)*motion.dt < .35:
            continue
        center = (motion.pos[a:z,left]+motion.pos[a:z,right])/2
        trans_resid = float(np.sqrt(np.mean((relative[a:z]-relative[a:z].mean(0))**2)))
        hypotheses.append({"start": int(a), "end": int(z), "body_ids": candidates,
                           "pose_traj": np.concatenate((center,motion.rot[a:z,left]),axis=1).tolist(),
                           "rigidity_rmse_m": trans_resid, "status": "underdetermined",
                           "geometry": None, "mass": None, "friction": None,
                           "confidence": .25,
                           "reason": "rigid_bimanual_motion_does_not_identify_object_shape_grasp_or_mass"})
    return {"status": "underdetermined" if hypotheses else "no_evidence",
            "hypotheses": hypotheses, "rotation_convention": "xyzw",
            "not_a_physical_execution_target": True}
