"""Evidence-conditioned stair profiles with a whole-body swept-volume bound.

This is an explicit structural prior: horizontal, full-width, grounded treads.
Only observed foot-support levels are instantiated; missing levels are not filled.
The original whole-body detector and exact all-frame validator remain in use.
"""
import numpy as np

from .contacts import runs
from .reconstruct import make_box


def support_windows(events, body_id, frames, dt):
    evidence = [e for e in events if e["body_id"] == body_id
                and e["normal"][2] > .9 and e["point"][2] > .04
                and e["evidence"].get("deceleration", False)
                and e["evidence"].get("max_approach_speed", 0.) >= .08]
    counts = np.zeros(frames, dtype=int)
    for e in evidence:
        counts[e["start"]:e["end"]] += 1
    if not counts.max():
        return []
    # Consensus may be shorter than the individual >=120 ms surface events.
    return [(int(a), int(z)) for a, z in runs(counts >= max(3, counts.max() * .25))
            if (z - a) * dt >= .08]


def safe_riser(points_local, height, lateral_lo, lateral_hi, tolerance):
    """Forwardmost lower robot surface bounds a grounded step's leading face."""
    below = points_local[(points_local[:, 2] < height - tolerance)
                         & (points_local[:, 1] >= lateral_lo)
                         & (points_local[:, 1] <= lateral_hi)]
    if not len(below):
        raise ValueError("A stair hypothesis needs an observed lower approach")
    return float(below[:, 0].max()) + .008


def reconstruct_stairs(scene, events, points, normals, model, motion, cfg):
    names = model.body_names
    feet = [names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")
            if n in names]
    delta = motion.pos[-1, 0] - motion.pos[0, 0]
    delta[2] = 0.
    report = {"structural_prior": "horizontal grounded treads across observed foot corridor",
              "width_is_observed_corridor_plus_margin": True,
              "unobserved_levels_filled": False, "accepted_levels": [], "rejected_levels": []}
    scene["stairs_inference"] = report
    if len(feet) != 2 or np.linalg.norm(delta) < .3:
        report["status"] = "insufficient_ascending_motion"
        return scene
    forward = delta / np.linalg.norm(delta)
    lateral = np.cross([0., 0., 1.], forward)
    basis = np.column_stack((forward, lateral, [0., 0., 1.]))
    local = points @ basis
    foot_mask = np.isin(model.surface.body, feet)
    lo = float(local[:, foot_mask, 1].min()) - .04
    hi = float(local[:, foot_mask, 1].max()) + .04
    report.update(forward=forward.tolist(), lateral_bounds=[lo, hi])
    hypotheses = []
    for body in feet:
        mask = model.surface.body == body
        foot = local[:, mask]
        for a, z in support_windows(events, body, len(points), motion.dt):
            original_height = float(np.quantile(foot[a:z, :, 2].min(axis=1), .1)) - .002
            if original_height <= .04:
                continue
            center = float(np.median(motion.pos[a:z, body] @ forward))
            best = None
            for retreat in np.linspace(0., cfg.contact_tolerance, 6):
                height = original_height - retreat
                leading = safe_riser(local.reshape(-1, 3), height, lo, hi, cfg.penetration_tolerance)
                contact = (np.abs(foot[a:z, :, 2] - height) <= cfg.contact_tolerance)
                contact &= foot[a:z, :, 0] >= leading + .002
                frame_coverage = float(np.mean(contact.sum(axis=1) >= 3))
                if frame_coverage >= .5 and leading <= center + .04:
                    best = {"body_id": body, "start": a, "end": z, "height": height,
                            "leading": leading, "center": center,
                            "foot_forward_max": float(foot[a:z, :, 0].max()),
                            "normal_retreat_m": float(retreat),
                            "contact_frame_fraction": frame_coverage}
                    break
            if best is None:
                report["rejected_levels"].append({"body_id": body, "start": a, "end": z,
                    "height": original_height, "reason": "no_swept_safe_supported_tread"})
            else:
                hypotheses.append(best)
    hypotheses.sort(key=lambda h: h["center"])
    accepted = []
    for h in hypotheses:
        if accepted and (h["height"] <= accepted[-1]["height"] + .04
                         or h["leading"] <= accepted[-1]["leading"] + .06):
            report["rejected_levels"].append(dict(h, reason="nonascending_or_insufficient_tread_depth"))
            continue
        accepted.append(h)
    if len(accepted) < 3:
        report["status"] = "insufficient_observed_levels"
        report["rejected_levels"].extend(dict(h, reason="insufficient_observed_levels") for h in accepted)
        return scene
    # Replace foot-only off-ground generic patches. Keep all non-foot evidence
    # and the ground; no global change to the full-body candidate detector.
    removed = {o["object_id"] for o in scene["objects"] if o["kind"] != "ground"
               and o["event_ids"] and all(events[i]["body_id"] in feet for i in o["event_ids"])}
    scene["objects"] = [o for o in scene["objects"] if o["object_id"] not in removed]
    for e in events:
        if e["object_id"] in removed:
            e.update(object_id=-1, status="unknown", reason="replaced_by_stair_support_prior")
    next_id = max([o["object_id"] for o in scene["objects"]] + list(removed) + [-1]) + 1
    for index, h in enumerate(accepted):
        end = accepted[index + 1]["leading"] if index + 1 < len(accepted) else h["foot_forward_max"] + .03
        if end - h["leading"] < .06:
            report["rejected_levels"].append(dict(h, reason="insufficient_tread_depth"))
            continue
        center = basis @ np.array([(h["leading"] + end) / 2, (lo + hi) / 2, h["height"] / 2])
        box = make_box(center, basis, [end - h["leading"], hi - lo, h["height"]], "stair_tread", .65)
        oid = next_id
        next_id += 1
        a, z, body = h["start"], h["end"], h["body_id"]
        ids = []
        for sample in np.flatnonzero(model.surface.body == body):
            foot = local[a:z, sample]
            valid = ((np.abs(foot[:, 2] - h["height"]) <= cfg.contact_tolerance)
                     & (foot[:, 0] >= h["leading"]) & (foot[:, 0] <= end)
                     & (-normals[a:z, sample, 2] >= cfg.normal_cos))
            if not valid.any():
                continue
            eid = len(events)
            ids.append(eid)
            point = points[a:z, sample][valid].mean(axis=0)
            point[2] = h["height"]
            events.append({"event_id": eid, "body_id": body, "sample_id": int(sample),
                "start": a, "end": z, "point": point.tolist(), "normal": [0., 0., 1.],
                "confidence": .65, "contact_type": 1, "status": "geometric_candidate", "object_id": oid,
                "evidence": {"stair_structure_prior": True, "observed_normal_stop_consensus": True,
                             "normal_retreat_m": h["normal_retreat_m"]}})
        surface = center.copy()
        surface[2] = h["height"]
        scene["objects"].append({"object_id": oid, "kind": "stair_tread", "primitives": [box],
            "surface_point": surface.tolist(), "surface_normal": [0., 0., 1.],
            "event_ids": ids, "confidence": .65, "structural_completion_is_prior": True,
            "normal_retreat_m": h["normal_retreat_m"],
            "observed_support": dict(h)})
        report["accepted_levels"].append(dict(h, object_id=oid, trailing=end))
    report["status"] = "candidate_requires_exact_validation"
    return scene
