"""Whole-body temporal support envelopes, independent of random point counts."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull, QhullError
from .contacts import runs
from .geometry import unit, tangent_basis


def patch_area(points, normal):
    """Area of the spatially occupied patch, not number of surface samples."""
    if len(points) < 3:
        return 0.
    xy = np.asarray(points) @ tangent_basis(normal)[:, :2]
    # Centred spatial quantization bounds the effect of oversampling curves.
    _, ids = np.unique(np.round(xy / .005), axis=0, return_index=True)
    xy = xy[ids]
    if len(xy) < 3: return 0.
    try: return float(ConvexHull(xy).volume)
    except QhullError: return 0.


def stable_intervals(values, dt, min_duration=.08, speed=.14, drift=.022):
    smooth = gaussian_filter1d(values, .7)
    velocity = np.gradient(smooth, dt)
    low = np.isfinite(values) & (np.abs(velocity) <= speed)
    output = []
    for a, z in runs(low):
        begin = int(a)
        for end in range(begin + 1, int(z) + 1):
            if np.ptp(values[begin:end]) > drift:
                if (end - 1 - begin) * dt >= min_duration:
                    output.append((begin, end - 1))
                begin = end - 1
        if (z - begin) * dt >= min_duration: output.append((begin, int(z)))
    return output, velocity


def discover_planes(model, motion, points, normals, cfg):
    directions = [np.array([0., 0., 1.])]
    # Vertical and inclined surfaces are hypotheses, not category labels.
    for z in (0., .5):
        for a in np.arange(12) * np.pi / 6:
            directions.append(np.array([np.cos(a)*np.sqrt(1-z*z), np.sin(a)*np.sqrt(1-z*z), z]))
    patches, diagnostics = [], []
    for body in model.collision_bodies:
        mask = model.surface.body == body
        ps, ns = points[:, mask], normals[:, mask]
        for direction in directions:
            projection = ps @ direction
            aligned = (-ns @ direction) >= .65
            if not np.all(aligned.any(axis=1)): continue
            envelope = np.where(aligned, projection, np.inf).min(axis=1)
            horizontal = bool(direction[2] > .99)
            intervals, velocity = stable_intervals(envelope, motion.dt,
                min_duration=cfg.min_duration_s if horizontal else .20,
                speed=.14 if horizontal else .10)
            for a, z in intervals:
                height = float(np.quantile(envelope[a:z], .1)) - .002
                if horizontal and height < .035: continue  # Explicit shared ground.
                contact = (np.abs(projection[a:z] - height) <= .016) & aligned[a:z]
                if not contact.any(): continue
                samples = ps[a:z][contact]
                area = patch_area(samples, direction)
                if area < .0005: continue
                before = max(0, a - int(np.ceil(.35/motion.dt)))
                after = min(len(points), z + int(np.ceil(.35/motion.dt)))
                excursion = max(float(envelope[before:z].max()-height),
                                float(envelope[a:after].max()-height))
                approach = float(max(0., -velocity[before:z].min()))
                leave = float(max(0., velocity[a:after].max()))
                arrival = a>before and float(-velocity[before:a].min())>=.12 and float(envelope[before:a].max()-height)>=.02
                transition = excursion >= .018 and max(approach, leave) >= .10
                row = dict(body_id=int(body), start=a, end=z, normal=direction.tolist(),
                    offset=height, center=np.median(samples, axis=0).tolist(),
                    area_m2=area, duration_s=(z-a)*motion.dt,
                    observed_transition=transition, normal_excursion_m=excursion,
                    approach_m_s=approach, departure_m_s=leave,
                    observed_arrival=bool(arrival),
                    points=samples, horizontal=horizontal)
                patches.append(row)
    # Merge nearby coplanar patches across links and time. Plane identity and
    # support duration survive; unrelated stopped samples cannot vote repeatedly.
    groups = []
    for patch in sorted(patches, key=lambda p: p['area_m2']*p['duration_s'], reverse=True):
        n = np.array(patch['normal']); c = np.array(patch['center'])
        match = None
        for g in groups:
            if np.dot(n,g['normal']) < .985 or abs(patch['offset']-g['offset']) > .018:
                continue
            if min(np.linalg.norm(c-np.array(q['center'])) for q in g['patches']) <= .45:
                match = g; break
        if match is None:
            groups.append(dict(normal=n,offset=patch['offset'],patches=[patch]))
        else: match['patches'].append(patch)
    for index, g in enumerate(groups):
        # Robust lower envelope rather than averaging a plane into the body.
        g['offset'] = float(np.quantile([p['offset'] for p in g['patches']], .25))
        g['points'] = np.concatenate([p['points'] for p in g['patches']])
        g['area_m2'] = patch_area(g['points'],g['normal'])
        g['bodies'] = sorted({p['body_id'] for p in g['patches']})
        g['transition'] = any(p['observed_transition'] for p in g['patches'])
        g['arrival'] = any(p['observed_arrival'] for p in g['patches'])
        g['duration_s'] = max(p['duration_s'] for p in g['patches'])
        g['candidate_id'] = index
        diagnostics.append({k:v for k,v in g.items() if k not in ('points','patches','normal')}
                           | {'normal':g['normal'].tolist(),
                              'windows':[{k:v for k,v in p.items() if k!='points'} for p in g['patches']]})
    return groups, diagnostics


def horizontal_foot_windows(groups, model, motion, points=None, normals=None):
    """Per-foot physical-area support intervals, with no global peak vote."""
    output = {}
    for name in ('left_ankle_roll_link','right_ankle_roll_link'):
        if name not in model.body_names: continue
        b = model.body_names.index(name)
        windows = sorted({(p['start'],p['end']) for g in groups if g['normal'][2]>.99
                          for p in g['patches'] if p['body_id']==b and p['area_m2']>=.0005})
        if points is not None:
            ids=model.surface.body==b;ps=points[:,ids];ns=normals[:,ids]
            lowest=ps[:,:,2].min(axis=1)
            vz=np.gradient(gaussian_filter1d(ps[:,:,2],.7,axis=0),motion.dt,axis=0)
            # A rolling sole can exchange its lowest point while a finite
            # physical patch remains still. Track that patch, not the changing
            # identity or derivative of the one instantaneous lowest point.
            stable=(np.abs(vz)<=.15)&(ns[:,:,2]<=-.88)&(ps[:,:,2]<=lowest[:,None]+.025)
            areas=np.array([patch_area(ps[t,stable[t]],[0,0,1]) for t in range(len(ps))])
            for a,z in runs(areas>=.0005):
                begin=int(a)
                for end in range(begin+1,int(z)+1):
                    if np.ptp(lowest[begin:end])>.045:
                        if (end-1-begin)*motion.dt>=.08:windows.append((begin,end-1))
                        begin=end-1
                if (z-begin)*motion.dt>=.08:windows.append((begin,int(z)))
        windows=sorted(set(windows))
        output[b] = windows
    return output
