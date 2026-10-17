"""Global, conservative whole-body scene hypotheses with contact re-estimation.

All effective collision links participate. Structural alternatives are selected
from measured motion, never file names. Unknown geometry stays out of physics.
"""
import copy
import time
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .contacts import runs
from .geometry import unit, tangent_basis
from .planes_v2 import discover_planes, horizontal_foot_windows, patch_area
from .reconstruct import make_box, box_distance, safe_box
from .wrench_v2 import EffortScorer, merge_patches


class SweptSurface:
    """Exact box-interior query on the sampled swept surface, with broad phase.

    This accelerates candidate fitting only. MuJoCo still validates all robot
    primitives at every recorded frame before a scene can be exported.
    """
    def __init__(self, points):
        self.points=np.asarray(points).reshape(-1,3)
        self.tree=cKDTree(self.points)

    def safe(self, box, tolerance):
        center=np.asarray(box['translation']);half=np.asarray(box['size'])/2-tolerance
        if np.any(half<=0):return True
        ids=self.tree.query_ball_point(center,float(np.linalg.norm(half)))
        if not ids:return True
        rotation=Rotation.from_quat(box['rotation']).as_matrix()
        local=(self.points[ids]-center)@rotation
        return not bool(np.any(np.all(np.abs(local)<half,axis=1)))


def traversal_basis(motion):
    forward=motion.pos[-1,0]-motion.pos[0,0];forward[2]=0
    if np.linalg.norm(forward)<.15: forward=np.array([1.,0.,0.])
    forward=unit(forward)
    return np.column_stack((forward,np.cross([0.,0.,1.],forward),[0.,0.,1.]))


def contact_track(distances, normal_velocity, areas, dt):
    """Hysteresis belongs to link/plane patches, allowing rolling sample changes."""
    strong=(distances<=.012)&(np.abs(normal_velocity)<.20)&(areas>=.0002)
    weak=(distances<=.025)&(np.abs(normal_velocity)<.40)
    state=np.zeros(len(distances),bool);active=False
    for t in range(len(state)):
        if not active and strong[t]: active=True
        if active and not weak[t]: active=False
        state[t]=active
    for a,z in runs(state):
        if (z-a)*dt<.08: state[a:z]=False
    return state


def surface_contacts(obj, model, motion, points, normals):
    """Return physical patch tracks and sparse force points for an existing face."""
    normal=np.asarray(obj['surface_normal']);box=obj['primitives'][0]
    point=np.asarray(obj['surface_point']);tracks={}
    force=[[] for _ in range(len(points))]
    for b in model.collision_bodies:
        ids=np.flatnonzero(model.surface.body==b)
        ps,ns=points[:,ids],normals[:,ids]
        plane=(ps-point)@normal
        aligned=(-ns@normal)>=.88
        distance=np.maximum(np.abs(box_distance(ps,box)),np.abs(plane))
        valid=aligned&(distance<=.025)
        nearest=np.where(aligned,distance,np.inf).min(axis=1)
        # The envelope derivative reflects contact-normal motion, including
        # body rotation; it does not require the entire rigid link to stop.
        signed=np.where(aligned,plane,np.inf).min(axis=1)
        finite=np.isfinite(signed)
        if not finite.any(): continue
        signed[~finite]=np.interp(np.flatnonzero(~finite),np.flatnonzero(finite),signed[finite])
        velocity=np.gradient(gaussian_filter1d(signed,.7),motion.dt)
        areas=np.array([patch_area(ps[t,valid[t]],normal) if valid[t].sum()>=3 else 0. for t in range(len(ps))])
        active=contact_track(nearest,velocity,areas,motion.dt)
        if not active.any(): continue
        tracks[int(b)]=(ids,valid&active[:,None],active,areas)
        for t in np.flatnonzero(active):
            p=ps[t,valid[t]]
            if not len(p): continue
            # Spatial extrema preserve moments while avoiding duplicate votes.
            chosen=[int(np.argmin(p@tangent_basis(normal)[:,0]))]
            for _ in range(min(3,len(p)-1)):
                d=np.linalg.norm(p[:,None]-p[chosen][None],axis=-1).min(axis=1)
                chosen.append(int(np.argmax(d)))
            for k in np.unique(chosen):
                projected=p[k]-normal*np.dot(p[k]-point,normal)
                force[t].append((projected,normal,int(b)))
    return tracks,force


def ground_object(points):
    lo=points.min(axis=(0,1));hi=points.max(axis=(0,1))
    return dict(object_id=0,kind='ground',primitives=[make_box(
        [(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,-.03],np.eye(3),[hi[0]-lo[0]+.8,hi[1]-lo[1]+.8,.06],'ground',.95)],
        surface_point=[0.,0.,0.],surface_normal=[0.,0.,1.],confidence=.95,
        event_ids=[],structural_completion_is_prior=False)


def fit_plane(group, sweep, basis, cfg):
    """Try connected solids / a broad pedestal, never four invented thin legs."""
    n=group['normal'];p=group['points']
    if n[2]>.99:
        rot=basis;local=p@rot;lo=local[:,:2].min(0)-.025;hi=local[:,:2].max(0)+.025
        extent=np.maximum(hi-lo,[.18,.18]);center=(lo+hi)/2
        footprints=[(center,extent)]
        # A union bounding box often spans the space swept by the other thigh.
        # Search observed subregions instead of rejecting the entire seat or
        # compensating with thin columns. Every crop must retain patch area.
        for fraction in (.75,.5):
            for axis in (0,1):
                size=extent.copy();size[axis]=max(.12,extent[axis]*fraction)
                for direction in (-1,0,1):
                    c=center.copy();c[axis]+=direction*(extent[axis]-size[axis])/2
                    inside=np.all(np.abs(local[:,:2]-c)<=size/2,axis=1)
                    if patch_area(p[inside],n)>=.0025:footprints.append((c,size))
        for retreat in np.arange(0.,.0201,.004):
            for center,extent in footprints:
                result=fit_horizontal_footprint(group,sweep,rot,center,extent,retreat,cfg)
                if result is not None:return result,None
        return None,'no_swept_safe_connected_horizontal_support'
    if abs(n[2])<.01:
        u=unit(np.cross([0.,0.,1.],n));rot=np.column_stack((u,[0.,0.,1.],n));local=p@rot
        width=max(.35,float(np.ptp(local[:,0]))+.12)
        top=max(.35,float(local[:,1].max())+.10)
        for retreat in np.arange(0.,.0201,.004):
            offset=group['offset']-retreat
            center=rot@np.array([np.median(local[:,0]),top/2,offset-.035])
            box=make_box(center,rot,[width,top,.07],'wall',.6)
            if sweep.safe(box,cfg.penetration_tolerance-.002):
                surface=center+n*.035
                return dict(kind='wall',primitives=[box],surface_point=surface.tolist(),surface_normal=n.tolist(),
                    confidence=.6,event_ids=[],structural_completion_is_prior=True,
                    normal_retreat_m=float(retreat),candidate_id=group['candidate_id']),None
        return None,'no_swept_safe_wall'
    # Inclined planes are retained as evidence until a connected solid can be
    # fit. A small tilted patch is not promoted into a freestanding obstacle.
    return None,'inclined_surface_requires_connected_volume'


def fit_horizontal_footprint(group,sweep,rot,center,extent,retreat,cfg):
    n=group['normal'];height=group['offset']-retreat
    if height<.035:return None
    surface=rot@np.r_[center,height]
    solid=make_box(rot@np.r_[center,height/2],rot,[*extent,height],'support_volume',.65)
    alternatives=[[solid]]
    slab=make_box(surface-n*.02,rot,[*extent,.04],'support_surface',.65)
    if height>.10:
        pedestal=make_box(rot@np.r_[center,(height-.04)/2],rot,
            [min(extent[0],.16),min(extent[1],.16),height-.04],'structural_prior_pedestal',.25)
        alternatives.append([slab,pedestal])
    for primitives in alternatives:
        if all(sweep.safe(box,cfg.penetration_tolerance-.002) for box in primitives):
            return dict(kind='platform',primitives=primitives,surface_point=surface.tolist(),
                surface_normal=n.tolist(),confidence=.65,event_ids=[],
                structural_completion_is_prior=True,normal_retreat_m=float(retreat),
                candidate_id=group['candidate_id'])
    return None


def fit_ladder(groups, model, motion, points, basis, cfg, sweep):
    feet={model.body_names.index(n) for n in ('left_ankle_roll_link','right_ankle_roll_link')}
    candidates=[]
    for group in groups:
        if group['normal'][2]<.99:continue
        for patch in group['patches']:
            if patch['body_id'] not in feet or patch['offset']<.08:continue
            candidates.append(dict(offset=patch['offset'],points=patch['points'],
                candidate_id=group['candidate_id'],duration=patch['duration_s']))
    clusters=[]
    for candidate in sorted(candidates,key=lambda g:g['offset']):
        if clusters and candidate['offset']-clusters[-1][0]['offset']<.10:
            clusters[-1].append(candidate)
        else:clusters.append([candidate])
    levels=[max(cluster,key=lambda p:p['duration']) for cluster in clusters]
    if len(levels)<3 or levels[-1]['offset']-levels[0]['offset']<.45:return [],'insufficient_ladder_levels'
    heights=np.array([g['offset'] for g in levels])
    centers=np.array([np.median(g['points']@basis,axis=0) for g in levels])
    slope,intercept=np.polyfit(heights,centers[:,0],1)
    if np.max(np.abs(slope*heights+intercept-centers[:,0]))>.12:return [],'ladder_levels_do_not_align'
    foot_points=points[:,np.isin(model.surface.body,list(feet))]@basis
    lo=float(foot_points[...,1].min())-.09;hi=float(foot_points[...,1].max())+.09
    objects=[];best_intercept=intercept
    # Shared front plane: all rungs must meet the same two side rails.
    for dx in (0.,.04,-.04,.08,-.08,.12,-.12,.16,-.16,.20,-.20):
        proposal=[]
        for g,h in zip(levels,heights):
            found=None
            near=g['points']@basis
            for depth in (.10,.06,.035):
                if found is not None:break
                inside=np.abs(near[:,0]-(slope*h+intercept+dx))<depth/2+.01
                if patch_area(g['points'][inside],[0,0,1])<.0005:continue
                for retreat in (0.,.006,.012,.018):
                    center=basis@np.array([slope*h+intercept+dx,(lo+hi)/2,h-retreat-.015])
                    box=make_box(center,basis,[depth,hi-lo,.03],'ladder_rung',.6)
                    if not sweep.safe(box,cfg.penetration_tolerance-.002):continue
                    surface=center.copy();surface[2]=h-retreat
                    found=dict(kind='ladder_rung',primitives=[box],surface_point=surface.tolist(),
                        surface_normal=[0.,0.,1.],confidence=.6,event_ids=[],structural_completion_is_prior=True,
                        normal_retreat_m=retreat,candidate_id=g['candidate_id'])
                    break
            if found is not None:proposal.append(found)
        if len(proposal)>len(objects):objects=proposal;best_intercept=intercept+dx
    intercept=best_intercept
    if len(objects)<3: return [],'too_few_collision_free_rungs'
    rails=[];top=max(o['surface_point'][2] for o in objects)+.15
    axis=unit(basis@np.array([slope,0,1.]));u=basis[:,1];v=np.cross(axis,u)
    rot=np.column_stack((u,v,axis))
    for y in [lo+.02,hi-.02]:
        a=basis@np.array([intercept,y,.03]);b=basis@np.array([slope*top+intercept,y,top])
        rail=make_box((a+b)/2,rot,[.06,.06,np.linalg.norm(b-a)],'structural_prior_rail',.3)
        if not sweep.safe(rail,cfg.penetration_tolerance-.002):return [],'ladder_frame_intersects_sweep'
        rails.append(rail)
    objects[0]['primitives']+=rails
    return objects,None


def reconstruct_v2(model,motion,points,normals,cfg):
    groups,diagnostics=discover_planes(model,motion,points,normals,cfg)
    sweep=SweptSurface(points)
    basis=traversal_basis(motion)
    scene={'coordinate_frame':'motion_world_z_up_meters','quaternion_order':'xyzw',
           'objects':[ground_object(points)],'rejected_objects':[],'ground_z':0.,'dynamic_objects':[]}
    report={'version':2,'method':'temporal_planes_connected_solids_wrench_selection',
            'all_collision_bodies_considered':model.collision_bodies.tolist(),
            'candidate_planes':diagnostics,'model_selection':[],
            'automatic_thin_columns':False,'filename_used_as_evidence':False,
            'ground_truth_geometry_available':False}
    scene['scene_inference']=report
    windows=horizontal_foot_windows(groups,model,motion,points,normals)
    delta=motion.pos[-1,0]-motion.pos[0,0];travel=np.linalg.norm(delta[:2])
    family='generic'
    if delta[2]>.35 and travel>.3 and delta[2]/travel<1.5:
        from .stairs import reconstruct_stairs
        scene=reconstruct_stairs(scene,[],points,normals,model,motion,cfg,support_intervals=windows)
        if sum(o['kind']=='stair_tread' for o in scene['objects'])>=3:family='stairs'
        scene.get('stairs_inference',{}).update(support_window_rule={
            'method':'deterministic_envelope_and_rolling_patch_area','minimum_patch_area_m2':.0005,
            'minimum_duration_s':cfg.min_duration_s,'random_point_vote_used':False})
    if delta[2]>.5 and (travel<.3 or delta[2]/max(travel,.01)>=1.5):
        ladder,reason=fit_ladder(groups,model,motion,points,basis,cfg,sweep)
        report['ladder_hypothesis']=reason or 'supported'
        if ladder:
            for o in ladder:o['object_id']=len(scene['objects']);scene['objects'].append(o)
            family='ladder'
    report['structural_family']=family
    print(f'v2: {len(groups)} plane candidates; family={family}',flush=True)
    frame_ids=np.unique(np.linspace(0,len(points)-1,min(48,len(points))).astype(int))
    scorer=EffortScorer(model,motion,frame_ids,cfg.friction)
    report['selection_objective']='normalized_root_wrench_squared_plus_0.1_normalized_joint_effort_squared'
    patches=[[] for _ in range(len(points))]
    for obj in scene['objects']:
        _,p=surface_contacts(obj,model,motion,points,normals);patches=merge_patches(patches,p)
    errors=scorer.score(patches);report['initial_selection_cost']=float(errors[frame_ids].mean())
    feet={model.body_names.index(n) for n in ('left_ankle_roll_link','right_ankle_roll_link')}
    pool=[]
    fitting_start=time.monotonic()
    for g in groups:
        # Existing multi-level structural models explain these foot planes.
        if family!='generic' and g['normal'][2]>.99 and set(g['bodies'])&feet:continue
        if not g['arrival']:
            report['model_selection'].append(dict(candidate_id=g['candidate_id'],status='unknown',reason='no_observed_arrival_before_contact'));continue
        if abs(g['normal'][2])<.01 and set(g['bodies']).issubset(feet):
            report['model_selection'].append(dict(candidate_id=g['candidate_id'],status='unknown',reason='sole_side_proximity_does_not_establish_wall'));continue
        if not set(g['bodies'])&feet and (g['duration_s']<.30 or g['area_m2']<.0025):
            report['model_selection'].append(dict(candidate_id=g['candidate_id'],status='unknown',reason='short_or_small_nonfoot_patch'));continue
        obj,reason=fit_plane(g,sweep,basis,cfg)
        if obj is None:
            report['model_selection'].append(dict(candidate_id=g['candidate_id'],status='unknown',reason=reason));continue
        tracks,p=surface_contacts(obj,model,motion,points,normals)
        active=sum(bool(x) for x in p)
        if active*motion.dt<.20:
            report['model_selection'].append(dict(candidate_id=g['candidate_id'],status='unknown',reason='no_persistent_contact_after_geometry_fit'));continue
        pool.append((obj,p,g))
    print(f'v2: {len(pool)} connected candidates fitted in {time.monotonic()-fitting_start:.1f}s',flush=True)
    # Greedy minimum-description-length selection with re-scoring after each
    # addition. Supporting the same frames twice earns no extra reward.
    while pool:
        scored=[]
        for obj,p,g in pool:
            trial=scorer.score(merge_patches(patches,p))
            gain=float((errors[frame_ids]-trial[frame_ids]).mean())
            cost=.012+.002*(len(obj['primitives'])-1)
            scored.append((gain-cost,gain,trial,obj,p,g))
        best=max(scored,key=lambda x:x[0])
        if best[0]<=0:
            for _,gain,_,obj,p,g in scored:
                report['model_selection'].append(dict(candidate_id=g['candidate_id'],status='unknown',
                    reason='insufficient_marginal_support_gain',wrench_gain=gain))
            break
        _,gain,errors,obj,p,g=best
        print(f'v2: select {g["candidate_id"]} {obj["kind"]}, wrench gain={gain:.4f}',flush=True)
        obj['object_id']=max(o['object_id'] for o in scene['objects'])+1
        obj['selection_wrench_gain']=gain;scene['objects'].append(obj)
        patches=merge_patches(patches,p)
        report['model_selection'].append(dict(candidate_id=g['candidate_id'],status='selected',
            object_id=obj['object_id'],wrench_gain=gain))
        pool=[x for x in pool if x[2]['candidate_id']!=g['candidate_id']]
    report['final_selection_cost']=float(errors[frame_ids].mean())
    report['selection_frame_ids']=frame_ids.tolist()
    return scene,groups


def events_from_scene(scene,model,motion,points,normals):
    """Re-estimate contact windows on the final collision-validated geometry."""
    events=[]
    for obj in scene['objects']:
        obj['event_ids']=[]
        tracks,_=surface_contacts(obj,model,motion,points,normals)
        for body,(ids,valid,active,areas) in tracks.items():
            for j,sample in enumerate(ids):
                for a,z in runs(valid[:,j]):
                    if z-a<1:continue
                    eid=len(events);obj['event_ids'].append(eid)
                    normal=np.array(obj['surface_normal']);p=points[a:z,sample].mean(axis=0)
                    p-=normal*np.dot(p-np.array(obj['surface_point']),normal)
                    events.append(dict(event_id=eid,body_id=body,sample_id=int(sample),start=int(a),end=int(z),
                        point=p.tolist(),normal=normal.tolist(),confidence=obj['confidence'],contact_type=1,
                        status='geometry_validated',object_id=obj['object_id'],
                        evidence={'scene_conditioned_patch_track':True,'rolling_sample_identity_allowed':True,
                                  'minimum_body_patch_dwell_s':.08,'median_patch_area_m2':float(np.median(areas[a:z]))}))
    return events
