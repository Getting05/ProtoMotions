"""Isaac Lab 3 / PhysX boundary. Import only after AppLauncher.

Uses version-specific raw contact buffers deliberately: averaged contact positions
can hide foot rotation/slip. Missing APIs fail rather than silently change labels.
"""
import torch
import warp as wp
from protomotions.simulator.isaaclab.simulator import IsaacLabSimulator
from .physics import FEET,ANCHOR,quat_matrix,mv
from .sensors import IMUSynthesizer,point_velocity,pelvis_target,reliable_contact


def as_torch(value):
    if isinstance(value,torch.Tensor): return value
    return wp.to_torch(value)


class EstimatorIsaacLabSimulator(IsaacLabSimulator):
    def _apply_root_velocity_impulse(self,linear_velocity,angular_velocity,env_ids):
        super()._apply_root_velocity_impulse(linear_velocity,angular_velocity,env_ids)
        if not hasattr(self,'_estimator_push_mask'):
            self._estimator_push_mask=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self._estimator_push_mask[env_ids]=True

    def _get_scene_cfg(self):
        cfg=super()._get_scene_cfg()
        for foot in FEET:
            sensor=getattr(cfg,'contact_sensor_'+foot)
            sensor.filter_prim_paths_expr=['/World/ground/terrain/mesh']
            sensor.track_contact_points=True
            sensor.max_contact_data_count_per_prim=128
        return cfg


class IsaacObservationReader:
    def __init__(self,simulator,physics,dt,contact_config):
        self.sim,self.physics=simulator,physics
        self.imu=IMUSynthesizer(physics,dt)
        self.contact_config=contact_config
        names=list(simulator._robot.body_names)
        self.indices={n:names.index(n) for n in ('pelvis',ANCHOR,*FEET)}
        self.previous_support=None
        if not simulator.terrain.is_flat():
            raise ValueError('This estimator supports flat terrain only')

    def reset(self,ids=None):
        self.imu.reset(ids)
        if self.previous_support is not None:
            if ids is None: self.previous_support.zero_()
            else: self.previous_support[ids]=False

    def _contact(self,foot,pos,vel,omega):
        sensor=self.sim._contact_sensor_map[foot]
        data=sensor.data  # triggers current-frame update before reading raw buffers
        force=as_torch(data.force_matrix_w)
        if force.shape[1:3] != (1,1):
            raise RuntimeError('Expected one foot body and one filtered ground shape')
        normal=force[:,0,0,2].clamp_min(0)
        points=as_torch(sensor._contact_points_flat)
        counts=as_torch(sensor._contact_counts).reshape(-1).long()
        starts=as_torch(sensor._contact_start_indices).reshape(-1).long()
        if counts.shape != normal.shape:
            raise RuntimeError('Unexpected contact manifold layout; check Isaac version')
        if counts.sum()>=len(points):
            raise RuntimeError('Contact buffer may be saturated; increase capacity')
        max_count=int(counts.max().item())
        if max_count==0:
            return normal,torch.full_like(normal,float('inf')),counts,torch.full_like(pos,float('nan'))
        offsets=torch.arange(max_count,device=counts.device)
        valid=offsets[None,:]<counts[:,None]
        index=(starts[:,None]+offsets).clamp(0,len(points)-1)
        contact_points=points[index]
        vp=point_velocity(vel[:,None],omega[:,None].expand_as(contact_points),pos[:,None],contact_points)
        slip=vp[...,:2].norm(dim=-1).masked_fill(~valid,0).max(-1).values
        mean=(contact_points.masked_fill(~valid[...,None],0).sum(1)/counts.clamp_min(1)[:,None])
        slip=slip.masked_fill(counts==0,float('inf'))
        mean=mean.masked_fill((counts==0)[:,None],float('nan'))
        return normal,slip,counts,mean

    @torch.no_grad()
    def read(self,previous_pd_target):
        dof=self.sim.get_dof_state()
        torque=self.sim.get_dof_forces().dof_forces
        data=self.sim._robot.data
        positions=as_torch(data.body_link_pos_w)
        rotations=quat_matrix(as_torch(data.body_link_quat_w))
        velocities=as_torch(data.body_link_lin_vel_w)
        angular=as_torch(data.body_ang_vel_w)
        root=self.indices['pelvis']; anchor=self.indices[ANCHOR]
        p,r,v,w=positions[:,root],rotations[:,root],velocities[:,root],angular[:,root]
        gravity,gyro,accel,valid=self.imu.sample(r,v,w)
        proprio=torch.cat((dof.dof_pos,dof.dof_vel,torque,gravity,gyro,accel,previous_pd_target),-1)
        ground=self.sim.terrain.get_ground_heights(p).reshape(-1)
        target=pelvis_target(p,v,rotations[:,anchor],ground)
        normal,slip,counts,contact_points=[],[],[],[]
        for foot in FEET:
            i=self.indices[foot]
            values=self._contact(foot,positions[:,i],velocities[:,i],angular[:,i])
            for array,value in zip((normal,slip,counts,contact_points),values): array.append(value)
        normal,slip,counts,contact_points=[torch.stack(x,1) for x in (normal,slip,counts,contact_points)]
        contact=reliable_contact(normal,slip,counts>0,**self.contact_config)
        support=(normal>self.contact_config['force_min']).any(-1)
        landing=torch.zeros_like(support) if self.previous_support is None else support&~self.previous_support
        self.previous_support=support.clone()
        candidates,details=self.physics(proprio,details=True)
        # Independent simulator truth evaluated at the SAME selected material point.
        actual_points=[]; actual_vel=[]
        for f,foot in enumerate(FEET):
            i=self.indices[foot]
            cp=positions[:,i]+mv(rotations[:,i],details['points'][:,f])
            actual_points.append(cp)
            actual_vel.append(point_velocity(velocities[:,i],angular[:,i],positions[:,i],cp))
        actual_points=torch.stack(actual_points,1); actual_vel=torch.stack(actual_vel,1)
        offset=mv(r[:,None],details['positions'])
        predicted_points=p[:,None]+offset
        relative_velocity=mv(details['jacobians'],dof.dof_vel[:,None])
        predicted_velocity=v[:,None]+torch.linalg.cross(w[:,None].expand_as(offset),offset)+mv(r[:,None],relative_velocity)
        audit=torch.stack(((predicted_points-actual_points).abs().flatten(1).max(-1).values,
                           (predicted_velocity-actual_vel).abs().flatten(1).max(-1).values),-1)
        return dict(proprio=proprio,target=target,contact=contact,valid=valid,
                    normal_force=normal,slip_speed=slip,contact_count=counts,
                    contact_points=contact_points,airborne=~support,landing=landing,
                    push=getattr(self.sim,'_estimator_push_mask',torch.zeros_like(support)).clone(),
                    slip=((normal>self.contact_config['force_min'])&(slip>=self.contact_config['slip_max'])).any(-1),
                    fk_audit=audit)
