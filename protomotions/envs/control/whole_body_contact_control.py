"""Static scene contact conditioning; free dynamics are never overwritten."""
from dataclasses import dataclass
import torch
from protomotions.components.motion_scene_contact import MotionSceneContact
from protomotions.envs.context_views import WholeBodyContactContext
from protomotions.envs.control.base import ControlComponent,ControlComponentConfig
from protomotions.utils import rotations


@dataclass
class WholeBodyContactConfig(ControlComponentConfig):
    _target_:str="protomotions.envs.control.whole_body_contact_control.WholeBodyContactControl"
    association_distance:float=.055
    max_objects_per_link:int=4


class WholeBodyContactControl(ControlComponent):
    def __init__(self,config,env):
        super().__init__(config,env)
        binding=getattr(env.scene_lib,"motion_scene_binding",None)
        if binding is None: raise ValueError("Whole-body contact control requires a verified motion-scene binding")
        ids=env.scene_lib.get_per_env_humanoid_motion_ids_tensor()
        if bool((ids<0).any()): raise ValueError("Unpaired flat environments are forbidden for scene-bound motions")
        self.fixed_ids=ids.to(env.device)
        self.reference=MotionSceneContact(binding["manifest"],env.motion_lib,ids.cpu().tolist(),
            env.robot_config.kinematic_info.body_names,env.device,config.max_objects_per_link)

    def step(self):
        pass

    def populate_context(self,ctx):
        e=self.env
        ids=ctx.env_ids if ctx.env_ids is not None else torch.arange(e.num_envs,device=e.device)
        mids=e.motion_manager.motion_ids[ids]
        # Initial observation construction can precede the first reset; use the
        # fixed scene IDs for that construction, and enforce pairing thereafter.
        expected=self.fixed_ids[ids]
        if bool((mids!=expected).any()) and bool((e.progress_buf[ids]>0).any()):
            raise ValueError("Motion/scene mismatch during rollout")
        mids=expected
        now=self.reference.query(mids,e.motion_manager.motion_times[ids])
        steps=e.config.control_components["mimic"].future_steps
        steps=list(range(1,steps+1)) if isinstance(steps,int) else steps
        times=e.motion_manager.motion_times[ids,None]+torch.as_tensor(steps,device=e.device)*e.dt
        future=self.reference.query(mids[:,None].expand_as(times),times)
        local=now["local_points"]
        q=ctx.current.rigid_body_rot[:,:,None,:].expand(*local.shape[:-1],4)
        offset=rotations.quat_rotate(q.reshape(-1,4),local.reshape(-1,3),w_last=True).reshape_as(local)
        actual=ctx.current.rigid_body_pos[:,:,None,:]+offset
        target=now["target_points"]+e.respawn_root_offset[ids,None,None,:]
        distances=torch.linalg.vector_norm(actual-target,dim=-1)
        valid=now["patch_weight"]>0
        # Require all represented expected objects, not just any nonzero force.
        # This is a conservative proximity association, not an exact pair sensor.
        match=((distances<self.config.association_distance)|~valid).all(-1)&valid.any(-1)
        c=WholeBodyContactContext()
        c.state=now["state"]; c.confidence=now["confidence"]
        c.future_state=future["state"]; c.future_confidence=future["confidence"]
        c.expected_patch_match=match
        ctx.whole_body_contact=c
