"""Reset rigid scene objects without assigning velocity to kinematic bodies."""
import torch


def write_scene_object_state(obj,states,env_ids,fixed_mask):
    if states.ndim!=2 or states.shape[1]!=13 or len(states)!=len(env_ids) or len(states)!=len(fixed_mask):
        raise ValueError("Scene root state and environment masks have inconsistent shapes")
    obj.write_root_pose_to_sim(states[:,:7],env_ids)
    dynamic=~fixed_mask
    if bool(dynamic.any()):
        obj.write_root_velocity_to_sim(states[dynamic,7:],env_ids[dynamic])
