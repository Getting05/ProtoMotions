"""Contact-aware Stage 1. Requires --scenes-file with a verified binding sidecar.

Encoder receives the future contact plan. Decoder receives proprioception and
latent only. New Stage 2 tokens must be extracted with this same contact input.
"""
from examples.experiments.mimic import fsq_lift_foot_smooth as baseline
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.control.whole_body_contact_control import WholeBodyContactConfig
from protomotions.envs.obs.whole_body_contact_obs import whole_body_contact_plan
from protomotions.envs.rewards.whole_body_contact_reward import whole_body_contact_reward,unexpected_contact_penalty

terrain_config=baseline.terrain_config
motion_lib_config=baseline.motion_lib_config


def scene_lib_config(args):
    cfg=baseline.scene_lib_config(args)
    if not cfg.scene_file: raise ValueError("fsq_whole_body_scene requires --scenes-file")
    cfg.require_motion_scene_binding=True
    return cfg


def env_config(robot_cfg,args):
    cfg=baseline.env_config(robot_cfg,args)
    cfg.control_components["whole_body_contact"]=WholeBodyContactConfig()
    c=EnvContext.whole_body_contact
    cfg.observation_components["whole_body_contact_plan"]=MdpComponent(
        whole_body_contact_plan,{"state":c.future_state,"confidence":c.future_confidence})
    cfg.reward_components.pop("contact_match_rew",None)
    cfg.reward_components["whole_body_contact"]=MdpComponent(whole_body_contact_reward,
        {"state":c.state,"confidence":c.confidence,
         "simulated_contacts":EnvContext.current.rigid_body_contacts,"expected_patch_match":c.expected_patch_match},
        {"weight":.15,"zero_during_grace_period":True})
    cfg.reward_components["unexpected_contact"]=MdpComponent(unexpected_contact_penalty,
        {"state":c.state,"confidence":c.confidence,"force_magnitudes":EnvContext.current_contact_force_magnitudes},
        {"weight":-.1,"force_threshold":25.,"zero_during_grace_period":True})
    return cfg


def agent_config(robot_config,env_config,args):
    cfg=baseline.agent_config(robot_config,env_config,args)
    key="whole_body_contact_plan"
    cfg.model.actor.mu_model.encoder.in_keys.append(key)
    cfg.model.actor.in_keys.append(key)
    cfg.model.critic.in_keys.append(key)
    cfg.model.in_keys.append(key)
    return cfg


def configure_robot_and_simulator(robot_cfg,simulator_cfg,args):
    baseline.configure_robot_and_simulator(robot_cfg,simulator_cfg,args)
    robot_cfg.update_fields(contact_bodies="all")


apply_inference_overrides=baseline.apply_inference_overrides
