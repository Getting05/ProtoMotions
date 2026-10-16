"""Exercise real scene/reset/contact/FSQ/backward/checkpoint paths, without training.

This is an integration smoke with an untrained policy, not tracking success.
"""
import argparse
from pathlib import Path
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion-file",required=True);p.add_argument("--scenes-file",required=True)
    p.add_argument("--simulator",choices=["isaaclab"],default="isaaclab")
    p.add_argument("--device",default="cpu");p.add_argument("--output",required=True)
    p.add_argument("--steps",type=int,default=8)
    args=p.parse_args()
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    Launcher=import_simulator_before_torch(args.simulator)
    import torch
    from tensordict import TensorDict
    from protomotions.robot_configs.factory import robot_config
    from protomotions.simulator.factory import simulator_config
    from protomotions.utils.component_builder import build_all_components
    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.agents.ppo.model import PPOModel
    from examples.experiments.mimic import fsq_whole_body_scene as exp
    torch.set_num_threads(1)
    device=torch.device(args.device)
    n=2
    args.batch_size=n;args.training_max_steps=1;args.num_envs=n
    extra={}
    if args.simulator=="isaaclab":
        launcher=Launcher({"headless":True,"device":str(device)})
        extra["simulation_app"]=launcher.app
    robot=robot_config("astro_p2")
    sim=simulator_config(args.simulator,robot,headless=True,num_envs=n,experiment_name="wbc_scene_smoke")
    exp.configure_robot_and_simulator(robot,sim,args)
    envcfg=exp.env_config(robot,args)
    for group in (envcfg.observation_components,envcfg.reward_components,envcfg.termination_components):
        for component in group.values():component.compile=False
    agentcfg=exp.agent_config(robot,envcfg,args)
    components=build_all_components(exp.terrain_config(args),exp.scene_lib_config(args),
                exp.motion_lib_config(args),sim,robot,device,**extra)
    env=BaseEnv(envcfg,robot,device,**components)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    try:
        print("WBC_SMOKE environment initialized",flush=True)
        obs,_=env.reset()
        print("WBC_SMOKE reset complete",flush=True)
        model=PPOModel(agentcfg.model).to(device)
        td=TensorDict(obs,batch_size=[n],device=device)
        model.materialize(td.clone());model.init_rollout_state(n,device)
        print("WBC_SMOKE model materialized",flush=True)
        model.train()
        result=model(td.clone())
        loss=result["mean_action"].square().mean()+result["value"].square().mean()
        loss.backward()
        gradients=[x.grad for x in model.parameters() if x.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-5);optimizer.step();optimizer.zero_grad()
        ckpt=output/"smoke_model.pt"
        torch.save({"model":model.state_dict(),"config":agentcfg.model},ckpt)
        restored=PPOModel(agentcfg.model).to(device)
        saved=torch.load(ckpt,map_location=device,weights_only=False)
        restored.materialize(td.clone())
        restored.load_state_dict(saved["model"])
        restored.init_rollout_state(n,device)
        model.eval();restored.eval()
        with torch.no_grad():
            a=model(td.clone())["mean_action"]
            b=restored(td.clone())["mean_action"]
            torch.testing.assert_close(a,b)
            rewards=[];forces=[]
            for _ in range(args.steps):
                result=restored(TensorDict(obs,batch_size=[n],device=device))
                obs,reward,reset,terminated,info=env.step(result["mean_action"])
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(v).all() for v in obs.values())
                state=env.simulator.get_robot_state()
                forces.append(float(torch.linalg.vector_norm(state.rigid_body_contact_forces,dim=-1).max()))
                rewards.append(float(reward.mean()))
                done=(reset|terminated).nonzero(as_tuple=False).flatten()
                if len(done):obs,_=env.reset(done)
            # An explicit subset query exercises non-contiguous reset/context indexing.
            ctx=env._build_context(torch.tensor([n-1],device=device))
            assert ctx.whole_body_contact.state.shape[0]==1
        report={"status":"passed","simulator":args.simulator,"steps":args.steps,"envs":n,
                "observation_shapes":{k:list(v.shape) for k,v in obs.items()},
                "reward_means":rewards,"max_contact_force_n":max(forces),
                "fsq_backward":True,"checkpoint_roundtrip":True,"subset_context":True,
                "motion_ids":env.motion_manager.motion_ids.cpu().tolist(),
                "scene_motion_ids":env.scene_lib.get_per_env_humanoid_motion_ids_tensor().cpu().tolist(),
                "tracking_success_evaluated":False,"policy":"untrained_smoke"}
        (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        (output/"failure.txt").unlink(missing_ok=True)
        print("WBC_STAGE1_SMOKE_PASS "+json.dumps(report),flush=True)
    except BaseException:
        import traceback
        message=traceback.format_exc()
        (output/"failure.txt").write_text(message)
        print(message,flush=True)
        raise
    finally:
        env.close()


if __name__=="__main__":main()
