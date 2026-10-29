"""Paired P2 frozen-GPC closed-loop evaluation. Run from repository root.

Truth rewards/metrics; observation replacement covers both actor inputs. The
estimated condition uses FK stationary-lowest-foot bootstrap for 25 actions, then
25 valid sensor frames. Ideal XY/yaw, flat ground and reference clock are retained.
No GPC training, controller weight edits, true contact inputs or truth warmup.
"""
import argparse
import json
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller', default='results/p2_fsq_bones_8gpu_lift_foot_smooth/inference_score_based.ckpt')
    p.add_argument('--estimator', default='results/p2_estimator_noisy_contact_20261016/fusion/best.pt')
    p.add_argument('--data-index', default='results/p2_estimator_expanded_20261016/data/index.json')
    p.add_argument('--motion-file', default='results/p2_estimator_expanded_20261016/reference_subset.pt')
    p.add_argument('--noise-config', default='results/p2_estimator_noisy_contact_20261016/config.json')
    p.add_argument('--output', required=True)
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--suite', choices=['smoke','test'], default='smoke')
    p.add_argument('--modes', nargs='+', choices=['truth','oracle','estimated'], default=['truth','oracle','estimated'])
    p.add_argument('--conditions', nargs='+', choices=['clean','noisy'], default=['clean','noisy'])
    p.add_argument('--max-steps', type=int, default=1500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--audit-steps', type=int, default=80)
    p.add_argument('--oracle-tolerance', type=float, default=0.002)
    a=p.parse_args()
    for path in (a.controller,a.estimator,a.data_index,a.motion_file,a.noise_config,
                 str(Path(a.controller).parent/'resolved_configs_inference.pt')):
        if not Path(path).is_file(): p.error('Missing '+path)
    if Path(a.output).exists(): p.error('Output must be new')
    if min(a.num_envs,a.max_steps,a.audit_steps)<1: p.error('Positive counts required')
    return a


def run(a, app, fabric):
    import hashlib
    import time
    import numpy as np
    import torch
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.utils.component_builder import build_all_components
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.components.motion_lib import MotionFileSwitchMode
    from protomotions.state_estimation.isaac_adapter import IsaacObservationReader
    from protomotions.state_estimation.closed_loop import BatchedEstimator, ObservationBridge, EpisodeNoise
    from protomotions.state_estimation.physics import FEET, ANCHOR
    from protomotions.utils import rotations

    out=Path(a.output); out.mkdir(parents=True)
    (out/'request.json').write_text(json.dumps(vars(a),indent=2))
    checkpoint=Path(a.controller)
    runtime=BatchedEstimator(a.estimator,str(fabric.device)); physics=runtime.physics
    bridge=ObservationBridge(physics)
    bundle=torch.load(a.estimator,map_location='cpu',weights_only=True)
    # These are the user's explicitly designated local frozen experiment artifacts.
    frozen=torch.load(checkpoint.parent/'resolved_configs_inference.pt',map_location='cpu',weights_only=False)
    rc,sc,ec,ac,mc,lc=[frozen[k] for k in ('robot','simulator','env','agent','motion_lib','scene_lib')]
    if 'isaaclab' not in sc._target_: raise ValueError('Expected IsaacLab checkpoint')
    if list(rc.kinematic_info.body_names)!=physics.body_names or list(rc.kinematic_info.dof_names)!=physics.dof_names:
        raise ValueError('Controller/estimator ordering mismatch')
    if rc.kinematic_info.body_names[rc.anchor_body_index]!=ANCHOR: raise ValueError('Wrong anchor')
    if ec.motion_manager.realign_motion_with_humanoid_on_each_step: raise ValueError('Reference realignment requires audit')
    rc.contact_bodies=list(dict.fromkeys(list(rc.contact_bodies or [])+list(FEET)))
    sc.num_envs=a.num_envs; sc.headless=True; sc.w_last=True; sc.domain_randomization=None
    sc.projectile.num_projectiles=0
    for key in ('gpu_max_rigid_contact_count','gpu_found_lost_pairs_capacity','gpu_found_lost_aggregate_pairs_capacity','gpu_max_rigid_patch_count'):
        setattr(sc.sim.physx,key,min(getattr(sc.sim.physx,key),2**19))
    tc,sc=convert_friction_for_simulator(TerrainConfig(),sc)
    sc._target_='protomotions.state_estimation.isaac_adapter.EstimatorIsaacLabSimulator'
    mc.motion_file=str(Path(a.motion_file).resolve()); mc.motion_file_shard_indices=None
    mc.motion_file_switch_mode=MotionFileSwitchMode.FIXED
    lc.scene_file=None; lc.asset_root=None
    ec.motion_manager.init_start_prob=1.; ec.motion_manager.subset_method=None
    ec.motion_manager.exclude_motion_ids=None; ec.motion_manager.exclude_motions_file=None
    ec.max_episode_length=max(ec.max_episode_length,a.max_steps+2)
    components=build_all_components(terrain_config=tc,scene_lib_config=lc,motion_lib_config=mc,
        simulator_config=sc,robot_config=rc,device=fabric.device,simulation_app=app)
    env=get_class(ec._target_)(config=ec,robot_config=rc,device=fabric.device,**components)
    if not env.terrain.is_flat() or abs(env.dt-runtime.dt)>1e-8: raise ValueError('Flat 50Hz required')
    agent=get_class(ac._target_)(config=ac,env=env,fabric=fabric,root_dir=checkpoint.parent)
    agent.setup(); agent.load(checkpoint,load_env=False,load_training_state=False)
    agent.eval(); agent.model.requires_grad_(False)
    config=json.loads(Path(a.noise_config).read_text())
    reader=IsaacObservationReader(env.simulator,physics,env.dt,config['contact'])
    names=[str(x) for x in env.motion_lib.motion_files]
    by_name={name:i for i,name in enumerate(names)}
    index=json.loads(Path(a.data_index).read_text())
    groups={}
    for ep in index['episodes']:
        if ep['split']=='test':
            if ep['motion_name'] not in by_name: raise ValueError('Test motion absent from selected library')
            groups.setdefault(ep['group_id'],dict(group_id=ep['group_id'],category=ep['category'],
                motion_name=ep['motion_name'],motion_id=by_name[ep['motion_name']]))
    motions=sorted(groups.values(),key=lambda x:x['motion_id'])
    if a.suite=='smoke':
        chosen=[]
        for category in ('stand','walk_forward','walk_backward','walk_sideways','turn','jump','run','squat'):
            matches=[m for m in motions if m['category']==category]
            if matches: chosen.append(matches[0])
        motions=chosen
    if not motions: raise ValueError('Empty test cohort')
    def digest(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
        return h.hexdigest()
    metadata=dict(motions=motions,dt=env.dt,history_length=runtime.history_length,
        estimator_mode=bundle['mode'],noise=config['noise'],
        controller_sha256=digest(a.controller),estimator_sha256=digest(a.estimator),
        source_sha256={p:digest(p) for p in ('protomotions/state_estimation/closed_loop.py', __file__)},
        observation_bindings={k:v.to_dict() for k,v in ec.observation_components.items()},
        retained_ideal_inputs=['pelvis XY','pelvis yaw','flat ground elevation','reference trajectory and clock',
                              'simulation-derived IMU specific force; no sensor calibration/delay/dropout'],
        warmup='Estimated mode first 25 actions use stationary lowest-foot FK h/v; truth and oracle modes use their own inputs from step zero. History reset each episode; first IMU sample invalid.',
        failure='Any native termination or mean all-body position error > 0.5m after frozen reset grace; truth only.',
        dynamics='Frozen nominal dynamics on flat terrain; clean/noisy vary sensors only; no pushes/friction randomization.',
        noise_scope='All noisy proprioceptive q/dq/gravity/gyro also feed FK; oracle replaces h/v only. Truth is the original raw simulator baseline (clean condition only); oracle uses FK with true h/v.',
        full_test_groups=len(groups),frozen_termination_components=list(ec.termination_components))
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    records=[]; audit={k:0. for k in ec.observation_components}; action_audit=0.; total=0
    started=time.perf_counter()
    log=(out/'episodes.jsonl').open('w',buffering=1)
    frames=(out/'frames.jsonl').open('w',buffering=1)
    def policy(obs):
        result=agent.model(agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs)))
        return result['mean_action']
    def geometry(x,estimate):
        root=env.context.current.root_rot
        yaw=rotations.get_euler_xyz(root,w_last=True)[2]
        return bridge.reconstruct(x,estimate,env.context.current.root_pos[:,:2],yaw,env.context.ground_heights)
    try:
        with torch.inference_mode():
            for condition in a.conditions:
                for mode in a.modes:
                    if condition == 'noisy' and mode == 'truth': continue
                    for start in range(0,len(motions),a.num_envs):
                        cohort=motions[start:start+a.num_envs]
                        padded=cohort+[cohort[-1]]*(a.num_envs-len(cohort))
                        torch.manual_seed(a.seed+start); torch.cuda.manual_seed_all(a.seed+start)
                        env.motion_manager.motion_ids[:]=torch.tensor([m['motion_id'] for m in padded],device=fabric.device)
                        env.motion_manager.motion_times[:]=0.
                        runtime.reset(); reader.reset()
                        obs=env.reset(disable_motion_resample=True,sample_flat=True)[0]
                        # Read at reset to establish finite-difference IMU history, with invalid flag.
                        sample=reader.read(torch.zeros_like(env._current_processed_action))
                        noise=EpisodeNoise(a.num_envs,config['noise'],a.seed+start,fabric.device)
                        active=torch.arange(a.num_envs,device=fabric.device)<len(cohort)
                        sums=[dict(n=0,track=0.,track_sq=0.,height=0.,xy_sq=0.,z_sq=0.,est_n=0,
                                   warmup_n=0,warmup_track=0.,reward=0.,max_track=0.) for _ in cohort]
                        for step in range(a.max_steps):
                            x=noise(sample['proprio']) if condition=='noisy' else sample['proprio'].clone()
                            estimate,ready=runtime.update(x,sample['valid'])
                            # Oracle check is performed against raw env obs on the SAME true state.
                            if total<a.audit_steps:
                                po=geometry(sample['proprio'],sample['target'])
                                oo=bridge.observations(ec.observation_components,env.context,po,sample['proprio'])
                                for key in audit:
                                    err=float((oo[key]-obs[key]).abs().max());audit[key]=max(audit[key],err)
                                    if err>a.oracle_tolerance: raise RuntimeError(f'Oracle mismatch {key}: {err}')
                                action_audit=max(action_audit,float((policy(oo)-policy(obs)).abs().max()))
                            if mode=='estimated' and not ready:
                                actor_obs=bridge.observations(ec.observation_components,env.context,geometry(x,estimate),x)
                            elif mode=='truth': actor_obs={k:obs[k] for k in ec.observation_components}
                            else:
                                state=estimate if mode=='estimated' else sample['target']
                                actor_obs=bridge.observations(ec.observation_components,env.context,geometry(x,state),x)
                            if ready:
                                error=(estimate-sample['target']).cpu().numpy()
                                for i,s in enumerate(sums):
                                    if active[i]:
                                        s['height']+=float(abs(error[i,0]));s['xy_sq']+=float((error[i,1:3]**2).sum())
                                        s['z_sq']+=float(error[i,3]**2);s['est_n']+=1
                            action=policy(actor_obs)
                            obs,reward,done,terminated,_=env.step(action)
                            total+=1
                            sample=reader.read(env._current_processed_action)
                            ctx=env.context
                            track=(ctx.current.rigid_body_pos-ctx.mimic.ref_state.rigid_body_pos).norm(dim=-1).mean(-1)
                            natural=env.motion_manager.get_done_tracks()
                            failed=terminated.bool()|((track>.5)&(env.progress_buf>ec.reset_grace_period))
                            for i,m in enumerate(cohort):
                                if not bool(active[i]): continue
                                s=sums[i];t=float(track[i]);s['n']+=1;s['track']+=t;s['track_sq']+=t*t
                                s['max_track']=max(s['max_track'],t);s['reward']+=float(reward[i])
                                if mode=='estimated' and not ready:s['warmup_n']+=1;s['warmup_track']+=t
                                if a.suite=='smoke':
                                    frames.write(json.dumps(dict(condition=condition,mode=mode,motion_id=m['motion_id'],
                                        step=step+1,ready=ready,track_error=t,estimate=estimate[i].tolist(),
                                        # Estimate belongs to pre-action sample, not post-step label.
                                        failed=bool(failed[i])))+'\n')
                                finished=bool(failed[i] or done[i] or natural[i]) or step+1==a.max_steps
                                if finished:
                                    en=s['est_n'];n=s['n']
                                    record=dict(**m,condition=condition,mode=mode,steps=n,seconds=n*env.dt,
                                        completed=bool(natural[i] and not failed[i]),failed=bool(failed[i]),
                                        native_terminated=bool(terminated[i]),
                                        truncated=not bool(failed[i] or done[i] or natural[i]),
                                        failure_during_warmup=bool(failed[i] and mode=='estimated' and not ready),warmup_steps=s['warmup_n'],
                                        track_mean_m=s['track']/n,track_rms_m=(s['track_sq']/n)**.5,
                                        track_max_m=s['max_track'],reward_mean=s['reward']/n,
                                        estimator_frames=en,height_mae_m=s['height']/en if en else None,
                                        xy_vector_rmse_mps=(s['xy_sq']/en)**.5 if en else None,
                                        z_rmse_mps=(s['z_sq']/en)**.5 if en else None)
                                    records.append(record);log.write(json.dumps(record)+'\n');active[i]=False
                                    print('EPISODE '+json.dumps(record),flush=True)
                            if not bool(active.any()): break
                        summary=dict(episodes=records,oracle_observation_max_abs=audit,oracle_action_max_abs=action_audit,
                                     wall_seconds=time.perf_counter()-started,simulation_batch_steps=total)
                        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        print('CLOSED_LOOP_COMPLETE '+str(out),flush=True)
    finally: log.close();frames.close()


def main():
    a=arguments()
    from protomotions.utils.simulator_imports import import_simulator_before_torch
    AppLauncher=import_simulator_before_torch('isaaclab')
    import torch
    from lightning.fabric import Fabric
    from protomotions.utils.fabric_config import FabricConfig
    torch.set_num_threads(a.threads)
    fabric=Fabric(**FabricConfig(accelerator='gpu',devices=1,num_nodes=1,loggers=[],callbacks=[]).as_kwargs())
    fabric.launch();fabric.seed_everything(a.seed)
    launcher=AppLauncher({'headless':True,'device':str(fabric.device)})
    exit_code=1
    try:run(a,launcher.app,fabric);exit_code=0
    except BaseException:
        import traceback;traceback.print_exc()
    finally:launcher.app.close(exit_code=exit_code)


if __name__=='__main__': main()
