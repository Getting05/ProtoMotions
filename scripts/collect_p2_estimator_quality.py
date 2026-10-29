"""Collect P2 expansion rollouts with per-episode QA and immediate divergence reset.

Run from repository root using Isaac Lab Python, PYTHONPATH=.
A manifest specifies exact motion IDs, categories and shared source group IDs.
No checkpoint, reference, or GPC training config is modified.
"""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--threads',type=int,default=4)
p.add_argument('--checkpoint',required=True); p.add_argument('--motion-file',required=True)
p.add_argument('--manifest',required=True); p.add_argument('--output',required=True)
p.add_argument('--config',default='configs/state_estimation/p2_pelvis.json')
p.add_argument('--num-envs',type=int,default=8); p.add_argument('--episodes',type=int,default=100)
p.add_argument('--max-steps',type=int,default=500); p.add_argument('--no-randomization',action='store_true')
a=p.parse_args()
# Parse/check local files before initializing a GPU simulator.
checkpoint=Path(a.checkpoint)
for path in (checkpoint,checkpoint.parent/'resolved_configs_inference.pt',Path(a.motion_file),Path(a.manifest)):
    if not path.is_file(): raise FileNotFoundError(path)
config=json.loads(Path(a.config).read_text())
manifest=json.loads(Path(a.manifest).read_text())
if not manifest['motions']: raise ValueError('Manifest needs motions')
allowed_categories={'stand','walk_forward','walk_backward','walk_sideways','turn','stop','squat','rise','run','jump','push','slip','dance','kneel'}
if any(m.get('category') not in allowed_categories for m in manifest['motions']):
    raise ValueError('Review manifest categories; unclassified motions are not accepted')
groups={}
for m in manifest['motions']:
    if 'split' in m:
        if m['split'] not in ('train','validation','test'): raise ValueError('Unknown manifest split')
        if groups.setdefault(m['group_id'],m['split']) != m['split']:
            raise ValueError('A motion group crosses dataset splits')
if min(a.num_envs,a.episodes,a.max_steps)<1: raise ValueError('Positive counts required')
from protomotions.utils.simulator_imports import import_simulator_before_torch
AppLauncher=import_simulator_before_torch('isaaclab')
import torch
torch.set_num_threads(a.threads)
from lightning.fabric import Fabric
from protomotions.utils.fabric_config import FabricConfig
from protomotions.utils.hydra_replacement import get_class
from protomotions.utils.component_builder import build_all_components
from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
from protomotions.simulator.base_simulator.config import (
    DomainRandomizationConfig,FrictionDomainRandomizationConfig,PushDomainRandomizationConfig)
from protomotions.components.terrains.config import TerrainConfig
from protomotions.state_estimation.physics import FootKinematics,DEFAULT_URDF,FEET,ANCHOR
from protomotions.state_estimation.data import write_episode,split_for_group,SCHEMA_VERSION
from protomotions.state_estimation.quality import invalid_frames,audit_episode,LIMITS


def run(app,fabric):
    from protomotions.state_estimation.isaac_adapter import IsaacObservationReader
    frozen=torch.load(checkpoint.parent/'resolved_configs_inference.pt',map_location='cpu',weights_only=False)
    rc,sc,ec,ac,mc,lc=[frozen[k] for k in ('robot','simulator','env','agent','motion_lib','scene_lib')]
    if 'isaaclab' not in sc._target_: raise ValueError('Expected IsaacLab Stage1 frozen config')
    rc.contact_bodies=list(dict.fromkeys(list(rc.contact_bodies or [])+list(FEET)))
    if rc.kinematic_info.body_names[rc.anchor_body_index] != ANCHOR:
        raise ValueError('Tracker heading anchor differs from estimator contract')
    from protomotions.robot_configs.base import ControlType
    if rc.control.control_type not in (ControlType.PROPORTIONAL,ControlType.BUILT_IN_PD):
        raise ValueError('Collector requires PD targets for the previous-action feature')
    sc.num_envs=a.num_envs; sc.headless=True; sc.w_last=True
    sc.projectile.num_projectiles=0
    for key in ('gpu_max_rigid_contact_count','gpu_found_lost_pairs_capacity','gpu_found_lost_aggregate_pairs_capacity','gpu_max_rigid_patch_count'):
        setattr(sc.sim.physx,key,min(getattr(sc.sim.physx,key),2**19))
    # Explicitly replace terrain with flat ground; retain friction conversion convention.
    tc=TerrainConfig()
    sc.domain_randomization=None
    if not a.no_randomization:
        dr=config['collection']
        sc.domain_randomization=DomainRandomizationConfig(
            friction=FrictionDomainRandomizationConfig(body_names=['.*'],
                static_friction_range=tuple(dr['friction_range']),dynamic_friction_range=tuple(dr['friction_range']),
                restitution_range=(0.,0.)),
            push=PushDomainRandomizationConfig(push_interval_range=tuple(dr['push_interval_range']),
                max_linear_velocity=tuple(dr['push_linear_velocity']),max_angular_velocity=tuple(dr['push_angular_velocity'])))
    tc,sc=convert_friction_for_simulator(tc,sc)
    sc._target_='protomotions.state_estimation.isaac_adapter.EstimatorIsaacLabSimulator'
    mc.motion_file=str(Path(a.motion_file).resolve()); mc.motion_file_shard_indices=None
    lc.scene_file=None; lc.asset_root=None
    ec.max_episode_length=a.max_steps+2
    components=build_all_components(terrain_config=tc,scene_lib_config=lc,motion_lib_config=mc,
        simulator_config=sc,robot_config=rc,device=fabric.device,simulation_app=app)
    env=get_class(ec._target_)(config=ec,robot_config=rc,device=fabric.device,**components)
    if abs(env.dt-1/config['control_frequency'])>1e-8:
        raise ValueError(f'Frozen controller rate {1/env.dt} Hz does not match estimator')
    physics=FootKinematics(DEFAULT_URDF,list(rc.kinematic_info.dof_names),list(rc.kinematic_info.body_names)).to(fabric.device)
    agent=get_class(ac._target_)(config=ac,env=env,fabric=fabric,root_dir=checkpoint.parent)
    agent.setup(); agent.load(checkpoint,load_env=False,load_training_state=False); agent.eval()
    agent.model.requires_grad_(False)
    reader=IsaacObservationReader(env.simulator,physics,env.dt,config['contact'])
    motions=manifest['motions']
    for motion in motions:
        for key in ('motion_id','motion_name','group_id','category'):
            if key not in motion: raise ValueError(f'Manifest missing {key}')
        mid=motion['motion_id']
        if mid<0 or mid>=env.motion_lib.num_motions(): raise ValueError('Invalid motion ID')
        if str(env.motion_lib.motion_files[mid]) != motion['motion_name']:
            raise ValueError('Motion names/IDs changed; regenerate and review manifest')
    output=Path(a.output); output.mkdir(parents=True,exist_ok=False)
    index=dict(schema_version=SCHEMA_VERSION,dt=env.dt,contract=physics.contract(),episodes=[],
               synthetic_only=False,source_checkpoint=str(checkpoint.resolve()),source_motion_file=mc.motion_file,
               collection_config=config,randomization=not a.no_randomization,
               imu_model='control-rate IMU point-velocity finite difference; initial sample invalid')
    buffers=[[] for _ in range(a.num_envs)]
    assigned=[None]*a.num_envs; lengths=[0]*a.num_envs; next_motion=0; completed=0
    def reset(ids):
        nonlocal next_motion
        for i in ids.tolist():
            if next_motion>=a.episodes:
                assigned[i]=None
                continue
            assigned[i]=motions[next_motion%len(motions)]; next_motion+=1
            env.motion_manager.motion_ids[i]=assigned[i]['motion_id']
            env.motion_manager.motion_times[i]=0.
            lengths[i]=0; buffers[i]=[]
        reader.reset(ids)
        return env.reset(ids,disable_motion_resample=True,sample_flat=True)[0]
    obs=reset(torch.arange(a.num_envs,device=fabric.device))
    with torch.inference_mode():
        step=0
        while completed<a.episodes:
            agent.pre_collect_step(step)
            actor_obs=agent.add_agent_info_to_obs(obs)
            result=agent.model(agent.obs_dict_to_tensordict(actor_obs))
            action=result['mean_action'] if 'mean_action' in result else result['action']
            if hasattr(env.simulator,'_estimator_push_mask'):
                env.simulator._estimator_push_mask.zero_()
            obs,_,done,terminated,_=env.step(action)
            sample=reader.read(env._current_processed_action)
            sample['tracking_error']=(env.context.current.rigid_body_pos-env.context.mimic.ref_state.rigid_body_pos).norm(dim=-1).mean(-1)
            done=done.bool()|env.motion_manager.get_done_tracks()
            # Transfer each batched field once; per-environment copies stay on CPU.
            invalid=invalid_frames(sample)
            done |= invalid
            sample_cpu={k:v.detach().cpu() for k,v in sample.items()}
            for i in range(a.num_envs):
                if assigned[i] is None: continue
                lengths[i]+=1
                frame={k:v[i].clone() for k,v in sample_cpu.items()}
                frame['time']=torch.tensor(lengths[i]*env.dt,dtype=torch.float64)
                buffers[i].append(frame)
                if lengths[i]>=a.max_steps: done[i]=True
                if done[i] and completed<a.episodes:
                    motion=assigned[i]; group=motion['group_id']
                    metadata=dict(file=f'episode_{completed:06d}.pt',group_id=group,
                        split=motion.get('split',split_for_group(group,config['seed'])),category=motion['category'],
                        motion_id=motion['motion_id'],motion_name=motion['motion_name'],
                        frames=lengths[i],terminated=bool(terminated[i]),
                        collection_seed=config['seed'],condition=motion['condition'],
                        source_motion_id=motion['source_motion_id'],quality_aborted=bool(invalid[i]))
                    episode={k:torch.stack([f[k] for f in buffers[i]]) for k in buffers[i][0]}
                    metadata['quality']=audit_episode(episode,metadata,env.dt,config['history_length'])
                    metadata['track_mean_m']=float(episode['tracking_error'].mean())
                    metadata['track_max_m']=float(episode['tracking_error'].max())
                    write_episode(output,episode,metadata); index['episodes'].append(metadata); completed+=1
                    (output/'index.json').write_text(json.dumps(index,indent=2))
                    print('COLLECTED '+json.dumps(metadata),flush=True)
            obs=reset(done.nonzero().flatten())
            step+=1
    print('QUALITY_COLLECTION_COMPLETE '+str(output),flush=True)
    return 0


fabric=Fabric(**FabricConfig(accelerator='gpu',devices=1,num_nodes=1,loggers=[],callbacks=[]).as_kwargs())
fabric.launch(); fabric.seed_everything(config['seed'])
launcher=AppLauncher({'headless':True,'device':str(fabric.device)})
exit_code=1
try:
    exit_code=run(launcher.app,fabric)
except BaseException:
    import traceback
    traceback.print_exc()
finally:
    launcher.app.close(exit_code=exit_code)
