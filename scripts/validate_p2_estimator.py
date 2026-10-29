"""Interactive P2 estimator validation: Isaac RGB + red estimated robot + charts.

From repo root (Isaac Lab environment), run --help. Open the printed loopback URL
through SSH forwarding when running on a remote host. No training or controller
feedback substitution occurs. The estimator receives measured proprioception only.
"""
import argparse
import json
import queue
import time
from pathlib import Path


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller',required=True,help='Frozen Stage1 .ckpt with sibling resolved_configs_inference.pt')
    p.add_argument('--estimator',required=True,help='Deployable baseline/fusion best.pt (Oracle is rejected)')
    p.add_argument('--motion-file',required=True,help='Packed .pt motion library; all clips selectable in browser')
    p.add_argument('--motion-id',type=int,default=0)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8876)
    p.add_argument('--output',help='Optional NEW directory: per-frame metrics/463D arrays and last rendered image')
    p.add_argument('--autoplay',action='store_true',help='Play immediately instead of waiting for browser')
    p.add_argument('--max-steps',type=int,default=0,help='Exit after this many total control steps; 0 = interactive')
    p.add_argument('--no-render',action='store_true',help='Physics/metrics smoke check without RTX camera')
    p.add_argument('--render-fps',type=float,default=10.)
    p.add_argument('--width',type=int,default=960);p.add_argument('--height',type=int,default=720)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    for path in (a.controller,a.estimator,a.motion_file,str(Path(a.controller).parent/'resolved_configs_inference.pt')):
        if not Path(path).is_file():p.error('Missing file: '+path)
    if a.render_fps<=0 or a.width<128 or a.height<128 or a.max_steps<0:p.error('Invalid frame size, rate, or step limit')
    if a.output and Path(a.output).exists():p.error('--output must be a new directory to avoid overwriting experiments')
    import socket
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try:probe.bind((a.host,a.port))
        except OSError as exc:p.error(f'Cannot listen on {a.host}:{a.port}: {exc}; choose another --port')
    return a


def run(a,app,fabric):
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.utils.component_builder import build_all_components
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.components.motion_lib import MotionFileSwitchMode
    from protomotions.robot_configs.base import ControlType
    from protomotions.state_estimation.isaac_adapter import IsaacObservationReader
    from protomotions.state_estimation.inference import RealtimeEstimator
    from protomotions.state_estimation.physics import FEET,ANCHOR
    from protomotions.state_estimation.viewer_core import from_measurements,DisplayPlacement,diagnostic_frame,encode_gpc
    from protomotions.state_estimation.viewer_server import Dashboard
    from deployment.urdf_kinematics import URDFKinematics

    checkpoint=Path(a.controller)
    runtime=RealtimeEstimator(a.estimator,str(fabric.device))
    bundle=torch.load(a.estimator,map_location='cpu',weights_only=True)
    contract=bundle['contract'];physics=runtime.physics
    frozen=torch.load(checkpoint.parent/'resolved_configs_inference.pt',map_location='cpu',weights_only=False)
    rc,sc,ec,ac,mc,lc=[frozen[k] for k in ('robot','simulator','env','agent','motion_lib','scene_lib')]
    if 'isaaclab' not in sc._target_:raise ValueError('This viewer requires an IsaacLab controller config')
    if rc.control.control_type not in (ControlType.PROPORTIONAL,ControlType.BUILT_IN_PD):raise ValueError('Expected PD controller')
    if list(rc.kinematic_info.body_names)!=contract['body_names'] or list(rc.kinematic_info.dof_names)!=contract['dof_names']:
        raise ValueError('Controller/estimator joint or body ordering mismatch')
    if rc.kinematic_info.body_names[rc.anchor_body_index]!=ANCHOR:raise ValueError('Wrong GPC anchor')
    rc.contact_bodies=list(dict.fromkeys(list(rc.contact_bodies or [])+list(FEET)))
    sc.num_envs=1;sc.headless=True;sc.w_last=True;sc.domain_randomization=None
    sc.projectile.num_projectiles=0
    for key in ('gpu_max_rigid_contact_count','gpu_found_lost_pairs_capacity','gpu_found_lost_aggregate_pairs_capacity','gpu_max_rigid_patch_count'):
        setattr(sc.sim.physx,key,min(getattr(sc.sim.physx,key),2**18))
    tc,sc=convert_friction_for_simulator(TerrainConfig(),sc)
    sc._target_='protomotions.state_estimation.isaac_adapter.EstimatorIsaacLabSimulator'
    mc.motion_file=str(Path(a.motion_file).resolve());mc.motion_file_shard_indices=None
    mc.motion_file_switch_mode=MotionFileSwitchMode.FIXED
    lc.scene_file=None;lc.asset_root=None
    ec.motion_manager.init_start_prob=1.;ec.motion_manager.subset_method=None
    ec.motion_manager.exclude_motion_ids=None;ec.motion_manager.exclude_motions_file=None
    ec.max_episode_length=max(ec.max_episode_length,100000)
    components=build_all_components(terrain_config=tc,scene_lib_config=lc,motion_lib_config=mc,
        simulator_config=sc,robot_config=rc,device=fabric.device,simulation_app=app)
    env=get_class(ec._target_)(config=ec,robot_config=rc,device=fabric.device,**components)
    if abs(env.dt-runtime.dt)>1e-8:raise ValueError('Estimator/controller frequency mismatch')
    names=[str(n) for n in env.motion_lib.motion_files]
    if not 0<=a.motion_id<len(names):raise ValueError('Motion ID out of range')
    agent=get_class(ac._target_)(config=ac,env=env,fabric=fabric,root_dir=checkpoint.parent)
    agent.setup();agent.load(checkpoint,load_env=False,load_training_state=False);agent.eval();agent.model.requires_grad_(False)
    collection=bundle.get('data_index',{}).get('collection_config',{})
    contact_config=collection.get('contact',dict(force_min=30.,slip_max=.1))
    reader=IsaacObservationReader(env.simulator,physics,env.dt,contact_config)
    kinematics=URDFKinematics(contract['urdf'],contract['body_names'],contract['dof_names'])
    placement=DisplayPlacement(contract['body_names'],env.dt)
    dashboard=Dashboard(a.host,a.port,names,dict(controller=str(checkpoint.resolve()),estimator=str(Path(a.estimator).resolve()),
        mode=bundle['mode'],body_names=contract['body_names'],dof_names=contract['dof_names'],
        controller_uses_estimates=False,display_alignment_only=True,frame_sequence=0,episode=0))
    output=Path(a.output) if a.output else None
    log=None;camera=None;ghost=None
    if output:
        output.mkdir(parents=True)
        log=(output/'metrics.jsonl').open('w',buffering=1)
        (output/'request.json').write_text(json.dumps(vars(a),indent=2))
    total_steps=0;episode=0;motion_id=a.motion_id;step=0;frame_count=0
    playing=a.autoplay;loop=True;mode='aligned';speed=1.;stopping=False;force_step=False
    ended=False;last_display=None;stop_reason=''
    camera_ready=False;last_render=-float('inf');last_point=None;frame=None
    started=time.perf_counter()

    def reset_motion(mid):
        nonlocal motion_id,episode,step,obs,last_point,last_render,ended,last_display,stop_reason
        ended=False;last_display=None;stop_reason=''
        motion_id=mid;episode+=1;step=0;last_point=None;last_render=-float('inf')
        env.motion_manager.motion_ids[:]=mid;env.motion_manager.motion_times[:]=0.
        runtime.reset();reader.reset();placement.reset()
        obs=env.reset(disable_motion_resample=True,sample_flat=True)[0]
        if ghost:ghost.set_visible(False)
        dashboard.update_status(ready=True,episode=episode,motion_id=mid,motion_name=names[mid],
            message=f'预热：需要 {runtime.history_length} 帧有效传感器历史（{runtime.history_length*env.dt:.2f} 秒）',playing=playing)
        print(f'VIEWER_MOTION {mid} {names[mid]}',flush=True)

    try:
        if not a.no_render:
            from protomotions.state_estimation.viewer_render import EstimatedRobot,FollowCamera,jpeg
            ghost=EstimatedRobot(env);camera=FollowCamera(env.simulator,a.width,a.height)
        reset_motion(motion_id)
        print(f'VIEWER_READY http://{a.host}:{a.port} mode={bundle["mode"]} motions={len(names)}',flush=True)
        with torch.inference_mode():
            while not stopping:
                tick=time.perf_counter()
                while True:
                    try:cmd=dashboard.commands.get_nowait()
                    except queue.Empty:break
                    action=cmd['action']
                    if action=='play':
                        if ended:reset_motion(motion_id)
                        playing=True
                    elif action=='pause':playing=False
                    elif action=='step':
                        if ended:reset_motion(motion_id)
                        playing=False;force_step=True
                    elif action=='reset':reset_motion(motion_id)
                    elif action=='motion':reset_motion(cmd['id']);playing=True
                    elif action=='placement':
                        mode=cmd['value']
                        if ghost and last_display is not None:
                            pose=last_display[mode]
                            ghost.set_pose(pose['body_pos'][0],pose['body_rot'][0])
                            encoded=jpeg(camera.frame(ghost));dashboard.publish(frame=encoded)
                            frame_count+=1;dashboard.update_status(frame_sequence=frame_count,placement=mode)
                    elif action=='speed':speed=cmd['value']
                    elif action=='loop':loop=cmd['value']
                    elif action=='stop':stopping=True
                if stopping:break
                if not playing and not force_step:
                    dashboard.update_status(playing=False,message=f'{stop_reason or "已暂停"} · Motion {motion_id} · {step*env.dt:.2f} s · 点击播放或单步')
                    time.sleep(.03);continue
                force_step=False
                agent.pre_collect_step(total_steps)
                actor_obs=agent.add_agent_info_to_obs(obs)
                result=agent.model(agent.obs_dict_to_tensordict(actor_obs))
                action=result['mean_action'] if 'mean_action' in result else result['action']
                obs,_,done,terminated,_=env.step(action)
                step+=1;total_steps+=1;timestamp=step*env.dt
                sample=reader.read(env._current_processed_action)
                prediction=runtime.update(sample['proprio'][0],timestamp) if bool(sample['valid'][0]) else None
                if prediction is not None:
                    x=sample['proprio'][0].detach().cpu().numpy()
                    estimate=prediction['state'].detach().cpu().numpy()
                    poses,gpc,tilt,omega_b=from_measurements(x,estimate,physics,kinematics)
                    # Simulator state enters only the diagnostics/display branch.
                    bodies=env.simulator.get_bodies_state()
                    truth={k:getattr(bodies,field).detach().cpu().numpy() for k,field in
                        [('body_pos','rigid_body_pos'),('body_rot','rigid_body_rot'),('body_vel','rigid_body_vel'),('body_ang_vel','rigid_body_ang_vel')]}
                    root=contract['body_names'].index('pelvis')
                    ground=float(env.simulator.terrain.get_ground_heights(bodies.rigid_body_pos[:,root]).reshape(-1)[0])
                    root_r=Rotation.from_quat(poses['body_rot'][0,root])
                    origin_velocity=poses['body_vel'][0,root]-np.cross(poses['body_ang_vel'][0,root],root_r.apply(kinematics.com_offsets[root]))
                    aligned,drift=placement.place(poses,truth,ground,tilt,omega_b,origin_velocity)
                    # Read the ORIGINAL env feature when available and audit it
                    # against the same exported encoder rather than another formula.
                    truth_relative={k:v.copy() for k,v in truth.items()};truth_relative['body_pos'][...,2]-=ground
                    encoded_truth=encode_gpc(truth_relative,rc.anchor_body_index)
                    if 'max_coords_obs' not in obs:raise ValueError('Controller env does not expose max_coords_obs')
                    original=obs['max_coords_obs'].detach().cpu().numpy()
                    if original.shape!=(1,463):raise ValueError(f'Unexpected max_coords_obs shape {original.shape}')
                    encoder_error=float(np.max(np.abs(encoded_truth-original)))
                    if encoder_error>2e-4:raise ValueError(f'GPC truth encoder mismatch: {encoder_error}')
                    qtruth=env.simulator.get_dof_state().dof_pos[0].detach().cpu().numpy()
                    point=diagnostic_frame(estimate,sample['target'][0].cpu().numpy(),poses,aligned,drift,truth,original,x[:30],qtruth,kinematics)
                    point.update(time=timestamp,episode=episode,motion_id=motion_id,
                        contact=sample['contact'][0].cpu().tolist(),gpc_truth_encoder_max_abs=encoder_error)
                    last_display=dict(aligned=aligned,drift=drift)
                    if ghost:ghost.set_pose((aligned if mode=='aligned' else drift)['body_pos'][0],(aligned if mode=='aligned' else drift)['body_rot'][0]);ghost.set_visible(True)
                    if output:
                        # Arrays retain the exact common order, point conventions,
                        # and original 463D truth for subsequent numerical checks.
                        np.savez_compressed(output/'last_state.npz',proprio=x,estimate=estimate,gpc_estimate=gpc,gpc_truth=original,
                            **{'estimated_'+k:v for k,v in poses.items()},**{'truth_'+k:v for k,v in truth.items()})
                        log.write(json.dumps(point,allow_nan=False)+'\n')
                    # 463 raw features are saved on disk, not streamed at 50 Hz.
                    dashboard.publish({k:v for k,v in point.items() if k!='gpc_estimate'})
                    last_point=point
                if camera and timestamp-last_render>=1/a.render_fps-1e-8:
                    if not camera_ready:camera.warmup();camera_ready=True
                    frame=camera.frame(ghost if prediction is not None else None)
                    encoded=jpeg(frame);dashboard.publish(frame=encoded)
                    frame_count+=1;last_render=timestamp
                    if output:(output/'last_frame.jpg').write_bytes(encoded)
                dashboard.update_status(playing=playing,frame_sequence=frame_count,placement=mode,
                    message=(f'Motion {motion_id} · {timestamp:.2f} s · {"XY / yaw 对齐" if mode=="aligned" else "积分漂移"} · '
                             +('估计有效' if prediction is not None else f'预热 {len(runtime.history)}/{runtime.history_length} 帧')))
                if a.max_steps and total_steps>=a.max_steps:break
                if bool(done[0]) or bool(env.motion_manager.get_done_tracks()[0]):
                    reason='失败终止' if bool(terminated[0]) else '动作结束'
                    print(f'VIEWER_EPISODE_END {motion_id} {reason}',flush=True)
                    if loop and not bool(terminated[0]):reset_motion(motion_id)
                    else:playing=False;ended=True;stop_reason=reason
                time.sleep(max(0,env.dt/speed-(time.perf_counter()-tick)))
        if output:
            (output/'summary.json').write_text(json.dumps(dict(total_steps=total_steps,episodes=episode,rendered_frames=frame_count,
                wall_seconds=time.perf_counter()-started,last_metrics=last_point,mode=bundle['mode'],controller_uses_estimates=False),indent=2))
        print(f'VIEWER_FINISHED steps={total_steps} frames={frame_count}',flush=True)
    finally:
        if log:log.close()
        if ghost:ghost.close()
        if camera:camera.close()
        dashboard.close()


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
    launcher=AppLauncher({'headless':True,'device':str(fabric.device),'enable_cameras':not a.no_render})
    exit_code=1
    try:
        run(a,launcher.app,fabric);exit_code=0
    except KeyboardInterrupt:
        print('Viewer stopped by user.',flush=True);exit_code=0
    except BaseException:
        import traceback
        traceback.print_exc()
    finally:launcher.app.close(exit_code=exit_code)


if __name__=='__main__':main()
