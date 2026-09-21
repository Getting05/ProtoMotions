"""Validate all outputs, package in G1 source order and preserve source motion weights."""
from pathlib import Path
import argparse,json,os,torch,numpy as np,yaml,mujoco,re
from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.robot_configs.factory import robot_config
p=argparse.ArgumentParser();p.add_argument('output_dir',type=Path);a=p.parse_args()
torch.set_num_threads(1)
srcroot=Path('/data/chenguanting/datasets/AMASS_Retargeted_for_G1/protomotions_g1')
src=torch.load(srcroot/'amass_g1.pt',map_location='cpu',mmap=True,weights_only=False)
cfg=robot_config('astro_p2')
limits={}
import xml.etree.ElementTree as E
for j in E.parse('protomotions/data/assets/astro_p2/urdf/astro_p2_30dof_primitive_collision.urdf').findall('joint'):
 if j.get('type')!='fixed': limits[j.get('name')]=j.find('limit').attrib
lo=torch.tensor([float(limits[n]['lower']) for n in cfg.kinematic_info.dof_names]);hi=torch.tensor([float(limits[n]['upper']) for n in cfg.kinematic_info.dof_names])
model=mujoco.MjModel.from_xml_path('protomotions/data/assets/astro_p2/mjcf/astro_p2_protomotions.xml')
state=mujoco.MjData(model)
qadr=np.array([model.joint(n).qposadr[0] for n in cfg.kinematic_info.dof_names])
bodyids=np.array([model.body(n).id for n in cfg.kinematic_info.body_names])
fk_max_error=0.;fk_frames=0;speed_outliers=[]
files=[];frames=0;fps_counts={};max_speed=0.;counts_speed=0;total_joints=0
for i,f in enumerate(src['motion_files']):
 rel=Path(f).relative_to(srcroot/'motions');out=a.output_dir/'motions'/rel
 d=torch.load(out,map_location='cpu',weights_only=False)
 n=int(src['motion_num_frames'][i]);assert d['dof_pos'].shape==(n,30),(rel,'shape');assert d['rigid_body_pos'].shape==(n,31,3)
 assert abs(float(d['fps'])-1/float(src['motion_dt'][i]))<1e-3,(rel,'fps')
 for k in ('dof_pos','dof_vel','rigid_body_pos','rigid_body_rot','rigid_body_vel','rigid_body_ang_vel'):assert torch.isfinite(d[k]).all(),(rel,k)
 assert ((d['dof_pos']>=lo-1e-5)&(d['dof_pos']<=hi+1e-5)).all(),(rel,'limits')
 assert torch.allclose(d['rigid_body_rot'].norm(dim=-1),torch.ones(n,31),atol=1e-4),(rel,'quat')
 assert d['rigid_body_contacts'].shape==(n,31)
 for t in sorted(set((0,n//2,n-1))):
  state.qpos[:3]=d['rigid_body_pos'][t,0].numpy()
  state.qpos[3:7]=d['rigid_body_rot'][t,0].numpy()[[3,0,1,2]]
  state.qpos[qadr]=d['dof_pos'][t].numpy()
  mujoco.mj_kinematics(model,state)
  err=float(np.abs(state.xpos[bodyids]-d['rigid_body_pos'][t].numpy()).max())
  assert err<1e-4,(rel,'independent FK',err)
  fk_max_error=max(fk_max_error,err);fk_frames+=1
 vel=(d['dof_pos'][1:]-d['dof_pos'][:-1]).abs()*float(d['fps'])
 vmax=torch.tensor([float(limits[n]['velocity']) for n in cfg.kinematic_info.dof_names])
 max_speed=max(max_speed,float(vel.max()) if vel.numel() else 0.)
 exceed=int((vel>vmax+0.01).sum())
 counts_speed+=exceed;total_joints+=vel.numel()
 if exceed:
  speed_outliers.append(dict(file=str(rel),peak_speed=float(vel.max()),exceed_count=exceed,fraction=exceed/max(vel.numel(),1)))
 frames+=n;fps_counts[str(d['fps'])]=fps_counts.get(str(d['fps']),0)+1
 files.append({'file':str(out),'weight':float(src['motion_weights'][i])})
 if i%1000==0:print(f'VALIDATE {i}/{len(src["motion_files"])}',flush=True)
assert frames==int(src['motion_num_frames'].sum())
assert len(list((a.output_dir/'motions').rglob('*.motion')))==len(files)
yml=a.output_dir/'motions_in_source_order.yaml';yml.write_text(yaml.safe_dump({'motions':files},sort_keys=False))
print('PACKAGING',flush=True)
lib=MotionLib(MotionLibConfig(motion_file=str(yml)),device='cpu')
assert lib.num_motions()==len(files)
for k in ('motion_dt','motion_num_frames','motion_lengths','motion_weights','length_starts'):assert torch.allclose(getattr(lib,k),src[k]),k
tmp=a.output_dir/'amass_astro_p2.partial.pt';lib.save_to_file(tmp);os.replace(tmp,a.output_dir/'amass_astro_p2.pt')
report=dict(motions=len(files),frames=frames,source=str(srcroot/'amass_g1.pt'),dofs=30,bodies=31,fps_counts=fps_counts,validation='All files: shape, finite values, normalized quaternion, P2 joint bounds, source frames/fps/weights/order preserved',peak_joint_speed=max_speed,joint_velocity_limit_exceed_fraction=counts_speed/max(total_joints,1),motions_exceeding_velocity_limits=len(speed_outliers),largest_velocity_outliers=sorted(speed_outliers,key=lambda r:-r['peak_speed'])[:20],independent_mujoco_fk_frames=fk_frames,independent_mujoco_fk_max_error=fk_max_error)
iterations=[]
for log in a.output_dir.glob('worker_*.log'):
 iterations.extend(int(v) for v in re.findall(r'iterations=(\d+) cost=',log.read_text()))
report['solver_chunks']=len(iterations)
report['solver_chunks_reaching_iteration_cap']=sum(v>=100 for v in iterations)
report['solver_iterations_median']=float(np.median(iterations)) if iterations else None
report['solver_iterations_p95']=float(np.percentile(iterations,95)) if iterations else None
(a.output_dir/'validation_report.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
