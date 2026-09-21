"""Supervise resumable GPU retargeting, validation, packaging, and eight shards."""
import os,json,time,subprocess,sys,fcntl
from pathlib import Path
repo=Path('/data/chenguanting/ProtoMotions')
work=Path('/data/chenguanting/workspace/astro_p2_retarget')
output=Path('/data/chenguanting/datasets/AMASS_Retargeted_for_G1/protomotions_astro_p2')
python='/data/chenguanting/.venvs/astro-p2-retarget/bin/python'
proto='/data/chenguanting/.venvs/protomotions-newton/bin/python'
source=Path('/data/chenguanting/datasets/AMASS_Retargeted_for_G1/protomotions_g1/motions')
output.mkdir(parents=True,exist_ok=True)
lock=(output/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rows=json.loads((work/'source_motion_index.json').read_text());gpus=list(range(8))
groups=[[] for _ in gpus];totals=[0]*len(gpus)
for row in sorted(rows,key=lambda x:(-x['frames'],x['file'])):
 j=min(range(len(gpus)),key=lambda j:(totals[j],len(groups[j]),j));groups[j].append(row['file']);totals[j]+=row['frames']
common=[python,'-u','retarget_g1_motion_to_astro_p2.py','--motion-dir',str(source),'--output-dir',str(output/'motions'),
 '--astro-urdf-path','protomotions/data/assets/astro_p2/urdf/astro_p2_30dof_primitive_collision.urdf',
 '--astro-mjcf-path','protomotions/data/assets/astro_p2/mjcf/astro_p2_protomotions.xml',
 '--chunk-frames','120','--chunk-overlap','20','--max-iterations','100','--fix-height','--skip-existing']
env=os.environ.copy();env.update(PYTHONPATH=str(repo),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_PLATFORMS='cuda,cpu')
env['XLA_FLAGS']='--xla_gpu_enable_triton_gemm=false --xla_gpu_triton_gemm_any=false'
env['PATH']='/usr/local/cuda/bin:'+env['PATH']
env['LD_LIBRARY_PATH']='/usr/local/cuda/lib64:/data/chenguanting/.venvs/protomotions-isaaclab/lib/python3.12/site-packages/nvidia/cudnn/lib:/data/chenguanting/.venvs/protomotions-newton/lib/python3.11/site-packages/nvidia/nccl/lib'
settings=dict(command=common,gpus=gpus,source_motions=len(rows),source_frames=sum(r['frames'] for r in rows),started=time.time())
(output/'run_config.json').write_text(json.dumps(settings,indent=2))
processes=[]
for j,gpu in enumerate(gpus):
 f=output/f'worker_{j}_files.json';f.write_text(json.dumps(sorted(groups[j])))
 log=(output/f'worker_{j}.log').open('a');e=env.copy();e['CUDA_VISIBLE_DEVICES']=str(gpu)
 cmd=common+['--file-list',str(f),'--manifest',str(output/f'worker_{j}_manifest.jsonl')]
 proc=subprocess.Popen(cmd,cwd=repo,env=e,stdout=log,stderr=subprocess.STDOUT);processes.append((proc,log))
 print(f'WORKER {j} GPU {gpu} PID {proc.pid} motions={len(groups[j])} frames={totals[j]}',flush=True)
while any(p.poll() is None for p,_ in processes):
 progress=[]
 for j in range(len(gpus)):
  f=output/f'worker_{j}_manifest.jsonl';progress.append(sum(1 for line in f.open() if '"status": "ok"' in line) if f.exists() else 0)
 (output/'progress.json').write_text(json.dumps(dict(completed=sum(progress),total=len(rows),workers=progress,seconds=time.time()-settings['started'],returncodes=[p.poll() for p,_ in processes])))
 print(f'PROGRESS {sum(progress)}/{len(rows)} workers={progress}',flush=True)
 time.sleep(30)
for p,l in processes:l.close()
codes=[p.returncode for p,_ in processes]
if any(codes):raise RuntimeError(f'Retarget workers failed: {codes}; see worker logs. Packaging withheld.')
packenv=os.environ.copy();packenv.update(PYTHONPATH=str(repo),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
for name,cmd in [('package',[proto,'-u','scripts/package_astro_p2.py',str(output)]),('shard',[proto,'-u','scripts/shard_astro_p2_motion_lib.py',str(output/'amass_astro_p2.pt'),str(output/'shards_8'),'--shards','8'])]:
 print(name.upper(),flush=True)
 with (output/f'{name}.log').open('w') as log:subprocess.run(cmd,cwd=repo,env=packenv,stdout=log,stderr=subprocess.STDOUT,check=True)
# Since P2 preserves frame counts and source order, membership must exactly match G1.
g1=json.loads((source.parent/'shards_8/sharding_manifest.json').read_text())
p2=json.loads((output/'shards_8/sharding_manifest.json').read_text())
for a,b in zip(g1['shards'],p2['shards']):
 assert a['original_motion_indices']==b['original_motion_indices']
 assert a['frames']==b['frames'] and a['motions']==b['motions']
(output/'COMPLETE.json').write_text(json.dumps(dict(motions=len(rows),frames=sum(r['frames'] for r in rows),shards=8,exact_g1_membership=True,seconds=time.time()-settings['started']),indent=2))
print('COMPLETE',flush=True)
