"""Held-out motion-family metrics, oracle coverage, diagnostic curves and latency."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import EpisodeWindows
from .models import PelvisEstimator,fuse_candidates
from .physics import FootKinematics
from .inference import RealtimeEstimator

REQUIRED_CATEGORIES=['stand','walk_forward','walk_backward','walk_sideways','turn','stop',
                     'squat','rise','run','jump','push','slip']


def metrics(pred,target):
    e=pred-target
    return dict(samples=len(e),height_mae=e[:,0].abs().mean().item(),
                vx_rmse=e[:,1].square().mean().sqrt().item(),vy_rmse=e[:,2].square().mean().sqrt().item(),
                xy_vector_rmse=e[:,1:3].square().sum(-1).mean().sqrt().item(),
                xy_rmse=e[:,1:3].square().mean().sqrt().item(),
                z_rmse=e[:,3].square().mean().sqrt().item())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--checkpoint'); p.add_argument('--physics-oracle',action='store_true')
    p.add_argument('--device',default='cuda:0'); p.add_argument('--history',type=int,default=25)
    p.add_argument('--threads',type=int,default=4); p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--noise-config',help='Explicit deployment-noise evaluation; clean by default')
    a=p.parse_args(); torch.set_num_threads(a.threads)
    if bool(a.checkpoint)==a.physics_oracle: p.error('Choose checkpoint OR physics-oracle')
    bundle=torch.load(a.checkpoint,map_location=a.device,weights_only=True) if a.checkpoint else None
    history=bundle['config']['history_length'] if bundle else a.history
    noise=json.loads(Path(a.noise_config).read_text())['noise'] if a.noise_config else None
    ds=EpisodeWindows(a.data,'validation',history,noise=noise)
    contract=ds.index['contract']
    physics=FootKinematics(contract['urdf'],contract['dof_names'],contract['body_names'],contract['imu']).to(a.device)
    if physics.sha256 != contract['sha256']: raise ValueError('Dataset URDF checksum mismatch')
    if bundle:
        if bundle['contract'] != contract: raise ValueError('Checkpoint/dataset contract differs')
        if abs(ds.index['dt']-1/bundle['config']['control_frequency'])>1e-8: raise ValueError('Frequency mismatch')
        # Prevent evaluating training groups from another supplied dataset as held out.
        training_groups={e['group_id'] for e in bundle['data_index']['episodes'] if e['split']=='train'}
        if training_groups & {e['meta']['group_id'] for e in ds.episodes}:
            raise ValueError('Evaluation contains checkpoint training motion groups')
        model=PelvisEstimator(mode=bundle['mode'],**bundle['config']['model']).to(a.device).eval()
        model.load_state_dict(bundle['model'])
    records=[]
    with torch.inference_mode():
        for b in DataLoader(ds,batch_size=a.batch_size):
            x=b['proprio'].to(a.device); contact=b['contact'].to(a.device)
            candidates=physics(x)
            if bundle:
                out=model(x,candidates,contact if bundle['mode']=='oracle' else None)
                pred=out['state'][:,-1]; prob=out['contact_probability'][:,-1]
                valid=torch.ones(len(x),device=x.device,dtype=torch.bool)
            else:
                pred,confidence=fuse_candidates(candidates[:,-1],contact[:,-1])
                prob=contact[:,-1]; valid=contact[:,-1].sum(-1)>0
            for i in range(len(x)):
                records.append(dict(episode=int(b['episode'][i]),end=int(b['end'][i]),
                                    pred=pred[i].cpu(),target=b['target'][i],p=prob[i].cpu(),
                                    contact=b['contact'][i,-1],valid=bool(valid[i])))
    output=Path(a.output); output.mkdir(parents=True,exist_ok=False)
    preds=torch.stack([r['pred'] for r in records]); truth=torch.stack([r['target'] for r in records])
    probabilities=torch.stack([r['p'] for r in records]); contacts=torch.stack([r['contact'] for r in records])
    valid=torch.tensor([r['valid'] for r in records])
    categories=[ds.episodes[r['episode']]['meta']['category'] for r in records]
    report=dict(mode=bundle['mode'] if bundle else 'physics_oracle',
                synthetic_only=ds.index.get('synthetic_only',False),noise=noise,
                supported_coverage=valid.float().mean().item(),
                aggregate=metrics(preds[valid],truth[valid]) if valid.any() else None,
                categories={},events={},missing_categories=sorted(set(REQUIRED_CATEGORIES)-set(categories)))
    limits=(bundle or {}).get('config',{}).get('acceptance',dict(height_mae=.03,xy_rmse=.1,z_rmse=.15))
    for category in sorted(set(categories)):
        mask=torch.tensor([c==category for c in categories])&valid
        m=metrics(preds[mask],truth[mask]) if mask.any() else dict(samples=0)
        m['meets_error_targets']=bool(mask.any()) and all(m[k]<=limits[k] for k in ('height_mae','xy_rmse','z_rmse'))
        report['categories'][category]=m
    for key in ('airborne','landing','slip','push'):
        mask=torch.tensor([bool(ds.episodes[r['episode']].get(key,torch.zeros(len(ds.episodes[r['episode']]['time'])))[r['end']]) for r in records])&valid
        event=metrics(preds[mask],truth[mask]) if mask.any() else dict(samples=0)
        event['meets_error_targets']=bool(mask.any()) and all(event[k]<=limits[k] for k in ('height_mae','xy_rmse','z_rmse'))
        report['events'][key]=event
    false_support=(probabilities>.5)&(contacts<.5)
    report['contact_false_positive_rate']=(false_support.sum()/(contacts<.5).sum().clamp_min(1)).item()
    if bundle and bundle['mode']!='oracle':
        runtime=RealtimeEstimator(a.checkpoint,a.device)
        x=ds.episodes[0]['proprio']; timings=[]
        for i in range(history+120):
            frame=x[i%len(x)]
            if str(a.device).startswith('cuda'): torch.cuda.synchronize()
            start=time.perf_counter(); runtime.update(frame,i*runtime.dt)
            if str(a.device).startswith('cuda'): torch.cuda.synchronize()
            if i>=history+20: timings.append((time.perf_counter()-start)*1000)
        report['runtime_ms']=dict(median=float(np.median(timings)),p95=float(np.percentile(timings,95)),
                                  device=a.device,threads=a.threads,includes_fk_and_window=True)
    report['acceptance_passed']=bool(bundle and not report['synthetic_only'] and not bundle.get('smoke_only')
        and bundle['mode']!='oracle' and not report['missing_categories']
        and all(m['meets_error_targets'] for m in report['categories'].values())
        and all(report['events'][k]['meets_error_targets'] for k in ('airborne','landing','slip','push'))
        and report.get('runtime_ms',{}).get('p95',float('inf'))<=limits.get('latency_p95_ms',20.))
    torch.save(records,output/'curves.pt')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # One full episode per available category; all numerical curves are saved above.
    for category in sorted(set(categories)):
        episode=next(r['episode'] for r,c in zip(records,categories) if c==category)
        selected=[r for r in records if r['episode']==episode]
        t=np.array([float(ds.episodes[episode]['time'][r['end']]) for r in selected])
        y=torch.stack([r['target'] for r in selected]).numpy()
        yhat=torch.stack([r['pred'] for r in selected]).numpy()
        cp=torch.stack([r['p'] for r in selected]).numpy(); ct=torch.stack([r['contact'] for r in selected]).numpy()
        fig,axes=plt.subplots(6,1,figsize=(11,15),sharex=True)
        for j,name in enumerate(['height (m)','vx (m/s)','vy (m/s)','vz (m/s)']):
            axes[j].plot(t,y[:,j],label='truth'); axes[j].plot(t,yhat[:,j],label='estimate'); axes[j].set_ylabel(name); axes[j].legend()
        for j,name in enumerate(['height (m)','vx (m/s)','vy (m/s)','vz (m/s)']):
            axes[4].plot(t,yhat[:,j]-y[:,j],label=name)
        axes[4].set_ylabel('estimate - truth'); axes[4].legend(ncol=4)
        for j,name in enumerate(['left','right']):
            axes[5].plot(t,cp[:,j],label=name+' predicted'); axes[5].plot(t,ct[:,j],'--',label=name+' reliable truth')
        axes[5].legend(); axes[5].set_xlabel('episode time (s)')
        fig.suptitle(('SYNTHETIC SOFTWARE TEST | ' if report['synthetic_only'] else '')+report['mode']+' | '+category)
        fig.tight_layout(rect=(0,0,1,.98))
        fig.savefig(output/('category_'+str(sorted(set(categories)).index(category))+'.png')); plt.close(fig)
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
