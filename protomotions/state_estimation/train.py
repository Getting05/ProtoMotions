"""python -m protomotions.state_estimation.train --data DATA --output RUN --mode fusion"""
import argparse
import json
from pathlib import Path
import random
import time
import torch
from torch.utils.data import DataLoader
from .data import EpisodeWindows
from .physics import FootKinematics
from .models import PelvisEstimator,supervised_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/state_estimation/p2_pelvis.json')
    p.add_argument('--mode',choices=['baseline','oracle','fusion'],default='fusion')
    p.add_argument('--device',default='cuda:0'); p.add_argument('--epochs',type=int)
    p.add_argument('--max-batches',type=int,default=0,help='Smoke only; 0 uses all batches')
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args(); torch.set_num_threads(a.threads)
    config=json.loads(Path(a.config).read_text())
    if a.epochs is not None: config['epochs']=a.epochs
    torch.manual_seed(config['seed']); random.seed(config['seed'])
    output=Path(a.output); output.mkdir(parents=True,exist_ok=False)
    train=EpisodeWindows(a.data,'train',config['history_length'],config['noise'],config['seed'])
    val=EpisodeWindows(a.data,'validation',config['history_length'])
    if abs(train.index['dt']-1/config['control_frequency'])>1e-8: raise ValueError('Control frequency mismatch')
    contract=train.index['contract']
    physics=FootKinematics(contract['urdf'],contract['dof_names'],contract['body_names'],contract['imu']).to(a.device)
    if physics.sha256 != contract['sha256']: raise ValueError('Dataset URDF checksum mismatch')
    if a.mode!='baseline':
        train.cache_candidates(physics,a.device); val.cache_candidates(physics,a.device)
    model=PelvisEstimator(mode=a.mode,**config['model']).to(a.device)
    mean,std=train.normalization()
    model.input_mean.copy_(mean); model.input_std.copy_(std)
    model.target_scale.copy_(torch.tensor(config['target_scale']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    loaders=[DataLoader(ds,batch_size=config['batch_size'],shuffle=i==0,num_workers=0,
                        generator=torch.Generator().manual_seed(config['seed']))
             for i,ds in enumerate([train,val])]
    best=float('inf'); history=[]; started=time.perf_counter()
    (output/'config.json').write_text(json.dumps(config,indent=2))
    for epoch in range(config['epochs']):
        row={'epoch':epoch}
        for phase,loader in zip(['train','validation'],loaders):
            model.train(phase=='train'); total=0.; state_total=0.; samples=0
            with torch.set_grad_enabled(phase=='train'):
                for batch_no,batch in enumerate(loader):
                    x=batch['proprio'].to(a.device); y=batch['target'].to(a.device); c=batch['contact'].to(a.device)
                    candidates=batch['candidates'].to(a.device) if a.mode!='baseline' else x.new_zeros(*x.shape[:2],2,4)
                    prediction=model(x,candidates,c if a.mode=='oracle' else None)
                    loss,parts=supervised_loss(prediction,y,c[:,-1],model.target_scale,config['loss_weights'])
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                    if phase=='train':
                        optimizer.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
                    total+=loss.item()*len(x); samples+=len(x)
                    state_total+=(config['loss_weights'][0]*parts['height']+config['loss_weights'][1]*parts['velocity'])*len(x)
                    if a.max_batches and batch_no+1>=a.max_batches: break
            row[phase]=total/samples
            row[phase+'_state_loss']=state_total/samples
        row['elapsed_seconds']=time.perf_counter()-started
        bundle=dict(schema_version=1,model=model.state_dict(),mode=a.mode,config=config,
                    contract=contract,epoch=epoch,validation_loss=row['validation'],validation_state_loss=row['validation_state_loss'],
                    selection_metric='validation_state_loss',training_windows=len(train),validation_windows=len(val),
                    data_index=train.index,smoke_only=bool(a.max_batches))
        torch.save(bundle,output/'last.pt')
        if row['validation_state_loss']<best:
            best=row['validation_state_loss']; torch.save(bundle,output/'best.pt')
        history.append(row)
        (output/'loss.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(row),flush=True)


if __name__=='__main__': main()
