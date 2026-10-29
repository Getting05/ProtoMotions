"""Train split-contact fusion on the unchanged grouped P2 dataset."""
import argparse,json,time,hashlib,random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import EpisodeWindows
from .sensors import corrupt_episode
from .physics import FootKinematics
from .models_v2 import SplitFusionEstimator,split_loss,ARCHITECTURE


class SupportWindows(EpisodeWindows):
    def __init__(self,*args,force_min=30.,**kwargs):
        super().__init__(*args,**kwargs)
        for ep in self.episodes:
            ep['support']=((ep['normal_force']>force_min)&(ep['contact_count']>0)).float()
            if (ep['contact']>ep['support']).any():raise ValueError('Stationary label without support')

    def __getitem__(self,index):
        item=super().__getitem__(index);i,end=self.windows[index]
        item['support']=self.episodes[i]['support'][end]
        return item


@torch.no_grad()
def cache_batched(ds,physics,device,chunk=4096):
    lengths=[len(e['proprio']) for e in ds.episodes]
    x=torch.cat([e['proprio'] for e in ds.episodes])
    values=torch.cat([physics(part.to(device)).cpu() for part in x.split(chunk)])
    for ep,value in zip(ds.episodes,values.split(lengths)):ep['candidates']=value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--resample-noise',action='store_true');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--epochs',type=int);p.add_argument('--max-batches',type=int,default=0)
    a=p.parse_args();config=json.loads(Path(a.config).read_text())
    if a.epochs is not None:config['epochs']=a.epochs
    if config['noise'].get('max_delay_frames',0):raise ValueError('Matched comparison currently requires zero delay')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(a.threads);torch.manual_seed(config['seed']);random.seed(config['seed'])
    started=time.perf_counter()
    window_history=config.get('window_history',config['history_length'])
    train=SupportWindows(a.data,'train',config['history_length'],config['noise'],config['seed'],force_min=config['contact']['force_min'],window_history=window_history)
    val=SupportWindows(a.data,'validation',config['history_length'],force_min=config['contact']['force_min'],window_history=window_history)
    if abs(train.index['dt']-1/config['control_frequency'])>1e-8:raise ValueError('Frequency mismatch')
    contract=train.index['contract']
    physics=FootKinematics(contract['urdf'],contract['dof_names'],contract['body_names'],contract['imu']).to(a.device)
    if physics.sha256!=contract['sha256']:raise ValueError('URDF changed')
    raw=None
    if a.resample_noise:
        raw=[torch.load(Path(a.data)/ep['meta']['file'],weights_only=True)['proprio'] for ep in train.episodes]
    cache_batched(train,physics,a.device);cache_batched(val,physics,a.device)
    model=SplitFusionEstimator(**config['model']).to(a.device)
    mean,std=train.normalization();model.input_mean.copy_(mean);model.input_std.copy_(std)
    model.target_scale.copy_(torch.tensor(config['target_scale']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    loaders=[DataLoader(ds,batch_size=config['batch_size'],shuffle=i==0,num_workers=0,
             generator=torch.Generator().manual_seed(config['seed'])) for i,ds in enumerate((train,val))]
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    metadata=dict(architecture=ARCHITECTURE,parameters=sum(p.numel() for p in model.parameters()),
        training_windows=len(train),validation_windows=len(val),resample_noise=a.resample_noise,
        selection_metric='clean_validation_state_loss',config=config,
        data_index_sha256=digest(Path(a.data)/'index.json'),
        epoch0_noisy_inputs_sha256=hashlib.sha256(torch.cat([e['proprio'] for e in train.episodes]).numpy().tobytes()).hexdigest(),
        train_windows_sha256=hashlib.sha256(json.dumps(train.windows).encode()).hexdigest(),
        validation_windows_sha256=hashlib.sha256(json.dumps(val.windows).encode()).hexdigest(),
        initial_parameters_sha256=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest(),
        window_history=window_history,
        source_sha256={p:digest(p) for p in (__file__,'protomotions/state_estimation/models_v2.py','protomotions/state_estimation/data.py')},
        labels='support=(normal_force>30N)&contact_count>0; conditional stationary supervised by existing reliable-contact label; supervision only',
        startup='Runtime requires history_length valid frames; comparison evaluator may enforce a longer shared FK bootstrap.',
        comparison='Fixed noise unless resample_noise is enabled; optional window_history aligns supervision endpoints across history lengths. Selection uses clean validation only.')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2));(output/'config.json').write_text(json.dumps(config,indent=2))
    print('TRAIN_SETUP '+json.dumps(metadata),flush=True)
    best=float('inf');history=[]
    for epoch in range(config['epochs']):
        if a.resample_noise and epoch:
            generator=torch.Generator().manual_seed(config['seed']+1000003*epoch)
            for ep,x in zip(train.episodes,raw):ep['proprio'],_=corrupt_episode(x,config['noise'],generator)
            cache_batched(train,physics,a.device)
        row=dict(epoch=epoch)
        for phase,loader in zip(('train','validation'),loaders):
            model.train(phase=='train');total=state_total=0.;samples=0
            with torch.set_grad_enabled(phase=='train'):
                for batch_no,b in enumerate(loader):
                    pred=model(b['proprio'].to(a.device),b['candidates'].to(a.device))
                    loss,parts=split_loss(pred,b['target'].to(a.device),b['contact'][:,-1].to(a.device),
                        b['support'].to(a.device),model.target_scale,config['loss_weights'])
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    if phase=='train':
                        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    n=len(b['proprio']);samples+=n;total+=loss.item()*n
                    state_total+=(config['loss_weights'][0]*parts['height']+config['loss_weights'][1]*parts['velocity'])*n
                    if a.max_batches and batch_no+1>=a.max_batches:break
            row[phase]=total/samples;row[phase+'_state_loss']=state_total/samples
        row['elapsed_seconds']=time.perf_counter()-started
        bundle=dict(schema_version=1,architecture=ARCHITECTURE,model=model.state_dict(),mode='fusion',config=config,
            contract=contract,epoch=epoch,validation_loss=row['validation'],validation_state_loss=row['validation_state_loss'],
            selection_metric='validation_state_loss',training_windows=len(train),validation_windows=len(val),
            data_index=train.index,smoke_only=bool(a.max_batches),resample_noise=a.resample_noise)
        torch.save(bundle,output/'last.pt')
        if row['validation_state_loss']<best:best=row['validation_state_loss'];torch.save(bundle,output/'best.pt')
        history.append(row);(output/'loss.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(row),flush=True)
    print('V2_TRAIN_COMPLETE '+str(output),flush=True)


if __name__=='__main__':main()
