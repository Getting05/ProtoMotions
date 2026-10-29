"""Episode files and group-based splits; windows never cross a reset boundary."""
import hashlib
import json
from pathlib import Path
import re
import torch
from torch.utils.data import Dataset
from .sensors import corrupt_episode

SCHEMA_VERSION = 1


def canonical_motion_group(name):
    """Convenience only. Manifest group_id must also group crops/retargeted variants."""
    stem = Path(name).stem
    return re.sub(r'_M$', '', stem)


def split_for_group(group,seed=42,validation_fraction=.2):
    number = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:16],16)/2**64
    return 'validation' if number < validation_fraction else 'train'


def read_index(directory):
    entries = json.loads((Path(directory)/'index.json').read_text())
    if entries['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported dataset schema')
    groups = {}
    for ep in entries['episodes']:
        if ep['split'] not in ('train','validation','test'): raise ValueError('Unknown dataset split')
        prior = groups.setdefault(ep['group_id'],ep['split'])
        if prior != ep['split']:
            raise ValueError('Motion family appears in multiple dataset splits')
    return entries


class EpisodeWindows(Dataset):
    def __init__(self,directory,split,history=25,noise=None,seed=42,stride=1,window_history=None):
        self.index=read_index(directory)
        self.history=history
        # Match supervised target frames across history-length ablations. The
        # returned input still contains exactly `history` frames, without padding.
        self.window_history=history if window_history is None else window_history
        if history < 1 or self.window_history < history:
            raise ValueError('window_history must be >= history >= 1')
        self.episodes,self.windows=[],[]
        generator=torch.Generator().manual_seed(seed)
        for meta in self.index['episodes']:
            if meta['split'] != split: continue
            ep=torch.load(Path(directory)/meta['file'],map_location='cpu',weights_only=True)
            n=len(ep['proprio'])
            if ep['proprio'].shape != (n,129) or ep['target'].shape != (n,4) or ep['contact'].shape != (n,2):
                raise ValueError('Episode shape mismatch')
            if not all(torch.isfinite(ep[k]).all() for k in ('proprio','target','contact')):
                raise ValueError('Nonfinite training tensors')
            # Finite values alone do not catch a diverged physics rollout. Reset/IMU
            # warmup frames are already excluded from every training window.
            valid=ep['valid']
            bad=(ep['proprio'].abs().amax(-1)>1e5)|(ep['target'][:,1:].abs().amax(-1)>20)|(ep['target'][:,0].abs()>5)
            if 'fk_audit' in ep:
                bad=bad|(ep['fk_audit'][:,0]>.005)|(ep['fk_audit'][:,1]>.02)
            if (bad&valid).any():
                raise ValueError(f'Physically invalid rollout: {meta["file"]}; quarantine or recollect before training')
            intervals=ep['time'][1:]-ep['time'][:-1]
            if not torch.allclose(intervals,torch.full_like(intervals,self.index['dt']),atol=1e-5,rtol=1e-4):
                raise ValueError('Episode timestamps do not match the control period')
            x,delay = corrupt_episode(ep['proprio'],noise or {},generator)
            ep={**ep,'proprio':x,'meta':meta}
            i=len(self.episodes); self.episodes.append(ep)
            # No padded training windows; derivative-invalid/reset/delayed samples excluded.
            for end in range(self.window_history-1+delay,n,stride):
                if ep['valid'][end-self.window_history+1:end+1].all():
                    self.windows.append((i,end))
        if not self.windows:
            raise ValueError(f'No valid {split} windows; collect disjoint motion groups and longer episodes')

    def __len__(self): return len(self.windows)

    def __getitem__(self,index):
        i,end=self.windows[index]; ep=self.episodes[i]
        start=end-self.history+1
        sample=dict(proprio=ep['proprio'][start:end+1],target=ep['target'][end],
                    contact=ep['contact'][start:end+1],episode=i,end=end)
        if 'candidates' in ep: sample['candidates']=ep['candidates'][start:end+1]
        return sample

    @torch.no_grad()
    def cache_candidates(self,physics,device,chunk_size=2048):
        # Augmentation is fixed per episode. Cache from the exact measured inputs;
        # never recompute identical overlapping-window FK during every epoch.
        for ep in self.episodes:
            ep['candidates']=torch.cat([physics(chunk.to(device)).cpu()
                for chunk in ep['proprio'].split(chunk_size)])

    def normalization(self):
        # Compute exclusively from this split, once per frame (not overlapping windows).
        x=torch.cat([e['proprio'][e['valid']] for e in self.episodes])
        return x.mean(0),x.std(0,unbiased=False).clamp_min(.01)


def write_episode(directory,episode,metadata):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    path=directory/metadata['file']
    if path.exists(): raise FileExistsError(path)
    torch.save({k:v.detach().cpu() for k,v in episode.items()},path)
