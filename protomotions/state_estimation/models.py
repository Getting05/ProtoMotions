"""Pure GRU baseline and contact-gated physics/GRU fusion."""
import torch
from torch import nn


def fuse_candidates(candidates, p, thresholds=(.08,.3,.3,.3)):
    kin = (p[...,None]*candidates).sum(-2)/(p.sum(-1,keepdim=True)+1e-8)
    disagreement = ((candidates[...,0,:]-candidates[...,1,:]) /
                    candidates.new_tensor(thresholds)).square().mean(-1,keepdim=True)
    # Single reliable support is not penalized for an arbitrary swing-foot candidate.
    agreement = 1 - (p[...,0:1]*p[...,1:2])*(1-torch.exp(-disagreement))
    confidence = (1-(1-p[...,0:1])*(1-p[...,1:2]))*agreement
    return kin, confidence


class ContactEstimator(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(129,hidden),nn.ELU())
        self.gru = nn.GRU(hidden,hidden,batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden,64),nn.ELU(),nn.Linear(64,2))

    def forward(self,x):
        y,_ = self.gru(self.embed(x))
        return self.head(y)


class PelvisEstimator(nn.Module):
    def __init__(self, mode='fusion', hidden=256, layers=2, contact_hidden=128,
                 disagreement=(.08,.3,.3,.3)):
        super().__init__()
        if mode not in ('baseline','fusion','oracle'):
            raise ValueError(mode)
        self.mode, self.disagreement = mode, disagreement
        self.register_buffer('input_mean',torch.zeros(129))
        self.register_buffer('input_std',torch.ones(129))
        self.register_buffer('target_scale',torch.tensor([.5,1.,1.,1.]))
        self.contact = ContactEstimator(contact_hidden)
        self.embed = nn.Sequential(nn.Linear(129 if mode=='baseline' else 144,hidden),nn.ELU())
        self.gru = nn.GRU(hidden,hidden,num_layers=layers,batch_first=True)
        self.head = nn.Linear(hidden,4 if mode=='baseline' else 8)

    def forward(self,proprio,candidates,contact_override=None):
        x = (proprio-self.input_mean)/self.input_std.clamp_min(1e-5)
        logits = self.contact(x)
        p = logits.sigmoid()
        if self.mode == 'oracle':
            if contact_override is None:
                raise ValueError('Oracle ablation requires supervision; it is not deployable')
            p = contact_override.to(x.dtype)
        elif contact_override is not None:
            raise ValueError('Contact truth is prohibited for deployable models')
        kin,c = fuse_candidates(candidates,p,self.disagreement)
        z = x if self.mode=='baseline' else torch.cat((x,(candidates/self.target_scale).flatten(-2),p,
                                                       kin/self.target_scale,c),-1)
        encoded,_ = self.gru(self.embed(z))
        heads = self.head(encoded)*self.target_scale.repeat(1 if self.mode=='baseline' else 2)
        pred = heads if self.mode=='baseline' else c*(kin+heads[...,:4])+(1-c)*heads[...,4:]
        return dict(state=pred, contact_logits=logits, contact_probability=logits.sigmoid(),
                    kin=kin,confidence=c)


def supervised_loss(output,target,contact,scale,weights=(1.,1.,.2)):
    error = (output['state'][:,-1]-target)/scale
    h = torch.nn.functional.smooth_l1_loss(error[:,0],torch.zeros_like(error[:,0]))
    v = torch.nn.functional.smooth_l1_loss(error[:,1:],torch.zeros_like(error[:,1:]))
    c = torch.nn.functional.binary_cross_entropy_with_logits(output['contact_logits'][:,-1],contact)
    return weights[0]*h+weights[1]*v+weights[2]*c,dict(height=h.item(),velocity=v.item(),contact=c.item())
