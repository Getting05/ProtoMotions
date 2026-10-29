"""Separate support/stationarity predictions and height/XY/Z fusion gates.

Inputs remain observable 129D proprioception and URDF-derived candidates.
Support and no-slip labels are training supervision only, never inference inputs.
"""
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = 'split_contact_fusion_v2'


def split_candidates(candidates, support, stationary, thresholds=(.08,.3,.3,.3)):
    """Sliding support can constrain height/Z while XY requires no slip."""
    weights = torch.stack((support, stationary, stationary, support), -1)
    kin = (weights*candidates).sum(-2)/weights.sum(-2).clamp_min(1e-8)
    difference = (candidates[...,0,:]-candidates[...,1,:])/candidates.new_tensor(thresholds)
    disagreement = torch.stack((difference[...,0].square(),
                                difference[...,1:3].square().mean(-1),
                                difference[...,3].square()),-1)
    p = torch.stack((support,stationary,support),-1)
    any_support = 1-(1-p[...,0,:])*(1-p[...,1,:])
    agreement = 1-p[...,0,:]*p[...,1,:]*(1-torch.exp(-disagreement))
    return kin, any_support*agreement


class SplitContactEstimator(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(129,hidden),nn.ELU())
        self.gru=nn.GRU(hidden,hidden,batch_first=True)
        self.head=nn.Sequential(nn.Linear(hidden,64),nn.ELU(),nn.Linear(64,4))

    def forward(self,x):
        y,_=self.gru(self.embed(x))
        return self.head(y)


class SplitFusionEstimator(nn.Module):
    def __init__(self,hidden=256,layers=2,contact_hidden=128,disagreement=(.08,.3,.3,.3)):
        super().__init__()
        self.mode='fusion';self.disagreement=disagreement
        self.register_buffer('input_mean',torch.zeros(129))
        self.register_buffer('input_std',torch.ones(129))
        self.register_buffer('target_scale',torch.tensor([.5,1.,1.,1.]))
        self.contact=SplitContactEstimator(contact_hidden)
        self.embed=nn.Sequential(nn.Linear(148,hidden),nn.ELU())
        self.gru=nn.GRU(hidden,hidden,num_layers=layers,batch_first=True)
        # Four FK residuals, four direct outputs, three learned confidence factors.
        self.head=nn.Linear(hidden,11)
        with torch.no_grad():self.head.bias[8:].fill_(2.)

    def forward(self,proprio,candidates,contact_override=None):
        if contact_override is not None:
            raise ValueError('True support/contact is prohibited at inference')
        x=(proprio-self.input_mean)/self.input_std.clamp_min(1e-5)
        logits=self.contact(x);support=logits[...,:2].sigmoid()
        stationary=support*logits[...,2:].sigmoid()
        kin,base=split_candidates(candidates,support,stationary,self.disagreement)
        z=torch.cat((x,(candidates/self.target_scale).flatten(-2),support,stationary,
                     kin/self.target_scale,base),-1)
        encoded,_=self.gru(self.embed(z));head=self.head(encoded)
        group_confidence=base*head[...,8:].sigmoid()
        confidence=group_confidence[...,[0,1,1,2]]
        residual=head[...,:4]*self.target_scale;direct=head[...,4:8]*self.target_scale
        state=confidence*(kin+residual)+(1-confidence)*direct
        return dict(state=state,contact_probability=stationary,
                    contact_logits=torch.logit(stationary.clamp(1e-6,1-1e-6)),
                    support_probability=support,support_logits=logits[...,:2],
                    stationary_logits=logits[...,2:],kin=kin,confidence=confidence,
                    group_confidence=group_confidence,base_confidence=base)


def split_loss(output,target,contact,support,scale,weights=(1.,1.,.2)):
    error=(output['state'][:,-1]-target)/scale
    h=F.smooth_l1_loss(error[:,0],torch.zeros_like(error[:,0]))
    v=F.smooth_l1_loss(error[:,1:],torch.zeros_like(error[:,1:]))
    s=F.binary_cross_entropy_with_logits(output['support_logits'][:,-1],support)
    conditional=F.binary_cross_entropy_with_logits(output['stationary_logits'][:,-1],contact,reduction='none')
    static=(conditional*support).sum()/support.sum().clamp_min(1.)
    auxiliary=.5*(s+static)
    return weights[0]*h+weights[1]*v+weights[2]*auxiliary,dict(height=h.item(),velocity=v.item(),support=s.item(),stationary=static.item())


def build_estimator(bundle):
    architecture=bundle.get('architecture','legacy')
    if architecture=='legacy':
        from .models import PelvisEstimator
        return PelvisEstimator(mode=bundle['mode'],**bundle['config']['model'])
    if architecture==ARCHITECTURE:
        if bundle['mode']!='fusion':raise ValueError('V2 only supports deployable fusion')
        return SplitFusionEstimator(**bundle['config']['model'])
    raise ValueError('Unknown estimator architecture: '+str(architecture))
