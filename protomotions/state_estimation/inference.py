"""Real robot API: measured proprioception only. Fixed window matches training."""
import torch
from .models import PelvisEstimator
from .physics import FootKinematics


class RealtimeEstimator:
    def __init__(self,checkpoint,device='cpu',urdf=None):
        bundle=torch.load(checkpoint,map_location=device,weights_only=True)
        config=bundle['config']; contract=bundle['contract']
        if bundle['mode']=='oracle': raise ValueError('Oracle checkpoint is not deployable')
        self.device=torch.device(device)
        self.physics=FootKinematics(urdf or contract['urdf'],contract['dof_names'],contract['body_names'],contract['imu']).to(device)
        if self.physics.sha256 != contract['sha256']: raise ValueError('URDF checksum differs from training')
        self.model=PelvisEstimator(mode=bundle['mode'],**config['model']).to(device).eval()
        self.model.load_state_dict(bundle['model'])
        self.history_length=config['history_length']
        self.dt=1./config['control_frequency']
        self.reset()

    def reset(self):
        self.history=[]; self.candidates=[]; self.last_time=None

    @torch.inference_mode()
    def update(self,proprio,timestamp):
        x=torch.as_tensor(proprio,device=self.device,dtype=torch.float32)
        if x.shape != (129,) or not torch.isfinite(x).all():
            raise ValueError('Expected finite 129D measured state in contract joint order')
        if torch.linalg.vector_norm(x[90:93]) < .5:
            raise ValueError('Projected gravity must be a valid unit vector')
        if self.last_time is not None and abs(timestamp-self.last_time-self.dt)>self.dt*.25:
            self.reset()
        self.last_time=float(timestamp)
        self.history.append(x.clone()); self.candidates.append(self.physics(x))
        self.history=self.history[-self.history_length:]; self.candidates=self.candidates[-self.history_length:]
        # Explicit warmup; consumers must not use a padded-history prediction as valid.
        if len(self.history)<self.history_length: return None
        result=self.model(torch.stack(self.history)[None],torch.stack(self.candidates)[None])
        return {k:v[0,-1].clone() for k,v in result.items()}
