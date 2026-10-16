"""Simulation-only sensor synthesis and supervision. Never imported by inference."""
import torch
from .physics import mv, heading_inverse


def origin_velocity(com_velocity,omega,origin_position,com_position):
    return com_velocity + torch.linalg.cross(omega,origin_position-com_position)


def point_velocity(velocity,omega,origin,point):
    return velocity + torch.linalg.cross(omega,point-origin)


def pelvis_target(position,velocity,anchor_rotation,ground_height):
    return torch.cat(((position[...,2]-ground_height).unsqueeze(-1),
                      mv(heading_inverse(anchor_rotation),velocity)),-1)


def reliable_contact(normal_force,slip_speed,has_point,force_min=30.,slip_max=.1):
    return ((normal_force>force_min)&(slip_speed<slip_max)&has_point&torch.isfinite(slip_speed)).float()


class IMUSynthesizer:
    """Finite difference of IMU point velocity at the control rate.

    Includes offset centripetal/tangential accelerations. Not a physics-substep
    IMU or an instantaneous acceleration sensor. Discard first post-reset sample.
    """
    def __init__(self,physics,dt):
        self.physics,self.dt = physics,dt
        self.previous = None
        self.valid = None

    def reset(self,ids=None):
        if self.valid is not None:
            if ids is None: self.valid.zero_()
            else: self.valid[ids] = False

    def sample(self,rotation,velocity,omega):
        rp = mv(rotation,self.physics.imu_position)
        imu_velocity = velocity + torch.linalg.cross(omega,rp)
        ri = rotation @ self.physics.imu_rotation
        g = velocity.new_tensor([0.,0.,-9.81]).expand_as(velocity)
        if self.previous is None:
            self.previous = imu_velocity.clone()
            self.valid = torch.zeros(velocity.shape[0],device=velocity.device,dtype=torch.bool)
        acceleration = (imu_velocity-self.previous)/self.dt
        valid = self.valid.clone()
        self.previous.copy_(imu_velocity); self.valid.fill_(True)
        return (mv(ri.transpose(-1,-2),g/9.81),mv(ri.transpose(-1,-2),omega),
                mv(ri.transpose(-1,-2),acceleration-g),valid)


def corrupt_episode(x,config,generator):
    """Only observable channels. Bias/delay are constant within one episode."""
    x = x.clone()
    groups=[(0,30,'q'),(30,60,'dq'),(60,90,'torque'),(90,93,'gravity'),
            (93,96,'gyro'),(96,99,'accel')]
    max_delay = 0
    for lo,hi,name in groups:
        std = config.get(name+'_std',0.)
        bias = config.get(name+'_bias_std',0.)
        x[:,lo:hi] += torch.randn(x[:,lo:hi].shape,generator=generator)*std
        x[:,lo:hi] += torch.randn((1,hi-lo),generator=generator)*bias
        delay = int(torch.randint(config.get('max_delay_frames',0)+1,(1,),generator=generator))
        max_delay = max(max_delay,delay)
        if delay:
            x[:,lo:hi] = torch.cat((x[:1,lo:hi].expand(delay,-1),x[:-delay,lo:hi]),0)
    x[:,90:93] = torch.nn.functional.normalize(x[:,90:93],dim=-1)
    return x,max_delay
