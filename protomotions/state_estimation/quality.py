"""Quality checks for measured P2 rollouts, including finite-but-diverged physics.

No tracking-error filter is used: valid slips, impacts and failures remain data.
Contact-point NaNs and slip infinity are legal only when no contact point exists.
"""
import torch

LIMITS = dict(fk_position=.005, fk_velocity=.02, proprio_absolute=1e5,
              pelvis_velocity=20., pelvis_height=5., gravity_norm_error=.002)


def invalid_frames(ep):
    x,y=ep['proprio'],ep['target'];valid=ep['valid'].bool()
    bad=~torch.isfinite(x).all(-1)|~torch.isfinite(y).all(-1)
    bad |= ~torch.isfinite(ep['fk_audit']).all(-1)
    physical=(x.abs().amax(-1)>LIMITS['proprio_absolute'])
    physical |= y[...,1:].abs().amax(-1)>LIMITS['pelvis_velocity']
    physical |= y[...,0].abs()>LIMITS['pelvis_height']
    physical |= ep['fk_audit'][...,0]>LIMITS['fk_position']
    physical |= ep['fk_audit'][...,1]>LIMITS['fk_velocity']
    physical |= (x[...,90:93].norm(dim=-1)-1).abs()>LIMITS['gravity_norm_error']
    return bad | (valid & physical)


def audit_episode(ep,meta,dt=.02,history=25):
    reasons=[];n=len(ep['proprio'])
    shapes={'proprio':(n,129),'target':(n,4),'contact':(n,2),'valid':(n,),
            'normal_force':(n,2),'slip_speed':(n,2),'contact_count':(n,2),
            'contact_points':(n,2,3),'fk_audit':(n,2),'time':(n,)}
    for k,shape in shapes.items():
        if k not in ep or tuple(ep[k].shape)!=shape:reasons.append('shape:'+k)
    if reasons:return dict(accepted=False,reasons=reasons,frames=n,windows=0)
    if n!=meta['frames']:reasons.append('metadata_frame_count')
    if not torch.isfinite(ep['time']).all() or not torch.allclose(ep['time'][1:]-ep['time'][:-1],torch.full_like(ep['time'][1:],dt),atol=1e-5,rtol=1e-4):reasons.append('timestamp')
    valid=ep['valid'].bool()
    if bool(valid[0]) or not bool(valid[1:].all()):reasons.append('reset_or_imu_validity')
    bad=invalid_frames(ep)
    if bad.any():reasons.append('nonfinite_or_diverged_physics')
    force=ep['normal_force'];count=ep['contact_count'];has=count>0
    if not torch.isfinite(force).all() or (force<0).any() or (count<0).any():reasons.append('force_or_count')
    if not torch.isfinite(ep['slip_speed'][has]).all() or (ep['slip_speed'][has]<0).any():reasons.append('contact_slip')
    if not torch.isfinite(ep['contact_points'][has]).all():reasons.append('contact_point')
    expected=((force>30)&(ep['slip_speed']<.1)&has&torch.isfinite(ep['slip_speed'])).float()
    if not torch.equal(expected,ep['contact']):reasons.append('contact_label')
    support=(force>30).any(-1)
    if not torch.equal(ep['airborne'],~support):reasons.append('airborne_label')
    if not torch.equal(ep['slip'],((force>30)&(ep['slip_speed']>=.1)).any(-1)):reasons.append('slip_label')
    if not torch.equal(ep['landing'][1:],support[1:]&~support[:-1]):reasons.append('landing_label')
    windows=int(valid.unfold(0,history,1).all(-1).sum()) if n>=history else 0
    if not windows:reasons.append('no_valid_history')
    return dict(accepted=not reasons,reasons=reasons,frames=n,windows=windows,
        first_bad_frame=int(bad.nonzero()[0]) if bad.any() else None,
        fk_max=ep['fk_audit'].max(0).values.tolist(),
        target_abs_max=ep['target'].abs().amax(0).tolist(),
        proprio_abs_max=float(ep['proprio'].abs().amax()),
        events={k:int(ep[k][valid].sum()) for k in ('airborne','landing','slip','push')},
        terminated=bool(meta.get('terminated',False)))
