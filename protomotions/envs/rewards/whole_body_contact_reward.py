"""Unknown-masked full-body supervision, with expected-patch association."""
import torch


def whole_body_contact_reward(state:torch.Tensor,confidence:torch.Tensor,
                              simulated_contacts:torch.Tensor,expected_patch_match:torch.Tensor)->torch.Tensor:
    weight=confidence*(state==1).to(confidence.dtype)
    matched=(simulated_contacts>0).to(confidence.dtype)*expected_patch_match.to(confidence.dtype)
    return (weight*matched).sum(-1)/weight.sum(-1).clamp_min(1e-6)


def unexpected_contact_penalty(state:torch.Tensor,confidence:torch.Tensor,
                               force_magnitudes:torch.Tensor,force_threshold:float=25.)->torch.Tensor:
    # Severe contacts only, with explicit negative supervision.
    weight=confidence*(state==0).to(confidence.dtype)
    return (weight*(force_magnitudes>force_threshold)).sum(-1)/weight.sum(-1).clamp_min(1e-6)
