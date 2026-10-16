"""Contact plan includes an explicit known mask; unknown never becomes negative."""
import torch


def whole_body_contact_plan(state:torch.Tensor,confidence:torch.Tensor)->torch.Tensor:
    positive=(state==1).to(confidence.dtype)
    known=(state>=0).to(confidence.dtype)
    return torch.stack((positive,known,confidence*known),dim=-1).flatten(start_dim=1)
