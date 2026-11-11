#!/usr/bin/env python3
"""Verify trusted SuperMimic FSQ93 checkpoint loading and tensor interfaces."""
import argparse
import torch
from tensordict import TensorDict
from protomotions.agents.common.latent import LATENT_KEY
from protomotions.agents.common.supermimic_fsq import (
    GOAL_KEY, OBS_KEY, SuperMimicFSQTracker, SuperMimicFSQTrackerConfig,
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    tracker = SuperMimicFSQTracker(SuperMimicFSQTrackerConfig(
        checkpoint_path=args.checkpoint, trust_checkpoint=True
    )).to(args.device).eval()
    assert not any(p.requires_grad for p in tracker.parameters())
    with torch.inference_mode():
        obs = torch.randn(2, 930, device=args.device)
        goal = torch.randn(2, 730, device=args.device)
        codes = tracker.encoder(TensorDict({GOAL_KEY: goal}, batch_size=[2]))[LATENT_KEY]
        assert codes.shape == (2, 40) and bool((codes == codes.round()).all())
        indices = tracker.quantizer.codes_to_indices(codes)
        torch.testing.assert_close(tracker.quantizer.indices_to_codes(indices), codes)
        action = tracker.decoder(TensorDict({OBS_KEY: obs, LATENT_KEY: codes}, batch_size=[2]))["action"]
        assert action.shape == (2, 29) and bool(torch.isfinite(action).all())
    print("PASS: frozen SuperMimic FSQ 40x9, 930D history, 730D goal, 29D action")

if __name__ == "__main__":
    main()