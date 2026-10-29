"""Exercise the actual RLFT/AMP class entry points, not a stand-in guard mixin."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.peft.prior_agent import DiscretePriorPEFTRLFTAgent
from protomotions.agents.peft.prior_amp_agent import DiscretePriorPEFTRLFTAMPAgent
from protomotions.utils import guarded_rlft
from protomotions.utils.valid_samples import current_valid_mask


class _Fabric:
    def __init__(self, world_size=1, global_count=None, global_kl=None):
        self.world_size = world_size
        self.global_count = global_count
        self.global_kl = global_kl
        self.reductions = []

    def all_reduce(self, value, reduce_op="mean"):
        self.reductions.append((float(value.detach()), reduce_op))
        override = self.global_count if reduce_op == "sum" else self.global_kl
        return value if override is None else value.new_tensor(override)


def _agent(monkeypatch, *, guarded=True, amp=False, fabric=None, target_kl=None):
    cls = DiscretePriorPEFTRLFTAMPAgent if amp else DiscretePriorPEFTRLFTAgent
    agent = object.__new__(cls)
    agent.device = torch.device("cpu")
    agent.fabric = fabric or _Fabric()
    agent.config = SimpleNamespace(target_kl=target_kl)
    agent.env = SimpleNamespace(simulator=SimpleNamespace(_physics_guard=object() if guarded else None))
    agent.actor, agent.critic = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
    nn.init.ones_(agent.actor.weight)
    nn.init.ones_(agent.critic.weight)
    agent.actor_optimizer = torch.optim.SGD(agent.actor.parameters(), lr=0.1)
    agent.critic_optimizer = torch.optim.SGD(agent.critic.parameters(), lr=0.1)
    agent.actor_kl = 0.0
    agent.calls, agent.inputs, agent.scopes = [], {}, []

    def loss(model_name, batch):
        agent.calls.append(model_name)
        agent.inputs[model_name] = {key: value.clone() for key, value in batch.items()}
        agent.scopes.append((bool(getattr(agent, "_guard_optimizing", False)),
                             None if current_valid_mask() is None else current_valid_mask().clone()))
        assert torch.isfinite(batch["x"]).all(), "Invalid row reached trainable model"
        return getattr(agent, model_name)(batch["x"]).mean()

    agent.actor_step = lambda batch: (loss("actor", batch), {"actor/kl": torch.tensor(agent.actor_kl)})
    agent.critic_step = lambda batch: (loss("critic", batch), {})
    agent._clamp_peft_m = lambda: agent.calls.append("clamp")

    def optimizer_step(self, loss, model, optimizer, model_name):
        self.calls.append("update_" + model_name)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {}

    # Keep the real PhysicsGuardRLFTMixin._step_optimizer and its DDP loss scale.
    monkeypatch.setattr(BaseAgent, "_step_optimizer", optimizer_step)
    if amp:
        def amp_tail(batch, idx, logs):
            agent.calls.append("amp_tail")
            assert torch.equal(batch["x"], agent.inputs["actor"]["x"])
            assert getattr(agent, "_guard_optimizing", False)
            return {**logs, "amp/tail": torch.tensor(1.0)}
        agent.amp_component = SimpleNamespace(optimize_batch_tail=amp_tail)
    return agent


def _batch(mask, *, poison_invalid=True):
    valid = torch.tensor(mask, dtype=torch.bool)
    x = torch.arange(1, len(mask) + 1, dtype=torch.float32)[:, None]
    if poison_invalid:
        x[~valid] = float("nan")
    return {"x": x, "action": torch.zeros(len(mask), 1), "physics_valid": valid}


@pytest.mark.parametrize("amp", [False, True])
def test_real_rlft_dispatch_filters_invalid_rows_before_actor_critic(monkeypatch, amp):
    agent = _agent(monkeypatch, amp=amp)
    batch = _batch([True, False, True])
    observer_calls = []
    original = guarded_rlft.optimization_guarded

    def observer(current_agent, callback, current_batch, idx):
        observer_calls.append(current_agent)
        return original(current_agent, callback, current_batch, idx)

    monkeypatch.setattr(guarded_rlft, "optimization_guarded", observer)
    result = agent.perform_optimization_step(batch, 0)

    assert observer_calls and all(current is agent for current in observer_calls)
    for name in ("actor", "critic"):
        assert torch.equal(agent.inputs[name]["x"], torch.tensor([[1.0], [3.0]]))
        assert "physics_valid" not in agent.inputs[name]
        torch.testing.assert_close(getattr(agent, name).weight, torch.tensor([[0.8]]))
    assert all(enabled and mask.tolist() == [True, True] for enabled, mask in agent.scopes)
    assert agent.fabric.reductions == [(2.0, "sum")]
    assert result["physics_guard/valid_samples"].item() == 2
    assert agent.calls.count("update_actor") == agent.calls.count("update_critic") == 1
    assert agent.calls.count("amp_tail") == int(amp)
    assert not agent._guard_optimizing and current_valid_mask() is None


def test_real_rlft_dispatch_without_guard_preserves_unfiltered_update(monkeypatch):
    agent = _agent(monkeypatch, guarded=False)
    batch = _batch([True, False, True], poison_invalid=False)
    result = agent.perform_optimization_step(batch, 0)
    assert agent.inputs["actor"].keys() == batch.keys()
    assert torch.equal(agent.inputs["actor"]["x"], batch["x"])
    assert agent.scopes == [(False, None), (False, None)]
    assert agent.fabric.reductions == []
    assert "physics_guard/valid_samples" not in result
    assert agent.calls == ["actor", "update_actor", "clamp", "critic", "update_critic"]


def test_real_rlft_dispatch_global_empty_skips_all_updates_and_resets_kl(monkeypatch):
    agent = _agent(monkeypatch)
    agent._kl_early_stop_triggered = True
    before = [p.detach().clone() for model in (agent.actor, agent.critic) for p in model.parameters()]
    result = agent.perform_optimization_step(_batch([False, False]), 0)
    assert agent.calls == []
    assert not agent._kl_early_stop_triggered
    assert result["physics_guard/skipped_empty_batch"].item() == 1
    assert agent.fabric.reductions == [(0.0, "sum")]
    after = [p.detach() for model in (agent.actor, agent.critic) for p in model.parameters()]
    assert all(torch.equal(old, new) for old, new in zip(before, after))


def test_real_rlft_dispatch_empty_local_rank_keeps_dummy_and_collectives(monkeypatch):
    fabric = _Fabric(world_size=2, global_count=3, global_kl=0.0)
    agent = _agent(monkeypatch, fabric=fabric, target_kl=0.1)
    agent.actor_kl = 0.2
    result = agent.perform_optimization_step(_batch([False, False]), 0)
    assert fabric.reductions == [(0.0, "sum"), (0.0, "mean")]
    assert agent.calls == ["actor", "update_actor", "clamp", "critic", "update_critic"]
    assert all(enabled and mask.tolist() == [False] for enabled, mask in agent.scopes)
    for name in ("actor", "critic"):
        assert torch.equal(agent.inputs[name]["x"], torch.zeros(1, 1))
        torch.testing.assert_close(getattr(agent, name).weight, torch.ones(1, 1))
    assert agent._guard_loss_scale.item() == 0
    assert result["physics_guard/valid_samples"].item() == 0
    assert result["physics_guard/skipped_empty_batch"].item() == 0
    assert current_valid_mask() is None and not agent._guard_optimizing


@pytest.mark.parametrize("guarded", [False, True])
def test_real_rlft_dispatch_preserves_target_kl_and_critic_updates(monkeypatch, guarded):
    agent = _agent(monkeypatch, guarded=guarded, target_kl=0.1)
    agent.actor_kl = 0.2
    result = agent.perform_optimization_step(_batch([True, True]), 0)
    assert "update_actor" not in agent.calls and "update_critic" in agent.calls
    assert result["actor/update_skipped"].item() == 1
    assert result["ppo/kl_early_stopped"].item() == 1
    assert agent.fabric.reductions == ([(2.0, "sum")] if guarded else []) + [
        (float(torch.tensor(0.2)), "mean")
    ]


def test_real_rlft_dispatch_preserves_global_valid_count_loss_and_kl_scaling(monkeypatch):
    fabric = _Fabric(world_size=2, global_count=4)
    agent = _agent(monkeypatch, fabric=fabric, target_kl=0.1)
    agent.actor_kl = 0.2
    result = agent.perform_optimization_step(_batch([True, False]), 0)
    assert fabric.reductions == [(1.0, "sum"), (float(torch.tensor(0.1)), "mean")]
    assert agent._guard_loss_scale.item() == 0.5
    for name in ("actor", "critic"):
        torch.testing.assert_close(getattr(agent, name).weight, torch.tensor([[0.95]]))
    assert result["actor/update_skipped"].item() == 0
