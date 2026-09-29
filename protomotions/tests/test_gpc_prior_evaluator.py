"""Prior metrics, rollout semantics, state isolation, and old-config migration."""
import argparse
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from protomotions.agents.evaluators.gpc_prior_evaluator import (
    GPCPriorEvaluator, SurvivalAccumulator, select_motion_ids, token_metrics, upright_failure,
)
from protomotions.agents.evaluators.gpc_prior_config import GPCPriorEvaluatorConfig
from protomotions.agents.evaluators.gpc_prior_cli import (
    add_prior_eval_arguments, apply_prior_eval_options, explicit_prior_eval_options,
)
from protomotions.agents.evaluators.config import EvaluatorConfig


def test_nll_is_unsmoothed_and_tokens_not_sequences():
    logits = torch.tensor([[[4., 0.], [0., 4.]], [[4., 0.], [4., 0.]]])
    targets = torch.tensor([[0, 1], [0, 1]])
    metrics = token_metrics(logits, targets)
    assert metrics['token_accuracy'].tolist() == [1., .5]
    assert metrics['sequence_accuracy'].tolist() == [1., 0.]
    assert metrics['nll'][0] == pytest.approx(np.log1p(np.exp(-4)), abs=1e-7)
    assert metrics['nll'][1] == pytest.approx(2 + np.log1p(np.exp(-4)))


def test_validation_partition_is_disjoint_and_repeatable():
    ids = [select_motion_ids(101, 12, 3, rank, 8) for rank in range(8)]
    assert len(set(torch.cat(ids).tolist())) == 96
    assert torch.equal(ids[3], select_motion_ids(101, 12, 3, 3, 8))
    with pytest.raises(ValueError):
        select_motion_ids(2, 3, 3, 3, 8)


def test_xyzw_and_persistent_failure_never_recovers():
    q = torch.tensor([[0., 0., 0., 1.], [1., 0., 0., 0.]])
    assert upright_failure(torch.ones(2), q, .3, .25).tolist() == [False, True]
    state = SurvivalAccumulator(torch.tensor([False, True]), .02, 100, 2)
    state.update(torch.tensor([True, False]), 1)
    assert state.alive.tolist() == [True, False]
    state.update(torch.tensor([True, False]), 2)
    state.update(torch.tensor([False, False]), 3)
    assert state.alive.tolist() == [False, False]
    assert state.seconds.tolist() == pytest.approx([.04, 0])


def test_explicit_options_upgrade_old_frozen_config():
    parser = argparse.ArgumentParser()
    add_prior_eval_arguments(parser)
    assert explicit_prior_eval_options(parser.parse_args([])) is None
    options = explicit_prior_eval_options(parser.parse_args([
        '--gpc-prior-eval', '--prior-validation-motion-file', '/validation.pt',
        '--prior-eval-num-envs', '8']))
    agent = SimpleNamespace(model=SimpleNamespace(
        _target_='x.DiscreteAutoregressiveLatentPriorModel'), evaluator=EvaluatorConfig())
    apply_prior_eval_options(agent, options)
    assert isinstance(agent.evaluator, GPCPriorEvaluatorConfig)
    assert agent.evaluator.num_eval_envs == 8
    assert agent.evaluator.validation_motion_file == '/validation.pt'
    with pytest.raises(ValueError):
        apply_prior_eval_options(agent, {'num_eval_envs': 0})


def test_evaluation_scope_restores_on_failure():
    model = torch.nn.Linear(1, 1)
    manager, lib = object(), object()
    env = SimpleNamespace(motion_lib=lib, motion_manager=manager,
        robot_config=SimpleNamespace(reset_noise='noise'),
        simulator=SimpleNamespace(_push_enabled=True),
        save_state=lambda: 'snapshot', restore_state=lambda state: restored.append(state))
    restored = []
    agent = SimpleNamespace(env=env, motion_lib=lib, model=model, eval=model.eval)
    evaluator = GPCPriorEvaluator(agent, SimpleNamespace(device=torch.device('cpu'), global_rank=0),
                                  GPCPriorEvaluatorConfig())
    torch_state, py_state, np_state = torch.get_rng_state(), random.getstate(), np.random.get_state()
    with pytest.raises(RuntimeError):
        with evaluator._evaluation_scope():
            assert not model.training
            env.motion_manager, env.motion_lib, agent.motion_lib = None, None, None
            torch.rand(3); random.random(); np.random.rand()
            raise RuntimeError('simulation failure')
    assert env.motion_manager is manager and env.motion_lib is lib and agent.motion_lib is lib
    assert env.robot_config.reset_noise == 'noise' and env.simulator._push_enabled
    assert model.training and restored == ['snapshot']
    assert torch.equal(torch_state, torch.get_rng_state())
    assert py_state == random.getstate()
    assert np.array_equal(np_state[1], np.random.get_state()[1])


@pytest.mark.parametrize("sensors", [True, False])
def test_autonomous_ignores_reference_done_and_uses_generate(sensors):
    class Model:
        calls = 0
        def generate(self, td):
            self.calls += 1
            return torch.ones(2, 3), {}
    model = Model()
    env = SimpleNamespace(dt=.1, contact_body_ids=torch.tensor([0] if sensors else [], dtype=torch.long))
    agent = SimpleNamespace(env=env, model=model, num_envs=2)
    evaluator = GPCPriorEvaluator(agent, SimpleNamespace(device=torch.device('cpu')),
                                  GPCPriorEvaluatorConfig(max_eval_steps=5, failure_persistence_steps=2))
    evaluator._reset_cohort = lambda ids: 'obs'
    evaluator._obs = lambda obs, n: TensorDict({'x': torch.ones(n, 1)}, batch_size=[n])
    # Reference clip reports done every step. Autonomous evaluation must continue.
    evaluator._step = lambda action, n: ('obs', None, torch.ones(n, dtype=torch.bool), None, None)
    state = SimpleNamespace(rigid_body_contacts=torch.ones(2, 1, dtype=torch.bool),
                            rigid_body_vel=torch.zeros(2, 1, 3))
    evaluator._physical_state = lambda n: (state, torch.tensor([model.calls >= 2, False]))
    metrics, details = evaluator._autonomous(torch.tensor([0, 1]))
    assert model.calls == 5
    assert metrics['upright_survival_horizon'] == .5
    assert details['capped_survival_seconds'] == pytest.approx([.3, .5])
    assert 'upright_survival_5s' not in metrics

    assert metrics["contact_sensing_available"] == float(sensors)
    assert ("mean_contact_body_xy_speed_mps" in metrics) == sensors
