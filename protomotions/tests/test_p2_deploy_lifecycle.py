"""P2 IMU continuity across evaluation's temporary simulator maintenance."""

from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.base_env.p2_deploy import P2DeployEnv
from protomotions.envs.obs.p2_deploy_sensors import P2DeploySensorBank


def _isaaclab_method(name):
    # Run the actual backend method with a fake scene. Importing the backend
    # would require IsaacLab and silently skip this regression on CPU CI.
    source = Path(__file__).parents[1] / "simulator/isaaclab/simulator.py"
    tree = ast.parse(source.read_text())
    backend = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "IsaacLabSimulator")
    method = next(node for node in backend.body
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


def _state(velocity=0.0):
    return SimpleNamespace(
        dof_pos=torch.arange(60.0).reshape(2, 30),
        dof_vel=torch.full((2, 30), velocity),
        root_rot=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1),
        root_vel=torch.full((2, 3), velocity),
        root_ang_vel=torch.full((2, 3), velocity / 10.0),
    )


def _env():
    env = P2DeployEnv.__new__(P2DeployEnv)
    env.num_envs, env.device = 2, "cpu"
    env.config = SimpleNamespace(num_state_history_steps=3)
    env._deploy_resetting = False
    env._deploy_bank = P2DeploySensorBank(2, 3, "cpu")
    env._deploy_bank.seed(_state(), torch.arange(2))
    env._deploy_bank.sample(_state(0.1))
    env._deploy_bank.commit()
    events = []
    sim = SimpleNamespace(
        _move_all_envs_out_of_contact=lambda **kwargs: events.append("hide"),
        _scene=SimpleNamespace(
            write_data_to_sim=lambda: events.append("write"),
            update=lambda dt: events.append("scene"),
        ),
        _sim=SimpleNamespace(
            step=lambda render: events.append("physics"),
            get_physics_dt=lambda: 0.005,
        ),
        # Maintenance must never read this artificial underground state.
        get_robot_state=lambda: _state(1000.0),
    )
    sim.flush_contact_cache = MethodType(_isaaclab_method("flush_contact_cache"), sim)
    env.simulator = sim
    env._install_deploy_sensor_hook()
    return env, events


def _assert_bank_state(bank, expected):
    actual = bank.state_dict()
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
        else:
            assert actual[name] == value


@pytest.mark.parametrize("with_guard", [False, True])
def test_contact_flush_does_not_sample_or_advance_p2_sensors(with_guard):
    env, events = _env()
    sim = env.simulator
    if with_guard:
        guard = SimpleNamespace(
            invalid=torch.zeros(2, dtype=torch.bool),
            inspect_and_recover=lambda state, substep: events.append("recover"),
        )
        sim._physics_guard = guard
        # Register the wrapper in the same order as a normal guard step.
        sim._scene.update(0.005)
        guard.inspect_and_recover(None, 0)
        events.clear()
    before = env._deploy_bank.state_dict()

    sim.flush_contact_cache()

    assert events == ["hide", "write", "physics", "scene"]
    _assert_bank_state(env._deploy_bank, before)
    assert sim._flushing_contact_cache is False
    # A subsequent real substep still samples, after guard recovery if present.
    sim._scene.update(0.005)
    if with_guard:
        assert env._deploy_bank.sample_count == before["sample_count"]
        guard.inspect_and_recover(None, 0)
        assert events[-1] == "recover"
    assert env._deploy_bank.sample_count == before["sample_count"] + 1


@pytest.mark.parametrize("was_flushing", [False, True])
def test_contact_flush_restores_marker_when_backend_fails(was_flushing):
    env, _ = _env()
    sim = env.simulator
    sim._flushing_contact_cache = was_flushing

    def fail(render):
        assert sim._flushing_contact_cache is True
        raise RuntimeError("physics failed")

    sim._sim.step = fail
    with pytest.raises(RuntimeError, match="physics failed"):
        sim.flush_contact_cache()
    assert sim._flushing_contact_cache is was_flushing


def test_trace_substeps_sample_normally_and_flush_bypasses_trace():
    from protomotions.utils.physics_trace import PhysicsTrace

    env, events = _env()
    sim = env.simulator
    sim.decimation, sim.headless, sim.num_envs = 4, True, 2
    sim._apply_control = lambda: events.append("control")
    sample = env._sample_deploy_sensors

    def sample_and_record(invalid=None):
        events.append("sample")
        sample(invalid)

    env._sample_deploy_sensors = sample_and_record
    trace = PhysicsTrace.__new__(PhysicsTrace)
    trace.sim, trace.step_id, trace.max_velocity = sim, 1, 0.0
    trace.context = lambda: {}
    trace.raw = lambda: events.append("trace_read") or {"joint_vel": torch.zeros(2, 30)}
    sim._physics_trace = trace
    before = env._deploy_bank.sample_count

    _isaaclab_method("_physics_step")(sim)

    assert events == ["trace_read"] + [
        "control", "write", "physics", "scene", "sample", "trace_read"
    ] * 4
    assert env._deploy_bank.sample_count == before + 4
    events.clear()
    sim.flush_contact_cache()
    assert events == ["hide", "write", "physics", "scene"]
    assert trace.step_id == 2
    assert env._deploy_bank.sample_count == before + 4


def test_evaluation_restores_p2_history_and_next_imu_sample_exactly():
    env, events = _env()
    with patch.object(BaseEnv, "save_state", return_value={"physical": "saved"}):
        snapshot = env.save_state()
    expected = snapshot["p2_deploy_sensors"]
    uninterrupted = P2DeploySensorBank(2, 3, "cpu")
    uninterrupted.load_state_dict(expected)

    env._deploy_bank.sample(_state(7.0))
    env._deploy_bank.commit()
    env._deploy_bank.valid[1] = False
    # The evaluation rollout did not mutate the snapshot's cloned buffers.
    _assert_bank_state(uninterrupted, expected)

    def restore_physics(current_env, saved):
        assert saved is snapshot
        assert current_env._deploy_resetting is True
        current_env.simulator.flush_contact_cache()
        events.append("restore_pose")
        current_env.simulator._scene.update(0.005)

    with patch.object(BaseEnv, "restore_state", restore_physics):
        env.restore_state(snapshot)

    assert events == ["hide", "write", "physics", "scene", "restore_pose", "scene"]
    assert env._deploy_resetting is False
    _assert_bank_state(env._deploy_bank, expected)
    next_state = _state(0.13)
    env._deploy_bank.sample(next_state)
    uninterrupted.sample(next_state)
    env._deploy_bank.commit()
    uninterrupted.commit()
    _assert_bank_state(env._deploy_bank, uninterrupted.state_dict())


@pytest.mark.parametrize("was_resetting", [False, True])
def test_evaluation_restore_preserves_sampling_guard_on_failure(was_resetting):
    env, _ = _env()
    env._deploy_resetting = was_resetting
    snapshot = {"p2_deploy_sensors": env._deploy_bank.state_dict()}
    with patch.object(BaseEnv, "restore_state", side_effect=RuntimeError("restore failed")):
        with pytest.raises(RuntimeError, match="restore failed"):
            env.restore_state(snapshot)
    assert env._deploy_resetting is was_resetting


def test_evaluation_restore_refuses_snapshot_without_sensor_history():
    env, _ = _env()
    with patch.object(BaseEnv, "restore_state") as restore:
        with pytest.raises(ValueError, match="P2 deployment sensor state"):
            env.restore_state({})
    restore.assert_not_called()


@pytest.mark.parametrize("was_resetting", [False, True])
def test_reset_suppresses_sampling_and_preserves_outer_guard(was_resetting):
    env, _ = _env()
    env._deploy_resetting = was_resetting
    before = env._deploy_bank.state_dict()

    def reset_physics(current_env):
        assert current_env._deploy_resetting is True
        current_env.simulator._scene.update(0.005)
        return "reset"

    with patch.object(BaseEnv, "reset", reset_physics):
        assert env.reset() == "reset"
    assert env._deploy_resetting is was_resetting
    _assert_bank_state(env._deploy_bank, before)
