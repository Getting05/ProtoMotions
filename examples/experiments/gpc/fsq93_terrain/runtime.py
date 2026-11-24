"""Native P2 states, MuJoCo contacts, and SuperMimic history for terrain RLFT."""
from pathlib import Path

import numpy as np
import torch

from p2drift.models.supermimic import SuperMimicTracker
from p2drift.models.tracker_backend import policy_frame
from p2drift.observations.history import History
from p2drift.runtime.simulator import create_simulator

from .scene import TerrainTask, generate, scene_asset, yaw_wxyz


def reset_cases(path):
    cases = torch.load(path, map_location="cpu", weights_only=True)
    states, contexts, history = (cases[key].float() for key in ("states", "xi", "history"))
    upright = 1 - 2 * states[:, 4:6].square().sum(-1)
    finite = states.isfinite().all(-1) & contexts.isfinite().all(-1) & history.isfinite().all((1, 2))
    stand = finite & (upright > 0.94) & (states[:, 2] > 0.46)
    valid = stand.nonzero(as_tuple=True)[0]
    if len(valid) == 0:
        raise ValueError("No upright authentic SuperMimic reset cases")
    # Low translational speed, angular speed and joint speed.
    quiet = states[valid, 7:9].norm(dim=-1)
    quiet += 0.15 * states[valid, 10:13].norm(dim=-1)
    quiet += 0.04 * states[valid, 42:].norm(dim=-1)
    keep = valid[quiet.argsort()[:min(16, len(valid))]]
    return states[keep], contexts[keep], history[keep]


def face_goal(state, spawn, target):
    """Apply world yaw/translation while both native history and local pose stay valid."""
    d = target[:, :2] - spawn[:, :2]
    angle = torch.atan2(d[:, 1], d[:, 0]) - yaw_wxyz(state[:, 3:7])
    c, s = angle.cos(), angle.sin()
    ch, sh = (angle / 2).cos(), (angle / 2).sin()
    w, x, y, z = state[:, 3:7].unbind(-1)
    state[:, 3:7] = torch.stack(
        (ch * w - sh * z, ch * x - sh * y, ch * y + sh * x, ch * z + sh * w), dim=-1
    )
    for a in (7, 10):
        vx, vy = state[:, a:a + 2].unbind(-1)
        state[:, a:a + 2] = torch.stack((c * vx - s * vy, s * vx + c * vy), dim=-1)
    state[:, :2] = spawn[:, :2]
    state[:, 2] += spawn[:, 2]
    return state


class TerrainEnv:
    def __init__(self, config, output, device, num_envs, tracker=None):
        self.device, self.num_envs = device, num_envs
        self.opts = config["stage3"]
        scene = generate(config["terrain"])
        self.task = TerrainTask(scene, config["task"], device)
        assets = config["assets"]
        robot = scene_asset(scene, assets["robot_xml"], Path(output) / "terrain.xml")
        self.pool = tuple(x.to(device) for x in reset_cases(assets["reset_cases"]))
        if tracker is None:
            tracker = SuperMimicTracker.from_checkpoint(assets["tracker"], "cpu", "require_native93")
        self.manifest = tracker.manifest
        self.sim = create_simulator(
            dict(config["simulation"], asset=str(robot), num_envs_per_rank=num_envs),
            self.manifest, device,
        )
        self.default_q = torch.as_tensor(
            self.manifest["action_config"]["pd_action_offset"][:29], device=device
        )
        self.history = History(10)
        self.elapsed = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.state = None
        self.target = torch.zeros(num_envs, 3, device=device)
        try:
            self.reset()
        except Exception:
            self.close()
            raise

    def reset(self, ids=None):
        first = ids is None
        if first:
            ids = torch.arange(self.num_envs, device=self.device)
        selection = torch.randint(len(self.pool[0]), (len(ids),), device=self.device)
        state, context, history = (values[selection] for values in self.pool)
        spawn, goal = self.task.sample(len(ids))
        self.target[ids] = goal
        state = face_goal(state, spawn, goal)
        self.sim.reset(state, context, None if first else ids)
        self.state = self.sim.backend.state().clone()
        if first:
            self.history.load_real_history(history)
        else:
            full_history = self.history.value().clone()
            full_history[ids] = history
            self.history.load_real_history(full_history)
        self.elapsed[ids] = 0

    def observation(self):
        grid, target = self.task.observations(self.state, self.target)
        return self.history.value().flatten(1), grid, target

    def step(self, action):
        result = self.sim.step(action)
        state = result.motor_full_state71
        invalid = result.numerical_invalid | ~state.isfinite().all(-1)
        previous_state = self.state
        self.state = torch.where(invalid[:, None], previous_state, state)
        self.history.append(policy_frame(
            self.manifest, self.state, torch.nan_to_num(result.sensor71), action,
            self.default_q, result.physics_context33
        ))
        self.elapsed += 1
        reward, reached, failed = self.task.outcome(self.state, self.target, previous_state)
        reached = reached & ~result.done & ~invalid
        terminal = result.done | invalid | failed | reached
        timeout = (self.elapsed >= self.opts["episode_steps"]) & ~terminal
        reward = torch.where(invalid, 0., reward)
        return reward, terminal, timeout, reached, failed, invalid

    def close(self):
        self.sim.close()


def audit_terrain(env):
    """Cross-check the task map and termination against compiled simulator geometry."""
    import mujoco

    model = env.sim.backend.cpu_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    task, scene = env.task, env.task.scene
    mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

    def ray(x, y, z, vertical):
        geom = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            model, data, np.array([x, y, z]), np.array([0., 0., vertical]),
            mask, 1, -1, geom
        )
        if distance < 0:
            raise RuntimeError(f"Missing terrain ray hit at {(x, y, z)}")
        return z + vertical * float(distance), model.geom(int(geom[0]))

    def map_at(x, y, z):
        state = env.state[:1].clone()
        state[0, :7] = torch.tensor([x, y, z, 1., 0., 0., 0.], device=env.device)
        grid, _ = task.observations(state, env.target[:1])
        center = task.config["map_samples"] // 2
        return grid[0, :, center, center].cpu().numpy()

    if scene.beams:
        for x, y, _, width, clearance, _ in scene.beams:
            height, geom = ray(x, y, .35, 1.)
            assert geom.name.startswith("hsi_beam_") and int(geom.contype[0]) == 1
            assert width > scene.lane_width
            np.testing.assert_allclose(height, clearance, atol=1e-6)
            np.testing.assert_allclose(map_at(x, y, .35), [-.35, height - .35, 1.], atol=1e-6)
        x, y, _, _, clearance, _ = scene.beams[0]
        before, after = env.state[:1].clone(), env.state[:1].clone()
        for below in (True, False):
            z = clearance + (-0.05 if below else 0.25)
            before[0, :3] = torch.tensor([x - 0.01, y, z], device=env.device)
            after[0, :3] = torch.tensor([x + 0.01, y, z], device=env.device)
            assert task.outcome(after, env.target[:1], before)[2].item() == (not below)
        state = env.state[:1].clone()
        state[0, 1] = scene.lane_width / 2 + 0.05
    else:
        for x, y, _, _, _, top in scene.floors:
            height, geom = ray(x, y, 4., -1.)
            assert geom.name.startswith("hsi_floor_") and int(geom.contype[0]) == 1
            np.testing.assert_allclose(height, top, atol=1e-6)
            np.testing.assert_allclose(map_at(x, y, top + .65), [-.65, 0., 0.], atol=1e-6)
        left, right = scene.floors[:2]
        gap_y = (left[1] + left[3] / 2 + right[1] - right[3] / 2) / 2
        height, geom = ray(left[0], gap_y, 4., -1.)
        assert geom.name == "p2_v2_ground"
        np.testing.assert_allclose(map_at(left[0], gap_y, 1.), [height - 1., 0., 0.], atol=1e-6)
        state = env.state[:1].clone()
        state[0, :3] = torch.tensor([left[0], gap_y, .65], device=env.device)
    _, _, failed = task.outcome(state, env.target[:1])
    assert failed.item()
    state[0, :3] = env.target[0]
    state[0, 2] += .65
    reward, reached, failed = task.outcome(state, env.target[:1])
    assert reward.item() == 1. and reached.item() and not failed.item()
    return dict(collider_count=len(scene.beams) or len(scene.floors),
                geometry_and_observation_check="passed")
