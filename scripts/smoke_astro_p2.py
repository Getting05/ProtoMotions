"""Load/reset/step P2 through ProtoMotions (no motion data or policy required)."""
import argparse
p = argparse.ArgumentParser()
p.add_argument('--simulator', default='mujoco')
p.add_argument('--steps', type=int, default=200)
p.add_argument('--num-envs', type=int, default=1)
p.add_argument('--cpu-only', action='store_true')
p.add_argument('--headless', action='store_true')
p.add_argument('--contacts', action='store_true')
a = p.parse_args()
if a.simulator == 'isaacgym' and a.cpu_only:
    p.error('This checkout of IsaacGym forces GPU PhysX; omit --cpu-only.')
from protomotions.utils.simulator_imports import import_simulator_before_torch
AppLauncher = import_simulator_before_torch(a.simulator)
import torch
from protomotions.robot_configs.factory import robot_config
from protomotions.simulator.factory import simulator_config
from protomotions.utils.hydra_replacement import get_class
from protomotions.components.terrains.config import TerrainConfig
from protomotions.components.terrains.terrain import Terrain
from protomotions.components.scene_lib import SceneLib
from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
r = robot_config('astro_p2', **({'contact_bodies': 'all'} if a.contacts else {}))
assert r.number_of_actions == 30
# Check the compiled model against the supplied URDF, independently of config data.
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
from protomotions.assets import resolve_asset_root
asset = Path(resolve_asset_root(r.asset.asset_root)) / r.asset.asset_file_name
model = mujoco.MjModel.from_xml_path(str(asset))
assert model.nu == 30 and model.nq == 37 and model.nv == 36
limits = {j.get('name'): j.find('limit').attrib for j in
          ET.parse(asset.parent.parent / 'urdf/astro_p2_30dof_primitive_collision.urdf').findall('joint')
          if j.get('type') != 'fixed'}
for name, q in zip(r.kinematic_info.dof_names, r.default_dof_pos):
    bound = limits[name]
    assert float(bound['lower']) <= q <= float(bound['upper']), name
    import numpy as np
    np.testing.assert_allclose(model.joint(name).range, [float(bound['lower']), float(bound['upper'])])
    ci = r.control.control_info[name]
    assert ci.effort_limit == float(bound['effort']), name
    assert ci.velocity_limit == float(bound['velocity']), name
assert set(r.control.control_info) == set(r.kinematic_info.dof_names)
for names in r.common_naming_to_robot_body_names.values():
    assert set(names) <= set(r.kinematic_info.body_names)
for name, val in r.control.control_info.items():
    assert all(getattr(val, k) > 0 for k in ('stiffness','damping','effort_limit','velocity_limit','armature')), name
device = torch.device('cpu' if a.cpu_only or a.simulator == 'mujoco' else 'cuda:0')
extra = {}
if a.simulator == 'isaaclab':
    launcher = AppLauncher({'headless': a.headless, 'device': str(device)})
    extra['simulation_app'] = launcher.app
c = simulator_config(a.simulator, r, headless=a.headless, num_envs=a.num_envs, experiment_name='astro_p2_smoke')
# Small smoke test does not need training-sized (16M-pair) GPU buffers.
if a.simulator == 'isaacgym':
    r.contact_pairs_multiplier = 1
    c.sim.physx.default_buffer_size_multiplier = 1.0
tc, c = convert_friction_for_simulator(TerrainConfig(), c)
t = Terrain(config=tc, num_envs=a.num_envs, device=device)
sim = get_class(c._target_)(config=c, robot_config=r, terrain=t, scene_lib=SceneLib.empty(num_envs=a.num_envs, device=device), device=device, **extra)
try:
    sim._initialize_with_markers({})
    state = sim.get_default_robot_reset_state()
    xy = t.sample_valid_locations(a.num_envs)
    state.root_pos[:, :2] = xy
    state.root_pos[:, 2] = t.get_ground_heights(xy).view(-1) + r.default_root_height
    state.dof_pos[:] = r.default_dof_pos.to(device)
    sim.reset_envs(state, env_ids=torch.arange(a.num_envs, device=device))
    target = r.default_dof_pos.to(device).expand(a.num_envs, -1).clone()
    for i in range(a.steps):
        sim.step(target)
        root = sim.get_root_state().root_pos
        assert torch.isfinite(root).all(), f'Nonfinite root at {i}'
        dofs = sim.get_dof_state().dof_pos
        assert torch.isfinite(dofs).all(), f'Nonfinite joints at {i}'
    print(f'P2_SMOKE_PASS (integration, not balance) simulator={a.simulator} envs={a.num_envs} steps={a.steps} dofs={r.number_of_actions} bodies={r.kinematic_info.num_bodies} final_root_z={root[:,2].tolist()}', flush=True)
finally:
    sim.close()
