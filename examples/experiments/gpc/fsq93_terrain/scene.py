"""One scene description drives collider construction, goals, and heightmaps."""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class Scene:
    kind: str
    # Rows are center x, center y, full x width, full y width, underside, top.
    floors: tuple
    beams: tuple
    starts: np.ndarray
    goals: np.ndarray
    lane_width: float | None


def load_config(path):
    path = Path(path)
    common = yaml.safe_load((path.parent / "common.yaml").read_text())
    terrain = yaml.safe_load(path.read_text())
    parent = terrain.pop("terrain_base", None)
    if parent is not None:
        terrain = yaml.safe_load(path.with_name(parent).read_text()) | terrain
    common["stage3"].update(terrain.pop("stage3", {}))
    common["terrain"] = terrain
    return common


def generate(config):
    kind = config["layout"]
    rng = np.random.default_rng(config["seed"])
    if kind == "barrier":
        length, width = config["corridor_length_m"], config["corridor_width_m"]
        lane = config["lane_width_m"]
        depth, thick = config["beam_depth_m"], config["beam_thickness_m"]
        xs = np.array(config["barrier_x_m"])
        heights = rng.uniform(*config["clearance_m"], len(xs))
        floors = ((length / 2, 0., length, width, -0.08, 0.),)
        beams = tuple((x, 0., depth, width, h, h + thick) for x, h in zip(xs, heights))
        if config["target_mode"] == "full_corridor":
            start_x, goal_x = np.array([0.80]), np.array([length - 0.80])
        elif config["target_mode"] == "one_barrier":
            start_x, goal_x = xs - 1.15, xs + 0.80
        else:
            raise ValueError("Unknown Barrier target mode")
        starts = np.column_stack((start_x, np.zeros((len(start_x), 2))))
        goals = np.column_stack((goal_x, np.zeros((len(goal_x), 2))))[:, None, :]
        return Scene(kind, floors, beams, starts, goals, lane)
    if kind not in ("platform_2x2", "platform_giant"):
        raise ValueError(f"Unknown terrain {kind!r}")
    nx, ny = config["cols"], config["rows"]

    def axis(n):
        widths = rng.uniform(*config["size_m"], n)
        gaps = rng.uniform(*config["gap_m"], n - 1)
        start = np.r_[0., np.cumsum(widths[:-1] + gaps)]
        return start + widths / 2 - (start[-1] + widths[-1]) / 2, widths

    x, wx = axis(nx)
    y, wy = axis(ny)
    base, delta = config["base_height_m"], config["height_delta_m"]
    tops = base + rng.uniform(0., delta, (nx, ny))
    floors, starts, goals = [], [], []
    for i in range(nx):
        for j in range(ny):
            floors.append((x[i], y[j], wx[i], wy[j], 0., tops[i, j]))
            starts.append((x[i], y[j], tops[i, j]))
            neighbors = [
                (x[a], y[b], tops[a, b])
                for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                if 0 <= a < nx and 0 <= b < ny
            ]
            # Pad with NaNs so the valid neighbors can be sampled in a batch.
            goals.append(neighbors + [(np.nan,) * 3] * (4 - len(neighbors)))
    return Scene(kind, tuple(floors), (), np.array(starts), np.array(goals), None)


def scene_asset(scene, robot_xml, output):
    # Absolute resource paths preserve the robot asset when the MJCF is generated
    # in a run directory.
    robot_xml, output = Path(robot_xml).resolve(), Path(output)
    tree = ET.parse(robot_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        for key in ("meshdir", "texturedir", "assetdir"):
            if compiler.get(key):
                compiler.set(key, str((robot_xml.parent / compiler.get(key)).resolve()))
    world = root.find("worldbody")
    if scene.kind == "barrier":
        # The walkway top coincides with the runtime's flat plane. Render it as
        # a thin overlay with collision disabled to avoid duplicate contacts.
        floor_rgba = ".30 .40 .44 1"
    else:
        floor_rgba = ".38 .61 .68 1"
    for prefix, boxes, color in (("floor", scene.floors, floor_rgba),
                                 ("beam", scene.beams, ".92 .46 .16 1")):
        for i, (x, y, wx, wy, bottom, top) in enumerate(boxes):
            visual_walkway = scene.kind == "barrier" and prefix == "floor"
            if visual_walkway:
                top = 0.002
            ET.SubElement(
                world, "geom", name=f"hsi_{prefix}_{i:02d}", type="box",
                pos=f"{x} {y} {(bottom + top) / 2}",
                size=f"{wx / 2} {wy / 2} {(top - bottom) / 2}", rgba=color,
                friction="1 .005 .0001", condim="3", group="0",
                contype="0" if visual_walkway else "1",
                conaffinity="0" if visual_walkway else "1",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output.resolve()


def yaw_wxyz(quat):
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


class TerrainTask:
    def __init__(self, scene, config, device):
        self.scene, self.config, self.device = scene, config, device
        self.starts = torch.as_tensor(scene.starts, dtype=torch.float32, device=device)
        self.goals = torch.as_tensor(scene.goals, dtype=torch.float32, device=device)
        self.goal_counts = self.goals[..., 0].isfinite().sum(-1)
        self.platforms = torch.as_tensor(scene.floors if scene.lane_width is None else (),
                                         dtype=torch.float32, device=device).reshape(-1, 6)
        self.beams = torch.as_tensor(scene.beams, dtype=torch.float32, device=device).reshape(-1, 6)
        self.minimum_top = float(scene.starts[:, 2].min())
        n = config["map_samples"]
        axis = (torch.arange(n, device=device) - (n - 1) / 2) * config["map_spacing"]
        lateral, forward = torch.meshgrid(axis.flip(0), axis, indexing="ij")
        self.forward, self.lateral = forward.flatten(), lateral.flatten()

    def sample(self, count):
        index = torch.randint(len(self.starts), (count,), device=self.device)
        neighbor = (torch.rand(count, device=self.device) * self.goal_counts[index]).long()
        return self.starts[index], self.goals[index, neighbor]

    @staticmethod
    def within(x, y, boxes):
        return ((x[..., None] - boxes[:, 0]).abs() <= boxes[:, 2] / 2) & (
            (y[..., None] - boxes[:, 1]).abs() <= boxes[:, 3] / 2
        )

    def observations(self, state, target):
        yaw = yaw_wxyz(state[:, 3:7])
        c, s = yaw.cos(), yaw.sin()
        x = state[:, 0:1] + c[:, None] * self.forward - s[:, None] * self.lateral
        y = state[:, 1:2] + s[:, None] * self.forward + c[:, None] * self.lateral
        floor = torch.zeros_like(x)
        if len(self.platforms):
            floor = (self.within(x, y, self.platforms) * self.platforms[:, 5]).amax(-1)
        overhead = torch.zeros_like(x)
        if len(self.beams):
            overhead = (self.within(x, y, self.beams) * self.beams[:, 4]).amax(-1)
        present = overhead > 0
        # Represent the underside separately, since the same XY can have both
        # a walkable floor and an obstacle above it.
        grid = torch.stack((
            (floor - state[:, 2:3]).clamp(-2., 2.),
            torch.where(present, (overhead - state[:, 2:3]).clamp(-2., 2.), 0.),
            present.float(),
        ), dim=1).unflatten(-1, (self.config["map_samples"],) * 2)
        dx, dy = (target[:, :2] - state[:, :2]).unbind(-1)
        local = torch.stack((c * dx + s * dy, -s * dx + c * dy), dim=-1)
        return grid, local

    def outcome(self, state, target, previous_state=None):
        d2 = (state[:, :2] - target[:, :2]).square().sum(-1)
        reward = torch.exp(-self.config["position_scale"] * d2)
        reached = d2 <= self.config["success_radius"] ** 2
        if self.scene.lane_width is None:
            reached = reached & (state[:, 2] > target[:, 2] + 0.25)
            failed = state[:, 2] < self.minimum_top + 0.25
        else:
            length = self.scene.floors[0][2]
            failed = (state[:, 1].abs() > self.scene.lane_width / 2) | (
                (state[:, 0] < 0.) | (state[:, 0] > length)
            )
            if previous_state is not None:
                before = previous_state[:, :1] - self.beams[:, 0]
                after = state[:, :1] - self.beams[:, 0]
                crossed = (before < 0) != (after < 0)
                # At the beam's center plane, a crossing with the pelvis above
                # the underside is an overpass. MuJoCo handles body contacts.
                travel = after - before
                t = -before / torch.where(crossed, travel, 1.)
                z = previous_state[:, 2:3] + t * (state[:, 2:3] - previous_state[:, 2:3])
                failed = failed | (crossed & (z >= self.beams[:, 4])).any(-1)
        return reward, reached & ~failed, failed
