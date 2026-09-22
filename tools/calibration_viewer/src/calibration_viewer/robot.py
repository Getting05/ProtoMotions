from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Mapping

import numpy as np
import yourdfpy


@dataclass
class RobotModel:
    path: Path
    urdf: yourdfpy.URDF
    base_link: str
    keypoint_links: dict[str, str]
    visuals_loaded: bool
    collisions_loaded: bool
    visual_error: str | None = None

    @classmethod
    def load(
        cls, path: str | Path, *, base_link: str, keypoint_links: Mapping[str, str]
    ) -> "RobotModel":
        path = Path(path).resolve()
        visual_error = None
        try:
            urdf = yourdfpy.URDF.load(
                path,
                build_scene_graph=True,
                build_collision_scene_graph=True,
                load_meshes=True,
                load_collision_meshes=True,
                filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent),
            )
        except Exception as mesh_error:
            visual_error = f"{type(mesh_error).__name__}: {mesh_error}"
            try:
                urdf = yourdfpy.URDF.load(
                    path,
                    build_scene_graph=False,
                    build_collision_scene_graph=True,
                    load_meshes=False,
                    load_collision_meshes=True,
                    filename_handler=partial(yourdfpy.filename_handler_magic, dir=path.parent),
                )
            except Exception:
                raise mesh_error
        visuals_loaded = urdf.scene is not None and bool(urdf.scene.geometry)
        collisions_loaded = (
            urdf.collision_scene is not None and bool(urdf.collision_scene.geometry)
        )
        if not visuals_loaded and visual_error is None:
            visual_error = "URDF contains no loadable visual geometry"
        available = set(urdf.link_map)
        missing = {v for v in keypoint_links.values() if v not in available}
        if base_link not in available:
            missing.add(base_link)
        if missing:
            raise ValueError(f"URDF is missing configured links: {sorted(missing)}")
        return cls(
            path,
            urdf,
            base_link,
            dict(keypoint_links),
            visuals_loaded=visuals_loaded,
            collisions_loaded=collisions_loaded,
            visual_error=visual_error,
        )

    @property
    def actuated_joint_names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.urdf.actuated_joints)

    @property
    def joint_limits(self) -> dict[str, tuple[float, float]]:
        out = {}
        for joint in self.urdf.actuated_joints:
            limit = joint.limit
            lo = -np.pi if limit is None or limit.lower is None else float(limit.lower)
            hi = np.pi if limit is None or limit.upper is None else float(limit.upper)
            out[joint.name] = (lo, hi)
        return out

    def update(self, joint_positions: Mapping[str, float]) -> None:
        cfg = {name: float(joint_positions.get(name, 0.0)) for name in self.actuated_joint_names}
        self.urdf.update_cfg(cfg)

    def keypoints(self) -> dict[str, np.ndarray]:
        result = {}
        base_world = self.urdf.get_transform(self.base_link)
        world_base = np.linalg.inv(base_world)
        for name, link in self.keypoint_links.items():
            link_world = self.urdf.get_transform(link)
            result[name] = (world_base @ link_world)[:3, 3].copy()
        return result
