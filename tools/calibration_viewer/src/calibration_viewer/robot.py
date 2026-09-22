from __future__ import annotations

from dataclasses import dataclass
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

    @classmethod
    def load(
        cls, path: str | Path, *, base_link: str, keypoint_links: Mapping[str, str]
    ) -> "RobotModel":
        path = Path(path).resolve()
        try:
            urdf = yourdfpy.URDF.load(path, load_meshes=True)
        except Exception as mesh_error:
            try:
                urdf = yourdfpy.URDF.load(path, load_meshes=False)
            except Exception:
                raise mesh_error
        available = set(urdf.link_map)
        missing = {v for v in keypoint_links.values() if v not in available}
        if base_link not in available:
            missing.add(base_link)
        if missing:
            raise ValueError(f"URDF is missing configured links: {sorted(missing)}")
        return cls(path, urdf, base_link, dict(keypoint_links))

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

