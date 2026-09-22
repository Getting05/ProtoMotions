from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import viser
from viser.extras import ViserUrdf

from .calibration import (
    CalibrationParams,
    calibrate_human_points,
    fit_calibration,
    geometry_metrics,
)
from .config import save_config
from .human import HumanSkeleton
from .robot import RobotModel


def _segments(points: dict[str, np.ndarray], edges: tuple[tuple[str, str], ...]) -> np.ndarray:
    segments = [[points[a], points[b]] for a, b in edges if a in points and b in points]
    return np.asarray(segments, dtype=float).reshape(-1, 2, 3)


def _metrics_markdown(human: dict[str, np.ndarray], robot: dict[str, np.ndarray], rmse: float) -> str:
    hm, rm = geometry_metrics(human), geometry_metrics(robot)
    labels = {
        "shoulder_width": "Shoulder width", "elbow_span": "Elbow span",
        "hip_width": "Hip width", "torso_length": "Torso length",
        "upper_arm": "Upper arm", "forearm": "Forearm",
        "thigh": "Thigh", "shank": "Shank",
    }
    rows = ["| measurement | SMPL target | robot |", "|---|---:|---:|"]
    for key, label in labels.items():
        a, b = hm[key], rm[key]
        rows.append(f"| {label} | {a:.3f} m | {b:.3f} m |" if a is not None and b is not None else f"| {label} | — | — |")
    rows.append(f"\n**Correspondence RMSE:** {rmse * 100:.2f} cm")
    return "\n".join(rows)


class CalibrationViewer:
    def __init__(
        self,
        human: HumanSkeleton,
        robot: RobotModel,
        config: dict[str, Any],
        *,
        output: Path,
        host: str,
        port: int,
    ) -> None:
        self.human = human
        self.robot = robot
        self.config = config
        self.output = output
        self.server = viser.ViserServer(host=host, port=port, label="Calibration Viewer")
        initial = config.get("calibration", {})
        self.params = CalibrationParams(
            np.asarray(initial.get("upper_scale", [1.0, 1.0, 1.0]), dtype=float),
            np.asarray(initial.get("lower_scale", [1.0, 1.0, 1.0]), dtype=float),
            float(initial.get("shoulder_offset", 0.0)),
            float(initial.get("elbow_offset", 0.0)),
        )
        self.joints = {name: 0.0 for name in robot.actuated_joint_names}
        self.joints.update({k: float(v) for k, v in config.get("robot", {}).get("t_pose_joint_positions", {}).items()})
        self.robot.update(self.joints)
        self.robot_vis = ViserUrdf(self.server, robot.urdf, root_node_name="/robot")
        self.robot_vis.update_cfg(np.array([self.joints[n] for n in robot.actuated_joint_names]))
        self.server.scene.add_grid("/ground", width=4, height=4, cell_size=0.1, section_size=1.0)
        self._build_gui()
        self._redraw()

    @property
    def raw_human(self) -> dict[str, np.ndarray]:
        points = self.human.by_name
        root = points["pelvis"]
        return {name: point - root for name, point in points.items()}

    def _build_gui(self) -> None:
        self.server.gui.add_markdown("# SMPL → URDF calibration\nBlue: calibrated SMPL targets · Red: robot link origins")
        with self.server.gui.add_folder("Morphology", expand_by_default=True):
            self.controls = {}
            defaults = list(self.params.upper_scale) + list(self.params.lower_scale) + [
                self.params.shoulder_offset, self.params.elbow_offset
            ]
            specs = [
                ("upper_x", 0.3, 2.0, 0.01), ("upper_y", 0.3, 2.0, 0.01), ("upper_z", 0.3, 2.0, 0.01),
                ("lower_x", 0.3, 2.0, 0.01), ("lower_y", 0.3, 2.0, 0.01), ("lower_z", 0.3, 2.0, 0.01),
                ("shoulder_offset_m", -0.20, 0.20, 0.005), ("elbow_offset_m", -0.20, 0.20, 0.005),
            ]
            for spec, initial in zip(specs, defaults, strict=True):
                label, lo, hi, step = spec
                handle = self.server.gui.add_slider(label, min=lo, max=hi, step=step, initial_value=initial)
                handle.on_update(lambda _, key=label: self._morphology_changed(key))
                self.controls[label] = handle
            fit = self.server.gui.add_button("Auto fit", color="blue")
            fit.on_click(lambda _: self._auto_fit())
        with self.server.gui.add_folder("Robot T-pose joints", expand_by_default=False):
            self.joint_controls = {}
            for name, (lo, hi) in self.robot.joint_limits.items():
                handle = self.server.gui.add_slider(name, min=lo, max=hi, step=0.01, initial_value=self.joints[name])
                handle.on_update(lambda _, key=name: self._joint_changed(key))
                self.joint_controls[name] = handle
        with self.server.gui.add_folder("Diagnostics", expand_by_default=True):
            self.metrics_gui = self.server.gui.add_markdown("Loading…")
            self.status_gui = self.server.gui.add_markdown("")
        export = self.server.gui.add_button("Export YAML", color="green")
        export.on_click(lambda _: self._export())

    def _read_controls(self) -> CalibrationParams:
        return CalibrationParams(
            np.array([self.controls[f"upper_{x}"].value for x in "xyz"]),
            np.array([self.controls[f"lower_{x}"].value for x in "xyz"]),
            float(self.controls["shoulder_offset_m"].value),
            float(self.controls["elbow_offset_m"].value),
        )

    def _morphology_changed(self, _: str) -> None:
        self.params = self._read_controls()
        self._redraw()

    def _joint_changed(self, name: str) -> None:
        self.joints[name] = float(self.joint_controls[name].value)
        self.robot.update(self.joints)
        self.robot_vis.update_cfg(np.array([self.joints[n] for n in self.robot.actuated_joint_names]))
        self._redraw()

    def _auto_fit(self) -> None:
        names = list(self.robot.keypoint_links)
        self.params, info = fit_calibration(self.raw_human, self.robot.keypoints(), names, initial=self.params)
        values = list(self.params.upper_scale) + list(self.params.lower_scale) + [self.params.shoulder_offset, self.params.elbow_offset]
        for handle, value in zip(self.controls.values(), values, strict=True):
            handle.value = float(value)
        warning = f"\n\n⚠️ {info['warning']}" if info["warning"] else ""
        self.status_gui.content = f"Fit rank: {info['rank']}/8 · condition: {info['condition']:.1f}{warning}"
        self._redraw()

    def _redraw(self) -> None:
        human = calibrate_human_points(self.raw_human, self.params)
        robot = self.robot.keypoints()
        common = [name for name in self.robot.keypoint_links if name in human and name in robot]
        error = [human[n] - robot[n] for n in common]
        rmse = float(np.sqrt(np.mean(np.square(error)))) if error else float("nan")
        human_array = np.asarray([human[n] for n in human])
        robot_array = np.asarray([robot[n] for n in robot])
        self.server.scene.add_point_cloud("/human/points", human_array, (55, 130, 255), point_size=0.025)
        self.server.scene.add_line_segments("/human/bones", _segments(human, self.human.edges), (55, 130, 255), thickness=0.008)
        self.server.scene.add_point_cloud("/robot/keypoints", robot_array, (245, 80, 70), point_size=0.027)
        residual_lines = np.asarray([[human[n], robot[n]] for n in common])
        self.server.scene.add_line_segments("/residuals", residual_lines, (245, 190, 45), thickness=0.003)
        self.metrics_gui.content = _metrics_markdown(human, robot, rmse)

    def _export(self) -> None:
        data = {
            "format_version": 1,
            "human": self.config.get("human", {}),
            "robot": {
                "base_link": self.robot.base_link,
                "keypoint_links": self.robot.keypoint_links,
                "t_pose_joint_positions": {k: float(v) for k, v in self.joints.items()},
            },
            "calibration": self.params.serializable(),
        }
        save_config(self.output, data)
        self.status_gui.content = f"✅ Saved `{self.output}`"

    def run(self) -> None:
        print(f"Calibration Viewer is running. Export path: {self.output}")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
