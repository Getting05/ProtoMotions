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
        show_robot_mesh: bool = True,
        show_smpl_mesh: bool = True,
    ) -> None:
        self.human = human
        self.robot = robot
        self.config = config
        self.output = output
        self.server = viser.ViserServer(host=host, port=port, label="Calibration Viewer")
        initial = config.get("calibration", {})
        self.params = CalibrationParams.from_mapping(initial)
        self._updating_controls = False
        self.joints = {name: 0.0 for name in robot.actuated_joint_names}
        self.joints.update({k: float(v) for k, v in config.get("robot", {}).get("t_pose_joint_positions", {}).items()})
        self.robot.update(self.joints)
        self.robot_root = None
        self.robot_vis = None
        robot_renderable = robot.visuals_loaded or robot.collisions_loaded
        if robot_renderable:
            self.robot_root = self.server.scene.add_frame(
                "/robot_mesh", show_axes=False, visible=show_robot_mesh
            )
            self.robot_vis = ViserUrdf(
                self.server,
                robot.urdf,
                root_node_name="/robot_mesh",
                load_meshes=robot.visuals_loaded,
                load_collision_meshes=robot.collisions_loaded,
                collision_mesh_color_override=(245, 80, 70, 0.42),
            )
            if robot.collisions_loaded:
                self.robot_vis.show_collision = not robot.visuals_loaded
            self.robot_vis.update_cfg(
                np.array([self.joints[n] for n in robot.actuated_joint_names])
            )
        self.server.scene.add_grid("/ground", width=4, height=4, cell_size=0.1, section_size=1.0)
        self._build_gui(show_robot_mesh, show_smpl_mesh)
        self._redraw()

    @property
    def raw_human(self) -> dict[str, np.ndarray]:
        points = self.human.by_name
        root = points["pelvis"]
        return {name: point - root for name, point in points.items()}

    @property
    def raw_human_vertices(self) -> np.ndarray | None:
        if self.human.vertices is None:
            return None
        pelvis = self.human.by_name["pelvis"]
        return self.human.vertices - pelvis

    def _build_gui(self, show_robot_mesh: bool, show_smpl_mesh: bool) -> None:
        self.server.gui.add_markdown(
            "# SMPL → URDF calibration\n"
            "Blue: SMPL · Red: robot link origins · Yellow: correspondence residuals"
        )
        with self.server.gui.add_folder("Visibility", expand_by_default=True):
            self.show_robot_mesh = self.server.gui.add_checkbox(
                "URDF model",
                initial_value=show_robot_mesh
                and (self.robot.visuals_loaded or self.robot.collisions_loaded),
            )
            self.show_smpl_mesh = self.server.gui.add_checkbox(
                "Source SMPL mesh (unwarped)", initial_value=show_smpl_mesh and self.human.vertices is not None
            )
            self.show_human_skeleton = self.server.gui.add_checkbox(
                "SMPL skeleton", initial_value=True
            )
            self.show_robot_keypoints = self.server.gui.add_checkbox(
                "Robot keypoints", initial_value=True
            )
            self.show_residuals = self.server.gui.add_checkbox(
                "Residual lines", initial_value=True
            )
            for handle in (
                self.show_robot_mesh,
                self.show_smpl_mesh,
                self.show_human_skeleton,
                self.show_robot_keypoints,
                self.show_residuals,
            ):
                handle.on_update(lambda _: self._visibility_changed())
        with self.server.gui.add_folder("Morphology", expand_by_default=True):
            self.mode_control = self.server.gui.add_dropdown(
                "Mode", options=("legacy", "bone"), initial_value=self.params.mode)
            self.asymmetric_control = self.server.gui.add_checkbox(
                "Independent left/right bones", initial_value=self.params.asymmetric)
            self.mode_control.on_update(lambda _: self._mode_changed())
            self.asymmetric_control.on_update(lambda _: self._mode_changed())
            self.parameter_folder = self.server.gui.add_folder("Active parameters")
            self.controls = {}
            self._build_parameter_controls()
            fit = self.server.gui.add_button("Auto fit", color="blue")
            fit.on_click(lambda _: self._auto_fit())
        with self.server.gui.add_folder("Robot T-pose joints", expand_by_default=False):
            self.joint_controls = {}
            for name, (lo, hi) in self.robot.joint_limits.items():
                handle = self.server.gui.add_slider(name, min=lo, max=hi, step=0.01, initial_value=self.joints[name])
                handle.on_update(lambda _, key=name: self._joint_changed(key))
                self.joint_controls[name] = handle
        with self.server.gui.add_folder("Diagnostics", expand_by_default=True):
            if self.robot.visuals_loaded:
                robot_mesh_status = "✅ URDF visual mesh loaded"
            elif self.robot.collisions_loaded:
                robot_mesh_status = (
                    "⚠️ URDF visual mesh files unavailable; showing collision geometry instead"
                )
            else:
                robot_mesh_status = (
                    f"⚠️ URDF geometry unavailable: {self.robot.visual_error}"
                )
            smpl_mesh_status = (
                "✅ SMPL body mesh loaded"
                if self.human.vertices is not None
                else "⚠️ SMPL body mesh unavailable: the pkl/model did not provide vertices and faces"
            )
            self.mesh_status_gui = self.server.gui.add_markdown(
                f"{robot_mesh_status}\n\n{smpl_mesh_status}"
            )
            self.metrics_gui = self.server.gui.add_markdown("Loading…")
            self.status_gui = self.server.gui.add_markdown("")
        export = self.server.gui.add_button("Export YAML", color="green")
        export.on_click(lambda _: self._export())

    def _build_parameter_controls(self) -> None:
        for handle in self.controls.values():
            handle.remove()
        self.controls = {}
        with self.parameter_folder:
            for (label, lo, hi, step), initial in zip(self.params.parameter_specs(), self.params.vector(), strict=True):
                # Preserve imported positive scales / offsets outside the fitting bounds.
                handle = self.server.gui.add_slider(label, min=min(lo, float(initial)), max=max(hi, float(initial)),
                                                    step=step, initial_value=float(initial))
                handle.on_update(lambda _: self._morphology_changed())
                self.controls[label] = handle

    def _mode_changed(self) -> None:
        if self._updating_controls:
            return
        self.params = self._read_controls()
        self.params.mode = self.mode_control.value
        self.params.asymmetric = self.asymmetric_control.value
        self._build_parameter_controls()
        self._redraw()

    def _read_controls(self) -> CalibrationParams:
        return self.params.with_vector(np.array([h.value for h in self.controls.values()]))

    def _morphology_changed(self) -> None:
        if not self._updating_controls:
            self.params = self._read_controls()
            self._redraw()

    def _joint_changed(self, name: str) -> None:
        self.joints[name] = float(self.joint_controls[name].value)
        self.robot.update(self.joints)
        if self.robot_vis is not None:
            self.robot_vis.update_cfg(
                np.array([self.joints[n] for n in self.robot.actuated_joint_names])
            )
        self._redraw()

    def _visibility_changed(self) -> None:
        if self.robot_root is not None:
            self.robot_root.visible = bool(self.show_robot_mesh.value)
        for handle_name, visible in (
            ("smpl_mesh_handle", self.show_smpl_mesh.value),
            ("human_points_handle", self.show_human_skeleton.value),
            ("human_bones_handle", self.show_human_skeleton.value),
            ("robot_points_handle", self.show_robot_keypoints.value),
            ("residuals_handle", self.show_residuals.value),
        ):
            handle = getattr(self, handle_name, None)
            if handle is not None:
                handle.visible = bool(visible)

    def _auto_fit(self) -> None:
        names = list(self.robot.keypoint_links)
        try:
            fitted, info = fit_calibration(self.raw_human, self.robot.keypoints(), names, initial=self.params)
        except ValueError as exc:
            self.status_gui.content = f"⚠️ Fit failed: {exc}"
            return
        self._updating_controls = True
        try:
            self.params = fitted
            for handle, value in zip(self.controls.values(), fitted.vector(), strict=True):
                handle.value = float(value)
        finally:
            self._updating_controls = False
        warning = f"\n\n⚠️ {info['warning']}" if info["warning"] else ""
        self.status_gui.content = f"Fit rank: {info['rank']}/{info['parameter_count']} · condition: {info['condition']:.1f}{warning}"
        self._redraw()

    def _redraw(self) -> None:
        try:
            human = calibrate_human_points(self.raw_human, self.params)
        except ValueError as exc:
            self.status_gui.content = f"⚠️ Invalid calibration: {exc}"
            return
        robot = self.robot.keypoints()
        common = [name for name in self.robot.keypoint_links if name in human and name in robot]
        error = [human[n] - robot[n] for n in common]
        rmse = float(np.sqrt(np.mean(np.square(error)))) if error else float("nan")
        human_array = np.asarray([human[n] for n in human])
        robot_array = np.asarray([robot[n] for n in robot])
        vertices = self.raw_human_vertices
        if vertices is not None and self.human.faces is not None:
            self.smpl_mesh_handle = self.server.scene.add_mesh_simple(
                "/human/mesh",
                vertices,
                self.human.faces,
                color=(55, 130, 255),
                opacity=0.28,
                side="double",
                visible=bool(self.show_smpl_mesh.value),
            )
        self.human_points_handle = self.server.scene.add_point_cloud(
            "/human/points",
            human_array,
            (55, 130, 255),
            point_size=0.025,
            visible=bool(self.show_human_skeleton.value),
        )
        self.human_bones_handle = self.server.scene.add_line_segments(
            "/human/bones",
            _segments(human, self.human.edges),
            (55, 130, 255),
            thickness=0.008,
            visible=bool(self.show_human_skeleton.value),
        )
        self.robot_points_handle = self.server.scene.add_point_cloud(
            "/robot_debug/keypoints",
            robot_array,
            (245, 80, 70),
            point_size=0.027,
            visible=bool(self.show_robot_keypoints.value),
        )
        residual_lines = np.asarray([[human[n], robot[n]] for n in common])
        self.residuals_handle = self.server.scene.add_line_segments(
            "/residuals",
            residual_lines,
            (245, 190, 45),
            thickness=0.003,
            visible=bool(self.show_residuals.value),
        )
        self.metrics_gui.content = _metrics_markdown(human, robot, rmse)

    def _export(self) -> None:
        robot_config = dict(self.config.get("robot", {}))
        robot_config.update(
            {
                "base_link": self.robot.base_link,
                "keypoint_links": self.robot.keypoint_links,
                "t_pose_joint_positions": {
                    k: float(v) for k, v in self.joints.items()
                },
            }
        )
        data = {
            "format_version": 2,
            "human": self.config.get("human", {}),
            "robot": robot_config,
            "calibration": self.params.serializable(),
            "root_trajectory": self.config.get("root_trajectory", {"scale": [1., 1., 1.], "offset": [0., 0., 0.]}),
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
