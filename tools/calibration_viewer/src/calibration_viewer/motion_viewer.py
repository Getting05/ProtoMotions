"""World-coordinate playback of exported targets and solved robot motion."""
from __future__ import annotations

import time
import threading

import numpy as np
from scipy.spatial.transform import Rotation
import viser
from viser.extras import ViserUrdf

from .human import SMPL_EDGES


def preview_motion(robot, data: dict, *, host="127.0.0.1", port=8080) -> None:
    server = viser.ViserServer(host=host, port=port, label="PyRoki motion preview")
    names = list(data["keypoint_names"])
    joint_names = list(data["joint_names"])
    count = len(data["joint_positions"])
    frame = server.gui.add_slider("Frame", min=0, max=max(count-1, 1), step=1, initial_value=0)
    playing = server.gui.add_checkbox("Playing", initial_value=count > 1)
    status = server.gui.add_markdown("")
    server.gui.add_markdown("Blue: calibrated human targets · Red: robot FK.\n\nKinematic preview; contact, collision and dynamics are not validated.")
    server.scene.add_grid("/ground", width=10, height=10)
    root_handle = server.scene.add_frame("/robot_mesh", show_axes=False)
    visual = None
    if robot.visuals_loaded or robot.collisions_loaded:
        visual = ViserUrdf(server, robot.urdf, root_node_name="/robot_mesh",
                           load_meshes=robot.visuals_loaded, load_collision_meshes=robot.collisions_loaded)
        if robot.collisions_loaded and not robot.visuals_loaded:
            visual.show_collision = True
    roots = data["pelvis_positions"]
    if count > 1:
        server.scene.add_line_segments("/trajectory", np.stack([roots[:-1], roots[1:]], axis=1),
                                       (80, 220, 160), thickness=.006)
    redraw_lock = threading.RLock()

    def redraw(_=None):
        with redraw_lock, server.atomic():
            _draw_frame()

    def _draw_frame():
        t = min(frame.value, count - 1)
        joints = dict(zip(joint_names, data["joint_positions"][t], strict=True))
        robot.update(joints)
        world_base = np.eye(4)
        world_base[:3, :3] = Rotation.from_quat(data["root_quat_xyzw"][t]).as_matrix()
        world_base[:3, 3] = data["root_positions"][t]
        world_urdf = world_base @ np.linalg.inv(robot.urdf.get_transform(robot.base_link))
        root_handle.position = world_urdf[:3, 3]
        xyzw = Rotation.from_matrix(world_urdf[:3, :3]).as_quat()
        root_handle.wxyz = xyzw[[3, 0, 1, 2]]
        if visual:
            visual.update_cfg(np.array([joints[n] for n in robot.actuated_joint_names]))
        for group, color in (("target", (55, 130, 255)), ("robot", (245, 80, 70))):
            points = data[f"{group}_keypoints"][t]
            server.scene.add_point_cloud(f"/{group}/points", points, color, point_size=.025)
            edges = np.asarray([[points[names.index(a)], points[names.index(b)]]
                                for a, b in SMPL_EDGES if a in names and b in names]).reshape(-1, 2, 3)
            server.scene.add_line_segments(f"/{group}/bones", edges, color, thickness=.008)
        status.content = f"Frame {t}/{count-1} · {t / float(data['fps']):.2f} s · positional RMSE {data['frame_rmse_m'][t]*100:.2f} cm"
    frame.on_update(redraw)
    redraw()
    try:
        while True:
            time.sleep(1. / float(data["fps"]))
            if playing.value and count > 1:
                frame.value = (frame.value + 1) % count
                redraw()
    except KeyboardInterrupt:
        server.stop()
