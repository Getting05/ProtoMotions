"""Generate a redistributable 8-DOF humanoid fixture and synchronized motions.

Run: python examples/generate_demo.py --output /tmp/calibration-demo
No licensed SMPL assets or external robot downloads are needed.
"""
from pathlib import Path
import argparse

import numpy as np
from scipy.spatial.transform import Rotation

from calibration_viewer.config import save_config
from calibration_viewer.human import SMPL_EDGES, SMPL_JOINT_NAMES
from calibration_viewer.robot import RobotModel


def generate(output: Path, frames: int = 12) -> None:
    output.mkdir(parents=True, exist_ok=True)
    origins = {}
    for _, child in SMPL_EDGES:
        side = 1 if child.startswith("left") else -1
        if child.startswith("spine") or child in ("neck", "head"):
            offset = [0, 0, .12]
        elif child.endswith(("hip", "collar", "shoulder")):
            offset = [0, side*.09, 0]
        elif child.endswith(("elbow", "wrist", "hand")):
            offset = [0, side*.22, 0]
        elif child.endswith(("knee", "ankle")):
            offset = [0, 0, -.35]
        else:
            offset = [.12, 0, 0]
        origins[child] = np.array(offset)
    xml = '<robot name="calibration_demo">'
    for name in SMPL_JOINT_NAMES:
        xml += f'<link name="{name}"><visual><geometry><sphere radius="0.025"/></geometry></visual></link>'
    for parent, child in SMPL_EDGES:
        active = child.endswith(("shoulder", "elbow", "hip", "knee"))
        kind = 'revolute' if active else 'fixed'
        xyz = ' '.join(map(str, origins[child]))
        axis = '0 0 1' if child.endswith('elbow') else '1 0 0' if child.endswith('shoulder') else '0 1 0'
        xml += f'<joint name="{child}_joint" type="{kind}"><parent link="{parent}"/><child link="{child}"/><origin xyz="{xyz}"/>'
        if active:
            xml += f'<axis xyz="{axis}"/><limit lower="-1.4" upper="1.4" effort="50" velocity="10"/>'
        xml += '</joint>'
    xml += '</robot>'
    urdf = output/'robot.urdf'
    urdf.write_text(xml)
    mapping = {n:n for n in SMPL_JOINT_NAMES}
    robot = RobotModel.load(urdf, base_link='pelvis', keypoint_links=mapping)
    time = np.linspace(0, 1, frames)
    q = np.array([.25*np.sin(2*np.pi*time + i*.7) for i in range(len(robot.actuated_joint_names))]).T
    roots = np.column_stack([.5*time, .2*time**2, .95 + .04*np.sin(np.pi*time)])
    rotations = Rotation.from_euler('z', (.5*time)[:, None])
    all_points = []
    for index, row in enumerate(q):
        robot.update(dict(zip(robot.actuated_joint_names, row, strict=True)))
        local = np.array([robot.keypoints()[n] for n in SMPL_JOINT_NAMES])
        all_points.append(local @ rotations.as_matrix()[index].T + roots[index])
    np.savez_compressed(output/'human.npz', joints=np.array(all_points), joint_names=np.array(SMPL_JOINT_NAMES),
                        root_quat_xyzw=rotations.as_quat(), fps=30.)
    np.savez_compressed(output/'reference.npz', joint_positions=q, joint_names=np.array(robot.actuated_joint_names),
                        root_positions=roots, root_quat_xyzw=rotations.as_quat(), fps=30., base_link='pelvis')
    save_config(output/'config.yaml', {'format_version': 2, 'human': {'axes':['x','y','z']},
                'robot': {'base_link':'pelvis', 'keypoint_links':mapping},
                'calibration': {'mode':'bone'}, 'root_trajectory': {'scale':[1.,1.,1.], 'offset':[0.,0.,0.]}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=12)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error('--frames must be at least 2')
    generate(args.output, args.frames)
