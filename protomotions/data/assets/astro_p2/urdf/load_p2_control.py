"""Read the P2 control extension without depending on a simulator or ROS.

This reader returns configuration. The caller applies PD gains and armature
through its simulator/controller API; parsing ordinary URDF alone does not.
"""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

NAMESPACE = 'urn:protomotions:p2:control:1'
NS = {'p2': NAMESPACE}
FIELDS = ('stiffness', 'damping', 'armature', 'friction', 'effort_limit',
          'velocity_limit', 'default_position')


def load_control_parameters(urdf_path):
    root = ET.parse(urdf_path).getroot()
    initial = root.find('p2:initial_state', NS)
    if initial is None:
        raise ValueError('URDF has no P2 control extension')
    result = {
        'schema': NAMESPACE,
        'source_revision': initial.get('source_revision'),
        'source_control_config': 'protomotions/robot_configs/astro_p2.py',
        'source_pose_config': 'protomotions/robot_configs/astro_p2_pose.py',
        'default_root_height': float(initial.get('root_height')),
        'soft_joint_pos_limit_factor': float(initial.get('soft_joint_pos_limit_factor')),
        'joints': {},
    }
    for joint in root.findall('joint'):
        if joint.get('type') == 'fixed':
            continue
        name = joint.get('name')
        ext = joint.find('p2:control', NS)
        limit = joint.find('limit')
        if name in result['joints'] or ext is None or limit is None:
            raise ValueError(f'Missing or duplicate control metadata: {name}')
        values = {field: float(ext.get(field)) for field in FIELDS}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f'Non-finite control value: {name}')
        if any(values[k] <= 0 for k in ('stiffness', 'damping', 'armature', 'effort_limit', 'velocity_limit')):
            raise ValueError(f'Non-positive motor parameter: {name}')
        if values['friction'] < 0:
            raise ValueError(f'Negative friction: {name}')
        if not float(limit.get('lower')) <= values['default_position'] <= float(limit.get('upper')):
            raise ValueError(f'Default position outside physical limits: {name}')
        if values['effort_limit'] != float(limit.get('effort')) or values['velocity_limit'] != float(limit.get('velocity')):
            raise ValueError(f'Control and URDF limits differ: {name}')
        result['joints'][name] = values
    if len(result['joints']) != 30:
        raise ValueError('P2 extension must cover all 30 joints, including legacy head_joint')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('urdf', type=Path)
    args = parser.parse_args()
    print(json.dumps(load_control_parameters(args.urdf), indent=2))
