"""URDF-only, batched foot constraints. Quaternions are xyzw; SI units throughout."""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn
from deployment.urdf_kinematics import URDFKinematics, _origin

DEFAULT_URDF = 'protomotions/data/assets/astro_p2/urdf/astro_p2_30dof_primitive_collision.urdf'
FEET = ('left_ankle_roll_link', 'right_ankle_roll_link')
ANCHOR = 'waist_roll_link'


def mv(r, v):
    return (r @ v.unsqueeze(-1)).squeeze(-1)


def quat_matrix(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def heading_inverse(r):
    """GPC heading: horizontal projection of anchor's positive X axis."""
    yaw = torch.atan2(r[..., 1, 0], r[..., 0, 0])
    c, s = yaw.cos(), yaw.sin()
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, s, z, -s, c, z, z, z, o), -1).reshape(*c.shape, 3, 3)


def tilt_from_gravity(gravity):
    """Any yaw gauge is valid: final heading-local velocity is yaw invariant."""
    up = -torch.nn.functional.normalize(gravity, dim=-1)
    ref = torch.zeros_like(up)
    ref[..., 0] = 1
    alt = torch.zeros_like(up)
    alt[..., 1] = 1
    ref = torch.where((up[..., :1].abs() > .95), alt, ref)
    x = torch.nn.functional.normalize(ref - (ref*up).sum(-1, keepdim=True)*up, dim=-1)
    y = torch.linalg.cross(up, x)
    return torch.stack((x, y, up), -2)


class FootKinematics(nn.Module):
    def __init__(self, urdf=DEFAULT_URDF, dof_names=None, body_names=None, imu='imu_2_link'):
        super().__init__()
        xml = ET.parse(urdf).getroot()
        joints = [j for j in xml.findall('joint') if j.get('type') != 'fixed']
        self.dof_names = dof_names or [j.get('name') for j in joints]
        self.body_names = body_names or ['pelvis']+[j.find('child').get('link') for j in joints]
        self.sha256 = hashlib.sha256(Path(urdf).read_bytes()).hexdigest()
        self.urdf, self.imu = str(urdf), imu
        tree = URDFKinematics(urdf, self.body_names, self.dof_names)
        if tree.root != 'pelvis' or len(self.dof_names) != 30:
            raise ValueError('Expected P2 pelvis-root 30 DOF tree')
        self.edges = []
        self.paths = {'pelvis': []}
        for i, (child, parent, kind, pos, rot, axis, index) in enumerate(tree.edges):
            if kind not in ('fixed', 'revolute', 'continuous'):
                raise ValueError('P2 estimator currently supports revolute/fixed joints')
            self.register_buffer(f'p{i}', torch.tensor(pos, dtype=torch.float32))
            self.register_buffer(f'r{i}', torch.tensor(rot, dtype=torch.float32))
            self.register_buffer(f'a{i}', torch.tensor(axis, dtype=torch.float32))
            self.edges.append((child, parent, index, i))
            self.paths[child] = self.paths[parent] + ([] if index is None else [index])
        # Initial version requires an IMU rigidly mounted on pelvis.
        if self.paths.get(imu) != []:
            raise ValueError('Use a pelvis-fixed IMU (P2 imu_2_link), not articulated torso IMU')
        static_p, static_r, _, _ = self.forward_tree(torch.zeros(1, 30))
        self.register_buffer('imu_position', static_p[imu][0].clone())
        self.register_buffer('imu_rotation', static_r[imu][0].clone())
        for f, foot in enumerate(FEET):
            centers, rotations, radii, lengths, cylinders = [], [], [], [], []
            for collision in tree.links[foot].findall('collision'):
                p, r = _origin(collision.find('origin'))
                geom = list(collision.find('geometry'))[0]
                if geom.tag not in ('sphere', 'cylinder'):
                    raise ValueError(f'Unsupported foot geometry {geom.tag}; use primitive URDF')
                centers.append(p); rotations.append(r)
                radii.append(float(geom.get('radius')))
                lengths.append(float(geom.get('length', '0'))/2)
                cylinders.append(geom.tag == 'cylinder')
            if not centers:
                raise ValueError(f'No foot collision primitives: {foot}')
            for name, values in [('centers', centers), ('rotations', rotations), ('radii', radii),
                                 ('lengths', lengths), ('cylinders', cylinders)]:
                self.register_buffer(f'{name}{f}', torch.tensor(np.asarray(values), dtype=torch.float32))

    def contract(self):
        return dict(urdf=self.urdf, sha256=self.sha256, dof_names=self.dof_names,
                    body_names=self.body_names, imu=self.imu, anchor=ANCHOR,
                    root='pelvis', velocity_point='origin', terrain='flat')

    def forward_tree(self, q):
        shape = q.shape[:-1]
        p = {'pelvis': q.new_zeros(*shape, 3)}
        r = {'pelvis': torch.eye(3, device=q.device, dtype=q.dtype).expand(*shape, 3, 3)}
        axes, origins = {}, {}
        for child, parent, index, i in self.edges:
            cp = p[parent] + mv(r[parent], getattr(self, f'p{i}'))
            cr = r[parent] @ getattr(self, f'r{i}')
            if index is not None:
                a = getattr(self, f'a{i}')
                axes[index], origins[index] = mv(cr, a), cp
                x, y, z = a.unbind()
                zero = x*0
                skew = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero)).reshape(3,3)
                angle = q[..., index, None, None]
                rr = torch.eye(3,device=q.device,dtype=q.dtype) + angle.sin()*skew + (1-angle.cos())*(skew@skew)
                cr = cr @ rr
            p[child], r[child] = cp, cr
        return p, r, axes, origins

    def support_points(self, rotations, up):
        """Lowest support of each primitive; average coplanar points within 1 um.

        This selects a material point at this instant. Its selection derivative
        is NOT part of the rigid-body contact-point velocity/Jacobian.
        """
        points = []
        for f, foot in enumerate(FEET):
            rf = rotations[foot]
            cg = getattr(self, f'rotations{f}')
            up_foot = mv(rf.transpose(-1,-2), up)
            direction = -mv(cg.transpose(-1,-2), up_foot.unsqueeze(-2))
            radial = direction.clone(); radial[..., 2] = 0
            radial = torch.nn.functional.normalize(radial, dim=-1)
            radius = getattr(self, f'radii{f}')[...,None]
            sphere = direction * radius
            cylinder = radial * radius
            cylinder[..., 2] = direction[..., 2].sign()*getattr(self, f'lengths{f}')
            local = torch.where(getattr(self, f'cylinders{f}')[...,None].bool(), cylinder, sphere)
            local = getattr(self, f'centers{f}') + mv(cg, local)
            z = (local * up_foot.unsqueeze(-2)).sum(-1)
            weight = (z <= z.min(-1,keepdim=True).values+1e-6).to(z.dtype)
            points.append((local*weight[...,None]).sum(-2)/weight.sum(-1,keepdim=True))
        return torch.stack(points, -2)

    def point_jacobians(self, q, local_points):
        p, r, axes, origins = self.forward_tree(q)
        positions, jacobians = [], []
        for f, foot in enumerate(FEET):
            point = p[foot]+mv(r[foot],local_points[...,f,:])
            columns = [torch.linalg.cross(axes[j], point-origins[j]) if j in self.paths[foot]
                       else torch.zeros_like(point) for j in range(30)]
            positions.append(point); jacobians.append(torch.stack(columns,-1))
        return torch.stack(positions,-2), torch.stack(jacobians,-3), r[ANCHOR]

    def forward(self, proprio, details=False):
        if proprio.shape[-1] != 129:
            raise ValueError('Expected [*,129] proprioception')
        q,dq = proprio[...,:30], proprio[...,30:60]
        gravity_b = mv(self.imu_rotation,proprio[...,90:93])
        omega_b = mv(self.imu_rotation,proprio[...,93:96])
        rw = tilt_from_gravity(gravity_b)
        _, rotations, _, _ = self.forward_tree(q)
        points = self.support_points(rotations,-torch.nn.functional.normalize(gravity_b,dim=-1))
        r, j, anchor_r = self.point_jacobians(q, points)
        vb = -torch.linalg.cross(omega_b.unsqueeze(-2).expand_as(r),r)-mv(j,dq.unsqueeze(-2))
        to_heading = heading_inverse(rw@anchor_r)@rw
        vh = mv(to_heading.unsqueeze(-3),vb)
        height = -mv(rw.unsqueeze(-3),r)[...,2:3]
        candidates = torch.cat((height,vh),-1)
        return (candidates,dict(points=points,positions=r,jacobians=j,to_heading=to_heading)) if details else candidates
