# SPDX-License-Identifier: Apache-2.0

"""Independent URDF FK and rigid-body velocities for observation reconstruction.

No simulator state, simulator kinematic model, or learned parameters are used.
Quaternions are xyzw; angular/linear velocities are in world axes. Heights
are pelvis-origin z minus ground height sampled under the observation anchor.
Only the output body/joint *names and order* come from the policy contract.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def _xyz(element, key="xyz", default="0 0 0"):
    return np.fromstring(element.get(key, default) if element is not None else default, sep=" ")


def _origin(element):
    return _xyz(element), Rotation.from_euler("xyz", _xyz(element, "rpy")).as_matrix()


class URDFKinematics:
    """Parse a tree and collapse fixed descendants' masses into selected bodies.

    The selected bodies must be the root plus every actuated joint's child.
    Fixed links are traversed even when absent from the observation, e.g. P2
    torso, palms and IMUs. Their inertial masses affect the merged body COM.
    """

    def __init__(self, urdf, body_names, dof_names):
        self.urdf = str(Path(urdf))
        xml = ET.parse(urdf).getroot()
        self.body_names, self.dof_names = list(body_names), list(dof_names)
        self.links = {link.get("name"): link for link in xml.findall("link")}
        self.joints = {}
        for joint in xml.findall("joint"):
            kind = joint.get("type")
            if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
                raise ValueError(f"Unsupported joint {joint.get('name')}: {kind}")
            if joint.find("mimic") is not None:
                raise ValueError("Mimic joints require an explicit mapping")
            child = joint.find("child").get("link")
            if child in self.joints:
                raise ValueError(f"Multiple parents for {child}")
            self.joints[child] = joint
        roots = set(self.links) - set(self.joints)
        if len(roots) != 1:
            raise ValueError(f"Expected one URDF root, got {roots}")
        self.root = roots.pop()
        if len(set(self.body_names)) != len(self.body_names) or len(set(self.dof_names)) != len(self.dof_names):
            raise ValueError("Duplicate body or DOF names")
        expected_bodies = {self.root} | {name for name, j in self.joints.items() if j.get("type") != "fixed"}
        expected_dofs = {j.get("name") for j in self.joints.values() if j.get("type") != "fixed"}
        if set(self.body_names) != expected_bodies or set(self.dof_names) != expected_dofs:
            raise ValueError("Body/DOF names must match the URDF articulated tree")
        self.edges = []
        visited = {self.root}
        pending = dict(self.joints)
        while pending:
            ready = [name for name, j in pending.items() if j.find("parent").get("link") in visited]
            if not ready:
                raise ValueError("Disconnected or cyclic URDF")
            for name in ready:
                j = pending.pop(name)
                pos, rot = _origin(j.find("origin"))
                axis = np.array([1., 0., 0.]) if j.get("type") == "fixed" else _xyz(j.find("axis"), default="1 0 0")
                if np.linalg.norm(axis) == 0:
                    raise ValueError(f"Zero joint axis: {j.get('name')}")
                axis /= np.linalg.norm(axis)
                idx = self.dof_names.index(j.get("name")) if j.get("type") != "fixed" else None
                self.edges.append((name, j.find("parent").get("link"), j.get("type"), pos, rot, axis, idx))
                visited.add(name)
        # Static transforms from each merged body to each fixed descendant.
        owners = {self.root: (self.root, np.zeros(3), np.eye(3))}
        for child, parent, kind, p, r, _, _ in self.edges:
            owner, pp, rr = owners[parent]
            owners[child] = (child, np.zeros(3), np.eye(3)) if kind != "fixed" else (owner, pp + rr @ p, rr @ r)
        masses = {name: 0.0 for name in self.body_names}
        moments = {name: np.zeros(3) for name in self.body_names}
        self.link_inertials = {}
        for name, link in self.links.items():
            inertial = link.find("inertial")
            mass = float(inertial.find("mass").get("value")) if inertial is not None else 0.0
            offset, _ = _origin(inertial.find("origin") if inertial is not None else None)
            self.link_inertials[name] = mass, offset
            owner, p, r = owners[name]
            masses[owner] += mass
            moments[owner] += mass * (p + r @ offset)
        self.masses = np.array([masses[n] for n in self.body_names])
        if np.any(self.masses <= 0):
            raise ValueError("Every merged body needs positive mass for COM velocity")
        self.com_offsets = np.array([moments[n] / masses[n] for n in self.body_names])

    def reconstruct(self, q, qd, root_rot, omega, velocity, height,
                    root_velocity_point="com", body_velocity_point="com"):
        """K(q,qd,R,omega,v,h); inputs [batch,D], [batch,4/3], [batch].

        No global x/y or ground elevation is needed for heading-local features.
        Root velocity may refer to 'origin' or 'com'. Output velocity may refer
        to 'origin', 'com', or legacy MuJoCo 'subtree_com' (the common tree COM
        reference used by data.cvel, *not* each body's own COM).
        """
        q, qd = np.asarray(q, dtype=np.float64), np.asarray(qd, dtype=np.float64)
        if q.ndim != 2:
            raise ValueError("q must be [batch, num_dofs]")
        batch = q.shape[0]
        if q.shape != (batch, len(self.dof_names)) or qd.shape != q.shape:
            raise ValueError("q and qd must be [batch, num_dofs]")
        root_rot = np.asarray(root_rot, dtype=np.float64)
        if root_rot.shape != (batch, 4):
            raise ValueError("root_rot must be [batch, 4] in xyzw order")
        for value in (omega, velocity):
            if np.asarray(value).shape != (batch, 3):
                raise ValueError("omega and velocity must be [batch, 3]")
        if np.asarray(height).shape not in {(batch,), (batch, 1)}:
            raise ValueError("height must be [batch] or [batch, 1]")
        if not all(np.isfinite(value).all() for value in (q, qd, root_rot, omega, velocity, height)):
            raise ValueError("K inputs must be finite")
        r = Rotation.from_quat(root_rot).as_matrix()
        w = np.asarray(omega, dtype=np.float64)
        v = np.asarray(velocity, dtype=np.float64).copy()
        p = np.zeros((batch, 3))
        p[:, 2] = np.asarray(height).reshape(batch)
        root_com = np.einsum("bij,j->bi", r, self.com_offsets[self.body_names.index(self.root)])
        if root_velocity_point == "com":
            v -= np.cross(w, root_com)
        elif root_velocity_point != "origin":
            raise ValueError(root_velocity_point)
        states = {self.root: (p, r, v, w)}
        for child, parent, kind, offset, fixed_rot, axis, idx in self.edges:
            pp, pr, pv, pw = states[parent]
            delta = np.einsum("bij,j->bi", pr, offset)
            cr = pr @ fixed_rot
            cp = pp + delta
            cv = pv + np.cross(pw, delta)
            cw = pw.copy()
            if kind in {"revolute", "continuous"}:
                aw = np.einsum("bij,j->bi", cr, axis)
                cr = cr @ Rotation.from_rotvec(q[:, idx, None] * axis).as_matrix()
                cw += aw * qd[:, idx, None]
            elif kind == "prismatic":
                aw = np.einsum("bij,j->bi", cr, axis)
                translation = aw * q[:, idx, None]
                cp += translation
                cv += np.cross(pw, translation) + aw * qd[:, idx, None]
            states[child] = cp, cr, cv, cw
        pos, mat, vel, ang = [np.stack([states[n][i] for n in self.body_names], axis=1) for i in range(4)]
        com_delta = np.einsum("bnij,nj->bni", mat, self.com_offsets)
        com_pos = pos + com_delta
        if body_velocity_point == "com":
            vel = vel + np.cross(ang, com_delta)
        elif body_velocity_point == "subtree_com":
            tree_com = np.sum(com_pos * self.masses[None, :, None], axis=1) / self.masses.sum()
            vel = vel + np.cross(ang, tree_com[:, None] - pos)
        elif body_velocity_point != "origin":
            raise ValueError(body_velocity_point)
        quat = Rotation.from_matrix(mat.reshape(-1, 3, 3)).as_quat().reshape(batch, -1, 4)
        return dict(body_pos=pos, body_rot=quat, body_vel=vel, body_ang_vel=ang)
