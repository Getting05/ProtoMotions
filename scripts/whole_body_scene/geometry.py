"""Collision geometry, deterministic surface samples and exact kinematic replay.

Uses the simulator's MJCF collision primitives, including geom-local transforms.
Visual meshes never become collision geometry. Unsupported types fail explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def tangent_basis(normal):
    n = unit(np.asarray(normal, dtype=float))
    axis = np.eye(3)[np.argmin(np.abs(n))]
    u = unit(np.cross(axis, n))
    return np.stack((u, np.cross(n, u), n), axis=1)


def primitive_sdf(points, kind, size):
    """Signed distance in geom coordinates; negative means inside a solid."""
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.linalg.norm(points, axis=-1) - size[0]
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        q = points.copy()
        q[..., 2] -= np.clip(q[..., 2], -size[1], size[1])
        return np.linalg.norm(q, axis=-1) - size[0]
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        q = np.stack((np.linalg.norm(points[..., :2], axis=-1)-size[0],
                      np.abs(points[..., 2])-size[1]), axis=-1)
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        q = np.abs(points) - size
    else:
        raise ValueError(f"Unsupported collision primitive {kind}")
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)


def primitive_samples(kind, size, count, rng):
    # Explicit cardinal samples ensure extremal surfaces, especially foot soles,
    # remain represented even at low sampling density.
    axes = np.concatenate((np.eye(3), -np.eye(3)))
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        normals = np.concatenate((axes, unit(rng.normal(size=(count, 3)))))
        return normals * size[0], normals
    if kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        r, h = size[:2]
        a = np.concatenate((np.arange(8)*np.pi/4, rng.uniform(0, 2*np.pi, count)))
        normals = np.stack((np.cos(a), np.sin(a), np.zeros_like(a)), axis=1)
        p = r*normals
        p[:, 2] = np.concatenate((np.zeros(8), rng.uniform(-h, h, count)))
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            cap_n = np.concatenate((axes, unit(rng.normal(size=(count, 3)))))
            cap_p = r*cap_n
            cap_p[:, 2] += np.where(cap_n[:, 2] >= 0, h, -h)
        else:
            angle = rng.uniform(0, 2*np.pi, count)
            radii = r*np.sqrt(rng.uniform(size=count))
            cap_p = np.column_stack((radii*np.cos(angle), radii*np.sin(angle),
                                      rng.choice([-h, h], count)))
            cap_p = np.concatenate((cap_p, [[0, 0, h], [0, 0, -h]]))
            cap_n = np.zeros_like(cap_p)
            cap_n[:, 2] = np.sign(cap_p[:, 2])
        return np.concatenate((p, cap_p)), np.concatenate((normals, cap_n))
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        pts, ns = [], []
        for n in axes:
            p = rng.uniform(-1, 1, (max(4, count//6), 3))*size
            p[:, np.flatnonzero(n)[0]] = n[np.flatnonzero(n)[0]]*size[np.flatnonzero(n)[0]]
            pts.append(np.concatenate((p, [n*size])))
            ns.append(np.tile(n, (len(p)+1, 1)))
        return np.concatenate(pts), np.concatenate(ns)
    raise ValueError(f"Unsupported collision primitive {kind}; no visual fallback")


@dataclass
class Surface:
    points: np.ndarray
    normals: np.ndarray
    body: np.ndarray
    geom: np.ndarray


class CollisionModel:
    def __init__(self, path, samples_per_geom=24, seed=20261016, sampling="random"):
        self.path = Path(path).resolve()
        if sampling not in ("random", "symmetric"):
            raise ValueError(f"Unknown collision sampling scheme: {sampling}")
        self.sampling = sampling
        self.asset_sha256 = sha256(path)
        tree = ET.parse(path)
        root = tree.getroot()
        # P2 explicitly marks visual geometry. Never load the large visual STL
        # files into the offline collision-only replay model.
        for parent in root.iter():
            for child in list(parent):
                if child.tag == "geom" and (child.get("class") == "visual" or
                    (child.get("contype") == "0" and child.get("conaffinity") == "0")):
                    parent.remove(child)
        used_meshes = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
        for asset in root.findall("asset"):
            for child in list(asset):
                if child.tag == "mesh" and child.get("name") not in used_meshes:
                    asset.remove(child)
        if used_meshes:
            raise ValueError("Collision meshes require an explicit mesh-distance backend; do not substitute visuals")
        self.xml = ET.tostring(root, encoding="unicode")
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                           for i in range(1, m.nbody)]
        self.geom_ids = np.flatnonzero((m.geom_contype | m.geom_conaffinity) != 0)
        self.collision_bodies = np.unique(m.geom_bodyid[self.geom_ids]-1)
        self.joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
                            for i in range(1, m.njnt)]
        rng = np.random.default_rng(seed)
        points, normals, bodies, geoms = [], [], [], []
        for g in self.geom_ids:
            if sampling == "symmetric":
                from .sampling_v2 import primitive_samples_deterministic
                p, n = primitive_samples_deterministic(m.geom_type[g], m.geom_size[g], samples_per_geom)
            else:
                p, n = primitive_samples(m.geom_type[g], m.geom_size[g], samples_per_geom, rng)
            rot = Rotation.from_quat(m.geom_quat[g][[1, 2, 3, 0]]).as_matrix()
            p = p@rot.T + m.geom_pos[g]
            n = n@rot.T
            # Remove surfaces buried in another collider of the same rigid body.
            exposed = np.ones(len(p), dtype=bool)
            for other in self.geom_ids:
                if other == g or m.geom_bodyid[other] != m.geom_bodyid[g]:
                    continue
                r = Rotation.from_quat(m.geom_quat[other][[1, 2, 3, 0]]).as_matrix()
                exposed &= primitive_sdf((p-m.geom_pos[other])@r,
                                          m.geom_type[other], m.geom_size[other]) >= -1e-5
            points.append(p[exposed]); normals.append(n[exposed])
            bodies.extend([m.geom_bodyid[g]-1]*int(exposed.sum()))
            geoms.extend([g]*int(exposed.sum()))
        self.surface = Surface(np.concatenate(points), np.concatenate(normals),
                               np.asarray(bodies), np.asarray(geoms))
        if set(self.surface.body) != set(self.collision_bodies):
            raise ValueError("At least one effective collision body has no exposed samples")

    def qpos(self, motion):
        q = np.concatenate((motion.pos[:, 0], motion.rot[:, 0, [3, 0, 1, 2]], motion.dof), axis=1)
        if q.shape[1] != self.model.nq:
            raise ValueError(f"Motion DOFs disagree with collision model: {q.shape[1]} != {self.model.nq}")
        return q

    def verify_fk(self, motion):
        errors, angle_errors = [], []
        for t in sorted(set([0, len(motion.pos)//2, len(motion.pos)-1])):
            self.data.qpos[:] = self.qpos(motion)[t]
            mujoco.mj_kinematics(self.model, self.data)
            errors.append(np.abs(self.data.xpos[1:]-motion.pos[t]).max())
            q = self.data.xquat[1:, [1, 2, 3, 0]]
            angle_errors.append((1-np.abs(np.sum(q*motion.rot[t], axis=1))).max())
        if max(errors) > 2e-4 or max(angle_errors) > 2e-5:
            raise ValueError(f"Motion/model body order or FK mismatch: {errors}, {angle_errors}")
        return float(max(errors))

    def world_samples(self, motion):
        s = self.surface
        r = Rotation.from_quat(motion.rot.reshape(-1, 4)).as_matrix().reshape(*motion.rot.shape[:2], 3, 3)
        world_offset = np.einsum("tsij,sj->tsi", r[:, s.body], s.points)
        p = motion.pos[:, s.body] + world_offset
        n = np.einsum("tsij,sj->tsi", r[:, s.body], s.normals)
        return p, n, r

    def floor_offset(self, motion, rotations, tolerance):
        """Exact vertical lower extent of the primitive sweep, in source frame.

        A descending clip may legitimately have negative world z. One constant
        translation (not per-frame grounding) places its lowest solid surface
        above the simulator floor and is saved as provenance.
        """
        m=self.model
        lower=np.inf
        for g in self.geom_ids:
            b=m.geom_bodyid[g]-1
            gr=Rotation.from_quat(m.geom_quat[g][[1,2,3,0]]).as_matrix()
            r=rotations[:,b]@gr
            center=motion.pos[:,b]+np.einsum("tij,j->ti",rotations[:,b],m.geom_pos[g])
            kind=m.geom_type[g];size=m.geom_size[g]
            if kind==mujoco.mjtGeom.mjGEOM_SPHERE: extent=size[0]
            elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE: extent=size[0]+size[1]*np.abs(r[:,2,2])
            elif kind==mujoco.mjtGeom.mjGEOM_CYLINDER:
                extent=size[1]*np.abs(r[:,2,2])+size[0]*np.sqrt(np.maximum(0,1-r[:,2,2]**2))
            elif kind==mujoco.mjtGeom.mjGEOM_BOX: extent=np.abs(r[:,2])@size
            else: raise ValueError(f"Unsupported extent geometry {kind}")
            lower=min(lower,float(np.min(center[:,2]-extent)))
        return np.array([0.,0.,-lower+.001 if lower < -tolerance else 0.])

    def minimum_swept_distance(self, points, motion, rotations, exclude_body=None):
        """Distance to the entire solid robot sweep, all frames and colliders.

        A contact point on a later intersecting body is rejected even if the
        contact frame itself is valid. Batching bounds temporary memory.
        """
        m = self.model
        best = np.full(len(points), np.inf)
        for g in self.geom_ids:
            b = int(m.geom_bodyid[g]-1)
            if exclude_body is not None and b == exclude_body:
                continue
            gr = Rotation.from_quat(m.geom_quat[g][[1, 2, 3, 0]]).as_matrix()
            for a in range(0, len(points), 128):
                local = np.einsum("tpj,tjk->tpk", points[None, a:a+128]-motion.pos[:, b, None], rotations[:, b])
                local = (local-m.geom_pos[g])@gr
                dist = primitive_sdf(local, m.geom_type[g], m.geom_size[g]).min(axis=0)
                best[a:a+128] = np.minimum(best[a:a+128], dist)
        return best

    def report(self):
        return {"asset": str(self.path), "asset_sha256": self.asset_sha256,
                "body_names": self.body_names, "collision_body_ids": self.collision_bodies.tolist(),
                "collision_body_names": [self.body_names[i] for i in self.collision_bodies],
                "non_collision_body_names": [n for i,n in enumerate(self.body_names) if i not in self.collision_bodies],
                "collision_geometries": len(self.geom_ids), "surface_samples": len(self.surface.points),
                "sampling_scheme": self.sampling,
                "samples_by_body": {n: int((self.surface.body == i).sum()) for i,n in enumerate(self.body_names)}}
