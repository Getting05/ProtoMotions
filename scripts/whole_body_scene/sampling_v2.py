"""Deterministic, reflection-symmetric collision-surface quadrature.

Density is a resolution budget, never a number of contact votes. Symmetric
primitives use symmetric point sets, independent of geometry order and seed.
"""
import numpy as np
import mujoco


def primitive_samples_deterministic(kind, size, count):
    size = np.asarray(size, float)
    rings = max(8, 8 * int(np.ceil(np.sqrt(count * 2) / 8)))
    theta = np.arange(rings) * (2 * np.pi / rings)
    circle = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(rings)))
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        resolution = max(3, int(np.ceil(np.sqrt(count / 6))))
        resolution += 1 - resolution % 2  # Include the exact face center.
        grid = np.linspace(-1, 1, resolution)
        points, normals = [], []
        for axis in range(3):
            other = [j for j in range(3) if j != axis]
            for sign in [-1, 1]:
                a, b = np.meshgrid(grid, grid)
                p = np.zeros((a.size, 3)); p[:, axis] = sign
                p[:, other[0]] = a.ravel(); p[:, other[1]] = b.ravel()
                n = np.zeros_like(p); n[:, axis] = sign
                points.append(p * size); normals.append(n)
        return np.concatenate(points), np.concatenate(normals)
    if kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE):
        levels = max(3, int(np.ceil(count / rings)))
        levels += 1 - levels % 2
        mu = np.linspace(-1, 1, levels + 2)[1:-1]
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            # Both equators are already included in the cylinder side grid.
            mu = mu[np.abs(mu) > 1e-12]
        ns = np.concatenate([circle * np.sqrt(1 - z*z) + [0, 0, z] for z in mu])
        ns = np.concatenate((ns, [[0, 0, -1], [0, 0, 1]]))
        p = size[0] * ns
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            return p, ns
        p[:, 2] += np.where(ns[:, 2] >= 0, size[1], -size[1])
        zs = np.linspace(-size[1], size[1], max(3, int(np.ceil(count / rings))))
        side = np.concatenate([circle * size[0] + [0, 0, z] for z in zs])
        return np.concatenate((p, side)), np.concatenate((ns, np.tile(circle, (len(zs), 1))))
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        zs = np.linspace(-size[1], size[1], max(3, int(np.ceil(count / rings))))
        p = [circle * size[0] + [0, 0, z] for z in zs]
        ns = [circle.copy() for z in zs]
        for sign in [-1, 1]:
            for radius in [0., size[0] / np.sqrt(2), size[0]]:
                cap = circle * radius + [0, 0, sign * size[1]]
                if radius == 0: cap = cap[:1]
                p.append(cap); ns.append(np.tile([0., 0., sign], (len(cap), 1)))
        return np.concatenate(p), np.concatenate(ns)
    raise ValueError(f'Unsupported collision primitive: {kind}')
