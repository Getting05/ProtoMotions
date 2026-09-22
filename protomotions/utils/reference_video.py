"""Visual-only reference robot for policy videos, driven by motion-library poses."""

REFERENCE_COLOR = (0.03, 0.25, 1.0)
REFERENCE_OPACITY = 0.4


def reference_pose(env):
    """Use the same time, spawn alignment and terrain correction as mimic targets."""
    state = env.motion_lib.get_motion_state(
        env.motion_manager.motion_ids, env.motion_manager.motion_times
    )
    positions = state.rigid_body_pos.clone()
    positions += env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
        positions
    )
    return positions[0].detach().cpu(), state.rigid_body_rot[0].detach().cpu()


class ReferenceRobot:
    """Copy renderable shapes, never articulations, collisions or physics APIs.

    Each visual link is driven directly by the reference's global body pose. This
    works for P2 and other robots without a second simulation or a second policy.
    """

    def __init__(self, env):
        simulator = env.simulator
        self.env = env
        self.stage = simulator._sim.stage
        self.path = "/World/PolicyVideoReference"
        self.body_indices = simulator.data_conversion.body_convert_to_sim.tolist()
        self._build(
            self.stage,
            simulator._robot.root_view.link_paths[0],
            self.body_indices,
        )

    def _build(self, stage, link_paths, body_indices):
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        root = UsdGeom.Xform.Define(stage, self.path)
        root.SetResetXformStack(True)
        material = UsdShade.Material.Define(stage, self.path + "/BlueReference")
        shader = UsdShade.Shader.Define(stage, self.path + "/BlueReference/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*REFERENCE_COLOR)
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        cache = UsdGeom.XformCache()
        link_paths = [Sdf.Path(str(path)) for path in link_paths]
        # Snapshot the source traversal before creating any render-only shapes.
        shapes = []
        for link_idx, link_path in enumerate(link_paths):
            link = stage.GetPrimAtPath(link_path)
            if not link.IsValid():
                raise ValueError(f"Reference robot link is missing: {link_path}")
            for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                imageable = UsdGeom.Imageable(prim)
                if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                    continue
                if imageable.ComputePurpose() in (
                    UsdGeom.Tokens.guide,
                    UsdGeom.Tokens.proxy,
                ):
                    continue
                # Nested rigid links belong to their own body, not their ancestor.
                owner = max(
                    (
                        i
                        for i, p in enumerate(link_paths)
                        if prim.GetPath().HasPrefix(p)
                    ),
                    key=lambda i: len(str(link_paths[i])),
                )
                if owner != link_idx:
                    continue
                local = (
                    cache.GetLocalToWorldTransform(prim)
                    * cache.GetLocalToWorldTransform(link).GetInverse()
                )
                shapes.append((link_idx, prim, local))
        if not shapes:
            raise ValueError("Reference robot has no visible geometry")
        self.transforms = []
        self.body_indices = list(body_indices)
        for link_idx in range(len(link_paths)):
            link = UsdGeom.Xform.Define(stage, f"{self.path}/body_{link_idx}")
            self.transforms.append(link.AddTransformOp())
        for shape_idx, (link_idx, source, local) in enumerate(shapes):
            target = stage.DefinePrim(
                f"{self.path}/body_{link_idx}/shape_{shape_idx}", source.GetTypeName()
            )
            for attr in source.GetAttributes():
                name = attr.GetName()
                if name.startswith(
                    ("xformOp", "physics:", "physx", "primvars:display")
                ):
                    continue
                value = attr.Get()
                if value is None:
                    continue
                copied = target.CreateAttribute(
                    name, attr.GetTypeName(), attr.IsCustom()
                )
                copied.Set(value)
                for key in ("interpolation", "elementSize"):
                    if attr.HasMetadata(key):
                        copied.SetMetadata(key, attr.GetMetadata(key))
            UsdGeom.Xformable(target).AddTransformOp().Set(local)
            UsdShade.MaterialBindingAPI.Apply(target).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )
        self.shape_count = len(shapes)

    def update(self):
        self.set_pose(*reference_pose(self.env))

    def set_pose(self, positions, rotations):
        from pxr import Gf

        for transform, body_idx in zip(self.transforms, self.body_indices):
            x, y, z, w = (float(v) for v in rotations[body_idx])
            matrix = Gf.Matrix4d().SetRotate(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
            matrix.SetTranslateOnly(Gf.Vec3d(*(float(v) for v in positions[body_idx])))
            transform.Set(matrix)

    def set_visible(self, visible):
        from pxr import UsdGeom

        root = UsdGeom.Imageable(self.stage.GetPrimAtPath(self.path))
        if visible:
            root.MakeVisible()
        else:
            root.MakeInvisible()

    def close(self):
        self.stage.RemovePrim(self.path)
