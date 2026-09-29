#!/usr/bin/env python3
"""Full SMPL/SMPL-H pose files -> calibrated Astro P2 trajectories.

Uses the neutral skeleton exported for calibration (no SMPL/torch/chumpy
runtime required), calibration_viewer's exact body-frame bone transform,
and the existing single-stage PyRoki optimizer. See README_ASTRO_P2_BONE.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
# Support the viewer both inside the repository and in its standalone workspace.
_viewer_candidates = [
    os.environ.get("CALIBRATION_VIEWER_SRC"),
    str(REPO_ROOT / "tools/calibration_viewer/src"),
    "/data/chenguanting/workspace/calibration-viewer-soma-test/src",
]
for _candidate in _viewer_candidates:
    if _candidate and (Path(_candidate) / "calibration_viewer").is_dir():
        sys.path.insert(0, _candidate)
        break
import calibration_viewer.calibration as _calibration_module
CALIBRATION_PACKAGE_ROOT = Path(_calibration_module.__file__).resolve().parent
from calibration_viewer.calibration import CalibrationParams
from calibration_viewer.human import SMPL_JOINT_NAMES, SMPL_EDGES, _axis_vector
from calibration_viewer.motion import HumanMotion, RootTrajectoryParams

VERSION = "astro-p2-bone-ground-hands-v4"
DEFAULT_CALIBRATION = "/data/chenguanting/workspace/calibration-viewer-run/astro_p2_calibration.yaml"
DEFAULT_REST = "/data/chenguanting/workspace/calibration-viewer-run/smpl_neutral_viewer.pkl"
KEYPOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_foot", "right_foot",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
)
WEIGHTS = dict(local_alignment=1., global_alignment=4., root_smoothness=1.,
               joint_smoothness=4., self_collision=20., joint_rest_penalty=1.,
               joint_vel_limit=50., foot_contact=30., foot_tilt=1.)


def read_mapping(path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            result = dict(data)
    else:
        # Only load trusted local pickle/npy files.
        if path.suffix.lower() == ".npy":
            result = np.load(path, allow_pickle=True).item()
        else:
            with path.open("rb") as stream:
                result = pickle.load(stream, encoding="latin1")
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected a dictionary")
    return result


def basis_for(axes):
    basis = np.stack([_axis_vector(x) for x in axes])
    if basis.shape != (3, 3) or not np.isclose(np.linalg.det(basis), 1.):
        raise ValueError("Axes must be a right-handed signed permutation of x,y,z")
    return basis


def load_calibration(path):
    with open(path, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict) or cfg.get("format_version") != 2:
        raise ValueError("Require a format_version: 2 calibration YAML")
    params = CalibrationParams.from_mapping(cfg.get("calibration"))
    if params.mode != "bone":
        raise ValueError("calibration.mode must be bone")
    root = RootTrajectoryParams.from_mapping(cfg.get("root_trajectory"))
    basis = basis_for(cfg.get("human", {}).get("axes", ["z", "x", "y"]))
    links = cfg.get("robot", {}).get("keypoint_links", {})
    if set(KEYPOINT_NAMES) - links.keys():
        raise ValueError("YAML must map all 15 robot keypoints")
    if cfg["robot"].get("base_link") != "pelvis":
        raise ValueError("The Astro P2 solver requires robot.base_link: pelvis")
    return cfg, params, root, basis


def load_rest(path):
    data = read_mapping(path)
    names = tuple(data.get("joint_names", SMPL_JOINT_NAMES))
    joints = np.asarray(data["joints"], dtype=float)
    if joints.shape != (len(names), 3) or not np.isfinite(joints).all():
        raise ValueError("Rest skeleton requires finite joints [24,3]")
    if len(set(names)) != len(names) or set(SMPL_JOINT_NAMES) - set(names):
        raise ValueError("Rest skeleton requires all 24 named SMPL joints")
    return joints[[names.index(n) for n in SMPL_JOINT_NAMES]]


def finite_array(value, label):
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains nonfinite values")
    return result


def forward_smpl(poses, trans, rest):
    """Exact joint FK on the calibrated neutral SMPL skeleton; no mesh LBS."""
    poses = finite_array(poses, "poses")
    if poses.ndim != 2 or poses.shape[1] not in (66, 72, 156):
        raise ValueError("poses must be [T,66], [T,72] SMPL, or [T,156] SMPL-H; SMPL-X is unsupported")
    trans = finite_array(trans, "trans")
    if trans.shape != (len(poses), 3) or len(poses) == 0:
        raise ValueError("trans must be nonempty [T,3], matching poses")
    local_aa = np.zeros((len(poses), 24, 3))
    count = 24 if poses.shape[1] == 72 else 22
    local_aa[:, :count] = poses[:, :3*count].reshape(-1, count, 3)
    local_R = Rotation.from_rotvec(local_aa.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
    rotations = np.empty_like(local_R)
    points = np.empty((len(poses), 24, 3))
    parents = {child: parent for parent, child in SMPL_EDGES}
    for i, name in enumerate(SMPL_JOINT_NAMES):
        if i == 0:
            rotations[:, i] = local_R[:, i]
            points[:, i] = trans + rest[i]
        else:
            parent = SMPL_JOINT_NAMES.index(parents[name])
            rotations[:, i] = rotations[:, parent] @ local_R[:, i]
            points[:, i] = points[:, parent] + np.einsum("tij,j->ti", rotations[:, parent], rest[i]-rest[parent])
    return points, rotations


def smooth_contacts(values):
    return np.asarray([values[max(0, i-2):min(len(values), i+3)].mean(axis=0)
                       for i in range(len(values))])


def prepare_motion(path, args, rest, params, root, canonical_basis):
    data = read_mapping(path)
    if "poses" in data:
        poses = finite_array(data["poses"], "poses")
    elif "body_pose" in data and "global_orient" in data:
        body = finite_array(data["body_pose"], "body_pose")
        orient = finite_array(data["global_orient"], "global_orient")
        if body.ndim != 2 or body.shape[1] not in (63, 69):
            raise ValueError("body_pose must be [T,63] or [T,69]")
        poses = np.concatenate((orient.reshape(-1, 3), body), axis=1)
    else:
        raise ValueError("Input requires full SMPL poses or global_orient + body_pose")
    if poses.ndim != 2 or not len(poses):
        raise ValueError("poses must be nonempty [T,D]")
    trans = data.get("trans", data.get("transl"))
    if trans is None:
        raise ValueError("Input is missing trans/transl [T,3]")
    trans = finite_array(trans, "trans")
    if trans.shape != (len(poses), 3):
        raise ValueError("trans and poses lengths differ")
    fps = args.input_fps
    if fps is None:
        fps = next((float(np.asarray(data[k]).item()) for k in
                    ("mocap_framerate", "mocap_frame_rate", "fps") if k in data), None)
    if fps is None or not np.isfinite(fps) or fps <= 0:
        raise ValueError("A positive source FPS is required; use --input-fps if absent")
    stop = None if args.max_frames is None else args.start_frame + args.max_frames
    frame_ids = np.arange(len(poses))[args.start_frame:stop:args.subsample_factor]
    if len(frame_ids) < 2:
        raise ValueError("At least two selected frames are required")
    positions, rotations = forward_smpl(poses[frame_ids], trans[frame_ids], rest)
    # AMASS world coordinates are already Z-up even though the SMPL neutral
    # template and the calibration source use Y-up. Keep those two bases separate.
    world_axes = args.input_axes
    if world_axes == "auto":
        if "mocap_framerate" in data or "mocap_frame_rate" in data:
            world_basis = np.eye(3)
            world_axes = "x,y,z (AMASS metadata)"
        else:
            world_basis = canonical_basis
            world_axes = "calibration human.axes"
    elif world_axes == "calibration":
        world_basis = canonical_basis
    else:
        world_basis = basis_for(world_axes.split(","))
    positions = positions @ world_basis.T
    rotations = world_basis @ rotations @ canonical_basis.T
    source = HumanMotion(SMPL_JOINT_NAMES, positions, fps/args.subsample_factor,
                         root_rotations=rotations[:, 0])
    _, calibrated = source.calibrated(params, root)
    indices = [SMPL_JOINT_NAMES.index(n) for n in KEYPOINT_NAMES]
    targets = np.empty((len(frame_ids), 18, 3))
    target_R = np.empty((len(frame_ids), 18, 3, 3))
    targets[:, :15] = calibrated[:, indices]
    target_R[:, :15] = rotations[:, indices]
    for side, wi, ai in (("left", 13, 15), ("right", 14, 16)):
        hand = SMPL_JOINT_NAMES.index(f"{side}_hand")
        direction = calibrated[:, hand] - targets[:, wi]
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError("Degenerate wrist-to-hand direction in rest skeleton")
        targets[:, ai] = targets[:, wi] + .11 * direction / norm
        target_R[:, ai] = target_R[:, wi]
    # Match the robot pelvis-forward cue at the same anatomical anchor.
    targets[:, 17] = targets[:, 0] + np.einsum("tij,j->ti", rotations[:, 0], [.18, 0., 0.])
    target_R[:, 17] = rotations[:, 0]
    foot_ids = [SMPL_JOINT_NAMES.index(n) for n in
                ("left_ankle", "left_foot", "right_ankle", "right_foot")]
    # Optional scene-origin alignment. Preserve an explicit root Z offset.
    ground_alignment = 0.
    if args.align_ground:
        ground_alignment = -float(np.percentile(calibrated[:, foot_ids, 2], 2)) + float(root.offset[2])
        targets[..., 2] += ground_alignment
    feet = positions[:, foot_ids]
    ground = float(np.percentile(feet[..., 2], 2)) if args.ground_height is None else args.ground_height
    speed = np.linalg.norm(np.gradient(feet, axis=0)*source.fps, axis=-1)
    inferred = (feet[..., 2] < ground+args.contact_height) & (speed < args.contact_speed)
    contacts = []
    for side, sl in (("left", slice(0, 2)), ("right", slice(2, 4))):
        key = f"{side}_foot_contacts"
        if key in data:
            c = finite_array(data[key], key)
            if c.shape not in ((len(poses),), (len(poses), 1), (len(poses), 2)) or np.any((c < 0) | (c > 1)):
                raise ValueError(f"{key} must be [T], [T,1] or [T,2] values in [0,1]")
            c = c[frame_ids].reshape(len(frame_ids), -1).mean(axis=1, keepdims=True)
        else:
            c = inferred[:, sl].mean(axis=1, keepdims=True)
        contacts.append(smooth_contacts(c))
    metadata = dict(source_path=str(Path(path).resolve()), source_frames=frame_ids.tolist(),
                    source_fps=fps, fps=source.fps, input_axes=world_axes, source_pose_dimensions=int(poses.shape[1]),
                    shape_policy="calibration neutral rest; source betas/DMPL/fingers ignored",
                    ground_alignment_m=ground_alignment, contact_source={s: "provided" if f"{s}_foot_contacts" in data else "height+speed heuristic"
                                    for s in ("left", "right")}, estimated_ground_height_m=ground)
    return targets, target_R, contacts[0], contacts[1], metadata


def initialize_solver(args, cfg, keypoint_names=KEYPOINT_NAMES):
    # JAX debug/solver callbacks require a local CPU backend even on CUDA.
    os.environ["JAX_PLATFORMS"] = "cuda,cpu" if args.device == "cuda" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import batch_retarget_to_astro_p2_from_keypoints as base
    urdf = base.yourdfpy.URDF.load(args.urdf_path, mesh_dir=args.mesh_dir)
    robot = base.pk.Robot.from_urdf(urdf)
    collision = base.build_p2_collision_model(urdf)
    base.ASTRO_P2_LINK_NAMES = list(robot.links.names)
    base.human_retarget_names = list(keypoint_names)
    ids = base.jnp.array([base.ASTRO_P2_LINK_NAMES.index(cfg["robot"]["keypoint_links"][n]) for n in keypoint_names])
    base.astro_p2_joint_retarget_indices = ids
    mask = np.zeros((len(keypoint_names), len(keypoint_names)))
    for a, b, w in base.direct_pairs:
        i, j = keypoint_names.index(a), keypoint_names.index(b)
        mask[i, j] = mask[j, i] = w
    if "neck" in keypoint_names:
        neck = keypoint_names.index("neck")
        for name in ("pelvis", "left_shoulder", "right_shoulder"):
            i = keypoint_names.index(name)
            mask[neck, i] = mask[i, neck] = 1.0
    # Convex hull extrema exactly preserve the lowest mesh point for any pose.
    import trimesh
    from scipy.spatial import ConvexHull
    hulls = []
    foot_links = ["left_ankle_roll_link", "right_ankle_roll_link"]
    for name in foot_links:
        vertices = []
        for visual in urdf.link_map[name].visuals:
            mesh = visual.geometry.mesh
            path = Path(args.urdf_path).parent / mesh.filename
            obj = trimesh.load(path, force="mesh")
            v = np.asarray(obj.vertices) * (1.0 if mesh.scale is None else mesh.scale)
            v = v @ visual.origin[:3, :3].T + visual.origin[:3, 3]
            vertices.append(v)
        v = np.concatenate(vertices)
        hulls.append(v[ConvexHull(v).vertices])
    size = max(map(len, hulls))
    base.bone_solver_options = dict(
        sole_points=np.stack([np.pad(v, ((0, size-len(v)), (0, 0)), mode="edge") for v in hulls]),
        sole_link_indices=np.array([list(robot.links.names).index(n) for n in foot_links]))
    print(f"JAX devices: {base.jax.devices()}", flush=True)
    return base, robot, collision, ids, base.jnp.asarray(mask)


def solve_motion(solver, targets, rotations, left, right, fps, args):
    base, robot, collision, ids, mask = solver
    print(f"Solving full motion: {len(targets)} frames (no chunking or blending)", flush=True)
    options = dict(getattr(base, "bone_solver_options", {}))
    if options:
        names = list(robot.joints.actuated_names)
        initial = np.zeros((len(targets), len(names)))
        for side in ("left", "right"):
            shoulder, elbow, wrist = [list(base.human_retarget_names).index(side + "_" + n) for n in ("shoulder", "elbow", "wrist")]
            upper = np.einsum("tji,tj->ti", rotations[:, 0], targets[:, elbow]-targets[:, shoulder])
            fore = np.einsum("tji,tj->ti", rotations[:, 0], targets[:, wrist]-targets[:, elbow])
            initial[:, names.index(side+"_shoulder_pitch_joint")] = np.arctan2(-upper[:, 0], -upper[:, 2])
            initial[:, names.index(side+"_shoulder_roll_joint")] = np.arctan2(upper[:, 1], np.linalg.norm(upper[:, [0, 2]], axis=-1))
            pitch = initial[:, names.index(side+"_shoulder_pitch_joint")]
            roll = initial[:, names.index(side+"_shoulder_roll_joint")]
            shoulder_r = Rotation.from_euler("Y", pitch[:, None]).as_matrix() @ Rotation.from_euler("X", roll[:, None]).as_matrix()
            fore_shoulder = np.einsum("tji,tj->ti", shoulder_r, fore)
            initial[:, names.index(side+"_shoulder_yaw_joint")] = np.arctan2(fore_shoulder[:, 1], fore_shoulder[:, 0])
            initial[:, names.index(side+"_elbow_joint")] = np.arctan2(-fore_shoulder[:, 2], np.linalg.norm(fore_shoulder[:, :2], axis=-1))
        options["initial_joints"] = np.clip(initial, np.asarray(robot.joints.lower_limits), np.asarray(robot.joints.upper_limits))
        options["ground_z"] = 0.0
    T, joints = base.solve_retargeting(
        robot=robot, robot_coll=collision, target_keypoints=targets,
        target_orientations=rotations, left_foot_contact=left, right_foot_contact=right,
        astro_p2_joint_retarget_indices=ids, astro_p2_retarget_mask=mask,
        weights={**WEIGHTS, "self_collision": 200.0}, subsample_factor=1, input_fps=fps, pelvis_forward_aux=True, **options)
    se3 = np.asarray(T.wxyz_xyz)
    result = (se3[:, 4:], se3[:, :4], np.asarray(joints))
    if len(result[0]) != len(targets) or not all(np.isfinite(x).all() for x in result):
        raise RuntimeError("Solver returned invalid or incomplete motion")
    p, quat, q = result
    # Check the complete trajectory returned by the optimizer.
    distances = np.asarray(collision.compute_self_collision_distance(robot, q))
    violation = np.maximum(np.asarray(robot.joints.lower_limits)-q, q-np.asarray(robot.joints.upper_limits))
    report = dict(frames=len(q), collision_geometry="PyRoki URDF capsules",
                  min_distance_m=float(distances.min()),
                  penetrating_frames_1mm=int(np.any(distances < -.001, axis=-1).sum()),
                  max_joint_limit_violation_rad=float(np.maximum(violation, 0).max()),
                  max_quaternion_norm_error=float(np.abs(np.linalg.norm(quat, axis=1)-1).max()),
                  solve_mode="full_sequence", solve_frames=len(targets), forward_aux_frame="pelvis")
    if options:
        fk = np.asarray(robot.forward_kinematics(q))
        local = fk[:, options["sole_link_indices"]]
        root_r = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
        foot_r = Rotation.from_quat(local[:, :, [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(len(q), 2, 3, 3)
        points = np.einsum("tfij,fvj->tfvi", foot_r, options["sole_points"]) + local[:, :, None, 4:]
        world = np.einsum("tij,tfvj->tfvi", root_r, points) + p[:, None, None, :]
        height = world[..., 2].min(axis=(1, 2))
        wrist_ids = [list(base.human_retarget_names).index(n) for n in ("left_wrist", "right_wrist")]
        robot_local = fk[:, np.asarray(ids)[wrist_ids], 4:]
        target_local = np.einsum("tji,tkj->tki", rotations[:, 0], targets[:, wrist_ids]-targets[:, :1])
        error = np.linalg.norm(robot_local-target_local, axis=-1)
        report.update(ground_geometry="URDF visual foot mesh convex hull", min_sole_height_m=float(height.min()),
                      ground_penetrating_frames_1mm=int((height < -.001).sum()),
                      wrist_pelvis_rmse_m=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                      wrist_front_when_target_behind_frames=((target_local[:,:,0]<-.03)&(robot_local[:,:,0]>.03)).sum(axis=0).tolist())
        report["quality_pass"] = bool(height.min() >= -.001 and np.max(np.sqrt(np.mean(error**2, axis=0))) < .08 and distances.min() >= -.001)
    return result, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="SMPL/SMPL-H .npz/.pkl/.npy file or recursive directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--calibration-config", type=Path, default=Path(DEFAULT_CALIBRATION))
    parser.add_argument("--smpl-rest-pose", type=Path, default=Path(DEFAULT_REST), help="The 24-joint neutral export used for calibration")
    parser.add_argument("--input-axes", default="auto", help="auto: AMASS metadata -> x,y,z; otherwise YAML human.axes. Or calibration / x,y,z / z,x,y")
    parser.add_argument("--input-fps", type=float)
    parser.add_argument("--subsample-factor", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, help="Optional raw frame limit for an explicit smoke test; default processes the full motion")
    parser.add_argument("--align-ground", action="store_true", help="Shift calibrated feet to scene Z=0, preserving root_trajectory.offset Z")
    parser.add_argument("--ground-height", type=float, help="Source world ground height in meters; otherwise estimate from feet")
    parser.add_argument("--contact-height", type=float, default=.08)
    parser.add_argument("--contact-speed", type=float, default=.2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Save calibrated targets without IK")
    parser.add_argument("--save-targets", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--urdf-path", default=str(REPO_ROOT/"protomotions/data/assets/astro_p2/urdf/astro_p2_retarget.urdf"))
    parser.add_argument("--mesh-dir", default=str(REPO_ROOT/"protomotions/data/assets/astro_p2/meshes"))
    args = parser.parse_args()
    if args.subsample_factor < 1 or args.start_frame < 0 or (args.max_frames is not None and args.max_frames < 2):
        parser.error("Invalid frame selection")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require 0 <= shard-index < num-shards")
    if not np.isfinite(args.contact_height) or not np.isfinite(args.contact_speed) or min(args.contact_height, args.contact_speed) <= 0:
        parser.error("Contact thresholds must be positive and finite")
    if args.ground_height is not None and not np.isfinite(args.ground_height):
        parser.error("ground-height must be finite")
    cfg, params, root, basis = load_calibration(args.calibration_config)
    rest = load_rest(args.smpl_rest_pose)
    if args.input.is_file():
        paths = [args.input]
        input_root = args.input.parent
    elif args.input.is_dir():
        input_root = args.input
        paths = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in (".npz", ".pkl", ".npy")
                       and p.name not in ("shape.npz",) and args.output_dir.resolve() not in p.resolve().parents)
    else:
        parser.error(f"Input does not exist: {args.input}")
    paths = paths[args.shard_index::args.num_shards]
    if not paths:
        parser.error("No motion files selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for path in (Path(__file__), args.calibration_config, args.smpl_rest_pose,
                 Path(__file__).with_name("batch_retarget_to_astro_p2_from_keypoints.py"),
                 CALIBRATION_PACKAGE_ROOT/"calibration.py",
                 CALIBRATION_PACKAGE_ROOT/"motion.py", Path(args.urdf_path)):
        digest.update(path.read_bytes())
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
               if k not in ("skip_existing", "num_shards", "shard_index", "output_dir", "input")}
    digest.update(json.dumps(options, sort_keys=True).encode())
    solver = None
    failures = []
    print(f"Bone calibration: {args.calibration_config}; {len(paths)} motions", flush=True)
    print("Using calibrated neutral SMPL shape; ignoring source betas, DMPL and finger articulation.", flush=True)
    for path in paths:
        relative = path.relative_to(input_root)
        output = args.output_dir/relative.parent/(relative.name+"_retargeted.npz")
        target_path = output.with_name(relative.name+"_targets.npz")
        output.parent.mkdir(parents=True, exist_ok=True)
        stamp = f"{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
        signature = hashlib.sha256(digest.digest()+stamp.encode()).hexdigest()
        check = target_path if args.prepare_only else output
        try:
            if args.skip_existing and check.exists():
                with np.load(check, allow_pickle=False) as previous:
                    if "signature" in previous and str(previous["signature"].item()) == signature:
                        print(f"Skipping current: {path}", flush=True)
                        continue
            targets, rotations, left, right, metadata = prepare_motion(path, args, rest, params, root, basis)
            meta_json = json.dumps(dict(metadata, calibration=cfg, options=options), ensure_ascii=False)
            if args.prepare_only or args.save_targets:
                np.savez_compressed(target_path, positions=targets, orientations=rotations,
                                    left_foot_contacts=left, right_foot_contacts=right,
                                    fps=metadata["fps"], metadata_json=meta_json, signature=signature)
            if args.prepare_only:
                print(f"Prepared {len(targets)} frames: {target_path}", flush=True)
                continue
            if solver is None:
                solver = initialize_solver(args, cfg)
            result, report = solve_motion(solver, targets, rotations, left, right, metadata["fps"], args)
            p, quat, q = result
            payload = dict(retarget_version=VERSION, signature=signature, base_frame_pos=p,
                           base_frame_wxyz=quat, joint_angles=q, joint_names=np.asarray(solver[1].joints.actuated_names),
                           fps=metadata["fps"], source_fps=metadata["source_fps"],
                           subsample_factor=args.subsample_factor, source_frames=np.asarray(metadata["source_frames"]),
                           left_foot_contacts=left, right_foot_contacts=right, metadata_json=meta_json)
            temp = output.with_suffix(".tmp.npz")
            np.savez_compressed(temp, **payload)
            os.replace(temp, output)
            output.with_suffix(".report.json").write_text(json.dumps(report, indent=2)+"\n")
            print(f"Saved {output}: {json.dumps(report)}", flush=True)
        except Exception as error:
            import traceback
            traceback.print_exc()
            failures.append(dict(path=str(path), error=str(error)))
    manifest = args.output_dir/f"run_shard_{args.shard_index}.json"
    manifest.write_text(json.dumps(dict(version=VERSION, selected=len(paths), failures=failures, options=options), indent=2)+"\n")
    if failures:
        raise SystemExit(f"{len(failures)} motion(s) failed; see {manifest}")


if __name__ == "__main__":
    main()
