from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, save_config
from .human import load_smpl_pkl
from .robot import RobotModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive SMPL-to-URDF morphology calibration")
    parser.add_argument("--smpl", required=True, type=Path, help="SMPL pkl: joints, model, or pose parameters")
    parser.add_argument("--urdf", required=True, type=Path, help="Robot URDF")
    parser.add_argument("--config", required=True, type=Path, help="Robot mapping YAML")
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        help="Optional URDF mesh directory; overrides robot.mesh_dir in the config",
    )
    parser.add_argument("--smpl-model", type=Path, help="SMPL model directory/file for pose-parameter pkl")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--gender", choices=("neutral", "male", "female"), default="neutral")
    parser.add_argument("--output", type=Path, default=Path("calibration.yaml"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--hide-urdf-mesh", action="store_true", help="Start with the URDF mesh hidden")
    parser.add_argument("--hide-smpl-mesh", action="store_true", help="Start with the SMPL mesh hidden")
    parser.add_argument("--mode", choices=("legacy", "bone"), help="Override calibration mode")
    parser.add_argument("--asymmetric", action="store_true", help="Fit independent left/right bone lengths")
    parser.add_argument("--fit-only", action="store_true", help="Fit once and write YAML without starting Viser")
    parser.add_argument("--fit-motion", action="store_true", help="Fit morphology to synchronized robot q_t")
    parser.add_argument("--robot-motion", type=Path, help="Reference NPZ with joint_names, joint_positions, fps and optional root_positions")
    parser.add_argument("--retarget", action="store_true", help="Run PyRoki IK on a complete motion")
    parser.add_argument("--motion-output", type=Path, default=Path("robot_motion.npz"))
    parser.add_argument("--preview-motion", action="store_true", help="Play the retargeted result in Viser")
    parser.add_argument("--fps", type=float, help="Source frame rate override")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--root-scale", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--root-offset", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--ik-position-weight", type=float, default=50.)
    parser.add_argument("--ik-rest-weight", type=float, default=.01)
    parser.add_argument("--ik-smoothness-weight", type=float, default=.1)
    parser.add_argument("--ik-max-iterations", type=int, default=100)
    parser.add_argument("--ik-error-threshold", type=float, default=.05, help="Per-frame positional RMSE warning threshold in meters")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.fit_only and (args.fit_motion or args.retarget):
        parser.error("--fit-only cannot be combined with --fit-motion/--retarget")
    if args.fit_motion and args.robot_motion is None:
        parser.error("--fit-motion requires synchronized --robot-motion")
    if args.preview_motion and not args.retarget:
        parser.error("--preview-motion requires --retarget")
    if args.retarget and args.motion_output.suffix.lower() != ".npz":
        parser.error("--motion-output must end in .npz")
    cfg = load_config(args.config)
    from .calibration import CalibrationParams
    params = CalibrationParams.from_mapping(cfg.get("calibration"))
    if args.mode:
        params.mode = args.mode
    if args.asymmetric:
        params.asymmetric = True
    cfg["calibration"] = params.serializable()
    human_cfg = cfg.get("human", {})
    axes = human_cfg.get("axes", ["z", "x", "y"])
    robot_cfg = cfg["robot"]
    mesh_dir = args.mesh_dir or robot_cfg.get("mesh_dir")
    if mesh_dir is not None:
        mesh_dir = Path(mesh_dir)
        if not mesh_dir.is_absolute():
            mesh_dir = args.urdf.resolve().parent / mesh_dir
    robot = RobotModel.load(
        args.urdf,
        base_link=robot_cfg["base_link"],
        keypoint_links=robot_cfg["keypoint_links"],
        mesh_dir=mesh_dir,
    )
    robot.update(robot_cfg.get("t_pose_joint_positions", {}))
    from .motion import RootTrajectoryParams
    root_data = dict(cfg.get("root_trajectory", {}))
    if args.root_scale is not None:
        root_data["scale"] = args.root_scale
    if args.root_offset is not None:
        root_data["offset"] = args.root_offset
    root = RootTrajectoryParams.from_mapping(root_data)
    cfg["root_trajectory"] = root.serializable()
    if args.fit_motion or args.retarget:
        _run_motion(args, cfg, robot, params, root)
        return
    human = load_smpl_pkl(args.smpl, frame=args.frame, model_path=args.smpl_model, gender=args.gender).transformed(axes)
    if args.fit_only:
        from .calibration import fit_calibration
        params, info = fit_calibration(human.by_name, robot.keypoints(), list(robot_cfg["keypoint_links"]), initial=params)
        save_config(args.output, {"format_version": 2, "human": human_cfg, "robot": robot_cfg,
                                  "calibration": params.serializable(), "root_trajectory": root.serializable(), "diagnostics": info})
        print(f"Saved {args.output} (RMSE {info['rmse_m'] * 100:.2f} cm, rank {info['rank']}/{info['parameter_count']})")
        return
    from .viewer import CalibrationViewer
    CalibrationViewer(
        human,
        robot,
        cfg,
        output=args.output,
        host=args.host,
        port=args.port,
        show_robot_mesh=not args.hide_urdf_mesh,
        show_smpl_mesh=not args.hide_smpl_mesh,
    ).run()


def _run_motion(args, cfg, robot, params, root):
    from .calibration import fit_calibration_frames
    from .motion import load_human_motion, reference_robot_frames, fit_root_trajectory
    motion = load_human_motion(args.smpl, axes=cfg.get("human", {}).get("axes", ["z", "x", "y"]),
                               fps=args.fps, model_path=args.smpl_model, gender=args.gender,
                               start=args.start, stop=args.stop, stride=args.stride)
    diagnostics = {}
    if args.fit_motion:
        reference, roots = reference_robot_frames(args.robot_motion, robot, motion,
                                                   source_fps=motion.fps * args.stride)
        params, diagnostics["morphology"] = fit_calibration_frames(
            motion.frames(), reference, list(robot.keypoint_links), initial=params)
        if roots is not None:
            root, diagnostics["root_trajectory"] = fit_root_trajectory(motion.roots, roots, root)
        else:
            diagnostics["root_trajectory"] = {"warning": "Reference has no root_positions; root parameters were preserved."}
        saved = dict(cfg, format_version=2, calibration=params.serializable(),
                     root_trajectory=root.serializable(), diagnostics=diagnostics)
        save_config(args.output, saved)
        info = diagnostics["morphology"]
        print(f"Saved {args.output}: {info['frame_count']} frames, RMSE {info['rmse_m']:.6f} m, rank {info['rank']}/{info['parameter_count']}")
        if info["warning"]:
            print(info["warning"])
        if diagnostics["root_trajectory"].get("warning"):
            print(diagnostics["root_trajectory"]["warning"])
    if args.retarget:
        from .retarget import retarget_motion, save_motion
        output, info = retarget_motion(
            motion, robot, params, root, initial_joints=cfg["robot"].get("t_pose_joint_positions"),
            position_weight=args.ik_position_weight, rest_weight=args.ik_rest_weight,
            smoothness_weight=args.ik_smoothness_weight, max_iterations=args.ik_max_iterations,
            error_threshold=args.ik_error_threshold)
        save_motion(args.motion_output, output)
        print(f"Saved {args.motion_output}: {info['frame_count']} frames, positional RMSE {info['position_rmse_m']:.6f} m; "
              f"{info['frames_above_threshold']} frames above {info['error_threshold_m']:.3f} m")
        if info["frames_without_termination_criterion"]:
            print(f"Warning: {info['frames_without_termination_criterion']} frames stopped without a solver termination criterion; inspect diagnostics.")
        if args.preview_motion:
            from .motion_viewer import preview_motion
            preview_motion(robot, output, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
