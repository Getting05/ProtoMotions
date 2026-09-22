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
    parser.add_argument("--fit-only", action="store_true", help="Fit once and write YAML without starting Viser")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    human_cfg = cfg.get("human", {})
    axes = human_cfg.get("axes", ["z", "x", "y"])
    human = load_smpl_pkl(args.smpl, frame=args.frame, model_path=args.smpl_model, gender=args.gender).transformed(axes)
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
    if args.fit_only:
        from .calibration import fit_calibration
        params, info = fit_calibration(human.by_name, robot.keypoints(), list(robot_cfg["keypoint_links"]))
        save_config(args.output, {"format_version": 1, "human": human_cfg, "robot": robot_cfg,
                                  "calibration": params.serializable(), "diagnostics": info})
        print(f"Saved {args.output} (RMSE {info['rmse_m'] * 100:.2f} cm, rank {info['rank']}/8)")
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


if __name__ == "__main__":
    main()
