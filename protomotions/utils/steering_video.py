# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IsaacLab steering review recorder, called by the regular inference loader."""

from contextlib import ExitStack
from dataclasses import asdict
import json
import math
from pathlib import Path

from protomotions.utils.steering_review import (
    CASES,
    SteeringProtocol,
    command_for_case,
    focus_environment,
    velocity_arrow_points,
)


def configure_steering_video(request, simulator, motion_lib, scene_lib, env):
    """Only specialize the current inference process, never training configs."""
    protocol = SteeringProtocol(**request["protocol"]).validate()
    repeats = len(request["initial_motion_ids"])
    if repeats < 1 or "steering" not in env.control_components:
        raise ValueError("A steering controller and initial motions are required")
    simulator.num_envs = len(CASES) * repeats
    env.motion_manager.subset_method = list(range(repeats))
    env.max_episode_length = 1000000000
    request["num_envs"] = simulator.num_envs
    return protocol


def record_steering_video(agent, request, checkpoint):
    """One rollout, multiple named camera views, aligned telemetry and metrics."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torch
    from pxr import Gf, UsdGeom

    from protomotions.envs.rewards.task import compute_heading_velocity_rew
    from protomotions.utils.policy_video import FollowCamera

    protocol = SteeringProtocol(**request["protocol"]).validate()
    env = agent.env
    repeats = len(request["initial_motion_ids"])
    if env.num_envs != len(CASES) * repeats:
        raise ValueError("Steering cohort size does not match the initial motion count")
    control = env.control_manager.components["steering"]
    output = Path(request["output"]).parent
    output.mkdir(parents=True, exist_ok=True)
    agent.eval()
    agent.fabric.seed_everything(protocol.seed)
    if agent.evaluator is not None:
        agent.evaluator._disable_perturbations()
    original_done_tracks = env.motion_manager.get_done_tracks
    env.motion_manager.get_done_tracks = lambda env_ids=None: torch.zeros(
        env.num_envs if env_ids is None else len(env_ids),
        dtype=torch.bool,
        device=env.device,
    )
    cameras, arrows, video_paths, arrow_samples = {}, {}, {}, []
    stage = env.simulator._sim.stage

    class FocusCamera(FollowCamera):
        env_index = 0

        def _follow(self):
            root = (
                env.simulator.get_robot_state()
                .root_pos[self.env_index]
                .detach()
                .cpu()
                .numpy()
            )
            eye, target = root + [3.0, -4.0, 1.5], root + [0.0, 0.0, 0.2]
            view = Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(*eye.tolist()), Gf.Vec3d(*target.tolist()), Gf.Vec3d(0, 0, 1)
            )
            self.transform.Set(view.GetInverse())

    def make_arrow(path, color):
        arrow = UsdGeom.BasisCurves.Define(stage, path)
        arrow.CreateTypeAttr("linear")
        arrow.CreateCurveVertexCountsAttr([2, 2, 2])
        arrow.CreateWidthsAttr([0.04] * 6)
        arrow.CreateDisplayColorAttr([color])
        return arrow

    def update_arrow(arrow, root, direction, z, speed, scale):
        points = velocity_arrow_points(root[:2], direction, z, speed, scale)
        if not points:
            UsdGeom.Imageable(arrow.GetPrim()).MakeInvisible()
        else:
            arrow.GetPointsAttr().Set([Gf.Vec3f(*point) for point in points])
            UsdGeom.Imageable(arrow.GetPrim()).MakeVisible()

    try:
        env.motion_manager.motion_ids[:] = (
            torch.arange(env.num_envs, device=env.device) % repeats
        )
        env.motion_manager.motion_times[:] = 0
        env.reset(disable_motion_resample=True, sample_flat=True)
        anchor = env.robot_config.anchor_body_index
        q = env.simulator.get_robot_state().rigid_body_rot[:, anchor]
        initial_yaw = torch.atan2(
            2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]),
            1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2),
        ).clone()
        for view in request["views"]:
            focus_environment(
                view, 0, repeats, request["video_motion_index"], protocol.duration
            )
            cameras[view] = FocusCamera(
                env.simulator,
                protocol.width,
                protocol.height,
                camera_path=f"/World/SteeringReview_{view}/Camera",
            )
            arrows[view] = (
                make_arrow(f"/World/SteeringReview_{view}/Movement", (1.0, 0.06, 0.03)),
                make_arrow(f"/World/SteeringReview_{view}/Facing", (0.1, 0.3, 1.0)),
            )
            video_paths[view] = output / f"{view}.mp4"
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                max(12, round(protocol.width / 54)),
            )
        except OSError:
            font = ImageFont.load_default()
        dt = env.dt
        trace = {
            k: []
            for k in [
                "time",
                "speed_command",
                "speed_along",
                "speed_horizontal",
                "lateral_speed",
                "velocity_error",
                "facing_error_deg",
                "pelvis_z",
                "upright",
                "reward",
                "root_xy",
                "command_angle",
            ]
        }
        first_fall = torch.full((env.num_envs,), float("inf"), device=env.device)
        bad_run = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        frames = 0
        target_frames = math.ceil(protocol.duration * protocol.fps)
        print(
            f"STEERING_REVIEW_BEGIN epoch={request['trigger_epoch']} envs={env.num_envs} views={request['views']}",
            flush=True,
        )
        with ExitStack() as stack, torch.inference_mode():
            writers = {}
            for view, path in video_paths.items():
                writers[view] = stack.enter_context(
                    imageio.get_writer(
                        str(path.with_suffix(".partial.mp4")),
                        fps=protocol.fps,
                        codec="libx264",
                        pixelformat="yuv420p",
                        macro_block_size=1,
                        ffmpeg_log_level="error",
                        output_params=["-movflags", "+faststart"],
                    )
                )
            for step in range(math.ceil(protocol.duration / dt)):
                commands = torch.tensor(
                    [command_for_case(case, step * dt, protocol) for case in CASES],
                    device=env.device,
                )
                commands = commands.repeat_interleave(repeats, dim=0)
                theta, facing = (
                    initial_yaw + commands[:, 0],
                    initial_yaw + commands[:, 1],
                )
                speed = commands[:, 2]
                control._tar_dir_theta.copy_(theta)
                control._tar_dir.copy_(torch.stack([theta.cos(), theta.sin()], -1))
                control._tar_face_dir.copy_(
                    torch.stack([facing.cos(), facing.sin()], -1)
                )
                control._tar_speed.copy_(speed)
                control._heading_change_steps[:] = 1000000000
                env._current_context = None
                env.compute_observations(context=env.context)
                before = (
                    env.simulator.get_robot_state().rigid_body_pos[:, anchor].clone()
                )
                agent.pre_collect_step(step)
                obs = agent.add_agent_info_to_obs(env.get_obs())
                outputs = agent.model(agent.obs_dict_to_tensordict(obs))
                action = (
                    outputs["mean_action"]
                    if "mean_action" in outputs
                    else outputs["action"]
                )
                _, _, dones, _, _ = env.step(action)
                if dones.any():
                    raise RuntimeError(
                        "Unexpected episode end; refusing to silently reset a bounded steering trial"
                    )
                state = env.simulator.get_robot_state()
                if not torch.isfinite(state.rigid_body_pos).all():
                    raise RuntimeError("Nonfinite robot state in steering review")
                pos, move = state.rigid_body_pos[:, anchor], control._tar_dir
                velocity = (pos - before) / dt
                along = (velocity[:, :2] * move).sum(-1)
                horizontal = velocity[:, :2].norm(dim=-1)
                lateral = (velocity[:, :2] - along[:, None] * move).norm(dim=-1)
                error = (velocity[:, :2] - speed[:, None] * move).norm(dim=-1)
                aq = state.rigid_body_rot[:, anchor]
                yaw = torch.atan2(
                    2 * (aq[:, 3] * aq[:, 2] + aq[:, 0] * aq[:, 1]),
                    1 - 2 * (aq[:, 1] ** 2 + aq[:, 2] ** 2),
                )
                face_yaw = torch.atan2(
                    control._tar_face_dir[:, 1], control._tar_face_dir[:, 0]
                )
                facing_error = (
                    torch.atan2((yaw - face_yaw).sin(), (yaw - face_yaw).cos()).abs()
                    * 180
                    / math.pi
                )
                z, rq = state.root_pos[:, 2], state.root_rot
                up = 1 - 2 * (rq[:, 0] ** 2 + rq[:, 1] ** 2)
                bad = (z < protocol.min_height) | (up < protocol.min_upright)
                bad_run = torch.where(bad, bad_run + 1, torch.zeros_like(bad_run))
                elapsed = (step + 1) * dt
                first_fall = torch.where(
                    (bad_run >= math.ceil(protocol.fall_hold / dt))
                    & torch.isinf(first_fall),
                    torch.full_like(first_fall, elapsed),
                    first_fall,
                )
                reward = compute_heading_velocity_rew(
                    pos, before, aq, move, speed, control._tar_face_dir, dt
                )
                row = {
                    "time": elapsed,
                    "speed_command": speed,
                    "speed_along": along,
                    "speed_horizontal": horizontal,
                    "lateral_speed": lateral,
                    "velocity_error": error,
                    "facing_error_deg": facing_error,
                    "pelvis_z": z,
                    "upright": up,
                    "reward": reward,
                    "root_xy": state.root_pos[:, :2],
                    "command_angle": theta,
                }
                for key, value in row.items():
                    trace[key].append(
                        value.detach().cpu().numpy().copy()
                        if torch.is_tensor(value)
                        else value
                    )
                if frames / protocol.fps <= elapsed + 1e-8 and frames < target_frames:
                    focuses = {}
                    for view, camera in cameras.items():
                        focus = focus_environment(
                            view,
                            elapsed,
                            repeats,
                            request["video_motion_index"],
                            protocol.duration,
                        )
                        focuses[view] = focus
                        camera.env_index = focus
                        root = pos[focus].detach().cpu().numpy()
                        update_arrow(
                            arrows[view][0],
                            root,
                            move[focus].tolist(),
                            0.12,
                            float(speed[focus]),
                            protocol.arrow_scale,
                        )
                        update_arrow(
                            arrows[view][1],
                            root,
                            control._tar_face_dir[focus].tolist(),
                            float(root[2]) + 0.25,
                            1.0,
                            1.0,
                        )
                    if frames == 0:
                        for camera in cameras.values():
                            camera.warmup()
                    repeats_at_frame = min(
                        target_frames - frames,
                        max(
                            1,
                            math.floor(
                                (elapsed - frames / protocol.fps) * protocol.fps + 1e-7
                            )
                            + 1,
                        ),
                    )
                    for view, camera in cameras.items():
                        focus = focuses[view]
                        im = Image.fromarray(camera.frame())
                        draw = ImageDraw.Draw(im)
                        draw.rectangle((0, 0, protocol.width, 91), fill=(20, 22, 26))
                        draw.text(
                            (10, 5),
                            f"Steering | epoch {request['trigger_epoch']} | {CASES[focus//repeats]} | t={elapsed:.2f}s",
                            font=font,
                            fill="white",
                        )
                        draw.text(
                            (10, 32),
                            f"Red: target {float(speed[focus]):.2f} m/s, arrow {float(speed[focus])*protocol.arrow_scale:.2f} m | actual {float(along[focus]):.2f} m/s",
                            font=font,
                            fill=(255, 190, 175),
                        )
                        draw.text(
                            (10, 59),
                            f"Blue: facing | pelvis {float(z[focus]):.2f} m | upright {float(up[focus]):.2f}",
                            font=font,
                            fill=(170, 205, 255),
                        )
                        for _ in range(repeats_at_frame):
                            writers[view].append_data(np.asarray(im))
                        arrow_samples.append(
                            {
                                "view": view,
                                "time": elapsed,
                                "environment": focus,
                                "speed_command": float(speed[focus]),
                                "red_arrow_length_m": float(speed[focus])
                                * protocol.arrow_scale,
                                "red_arrow_visible": bool(speed[focus] > 0),
                            }
                        )
                    frames += repeats_at_frame
                if step % max(1, round(2 / dt)) == 0:
                    print(
                        f"STEERING_PROGRESS t={elapsed:.2f} frames={frames} upright={int(torch.isinf(first_fall).sum())}/{env.num_envs}",
                        flush=True,
                    )
        for path in video_paths.values():
            path.with_suffix(".partial.mp4").replace(path)
        arrays = {key: np.asarray(value) for key, value in trace.items()}
        np.savez_compressed(output / "telemetry.npz", **arrays)
        fall_times = first_fall.cpu().numpy()
        cases = []
        mask = arrays["time"] >= protocol.steady_start
        for index, case in enumerate(CASES):
            ids = slice(index * repeats, (index + 1) * repeats)
            metrics = {
                key: float(arrays[key][mask, ids].mean())
                for key in [
                    "speed_command",
                    "speed_along",
                    "speed_horizontal",
                    "lateral_speed",
                    "velocity_error",
                    "facing_error_deg",
                    "reward",
                ]
            }
            metrics.update(
                case=case,
                environments=repeats,
                survived=int(np.isinf(fall_times[ids]).sum()),
                first_fall_time=[
                    None if math.isinf(float(v)) else float(v) for v in fall_times[ids]
                ],
                min_pelvis_z=float(arrays["pelvis_z"][:, ids].min()),
                min_upright=float(arrays["upright"][:, ids].min()),
            )
            cases.append(metrics)
        guard = getattr(env.simulator, "_physics_guard", None)
        summary = {
            "epoch": request["trigger_epoch"],
            "checkpoint": str(Path(checkpoint).resolve()),
            "protocol": asdict(protocol),
            "dt": dt,
            "num_envs": env.num_envs,
            "cases": cases,
            "initial_motion_ids": request["initial_motion_ids"],
            "selected_motion_sha256": request["selected_motion_sha256"],
            "prior_sha256": request["prior_sha256"],
            "total_survived": int(np.isinf(fall_times).sum()),
            "robot_done_events": 0,
            "physics_guard_recovered": int(getattr(guard, "total_recovered", 0)),
            "frames": frames,
            "fps": protocol.fps,
            "duration": protocol.duration,
            "views": list(cameras),
            "reference_overlay": False,
            "survival_definition": "pelvis height/upright thresholds held for fall_hold; posture proxy, not contact-labelled falls",
        }
        (output / "metrics.json").write_text(json.dumps(summary, indent=2))
        (output / "arrow_samples.json").write_text(json.dumps(arrow_samples, indent=2))
        print("STEERING_REVIEW_COMPLETE", json.dumps(summary), flush=True)
    finally:
        env.motion_manager.get_done_tracks = original_done_tracks
        for camera in cameras.values():
            camera.close()
        for view in cameras:
            stage.RemovePrim(f"/World/SteeringReview_{view}")
        for path in video_paths.values():
            path.with_suffix(".partial.mp4").unlink(missing_ok=True)
