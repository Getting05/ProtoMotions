"""Bounded single-robot inference and IsaacLab offscreen MP4 capture."""

import json
import logging
import math
from pathlib import Path

log = logging.getLogger(__name__)


def prepare_video_motion(request, directory):
    """Materialize only the selected packed motion on CPU (never the full GPU shard).

    Scene-linked libraries keep their indices because scene assignment uses them.
    """
    import torch

    source = Path(request["motion_file"])
    if source.suffix != ".pt" or request["scene_file"] is not None:
        return
    try:
        data = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:
        data = torch.load(source, map_location="cpu", weights_only=False)
    motion = request["wandb_video_motion_id"]
    if not 0 <= motion < len(data["motion_lengths"]):
        raise ValueError(f"Motion ID {motion} is outside the selected motion library")
    start = int(data["length_starts"][motion])
    end = start + int(data["motion_num_frames"][motion])
    frame_fields = {
        "gts",
        "grs",
        "gvs",
        "gavs",
        "dvs",
        "dps",
        "contacts",
        "lrs",
        "goal_states",
    }
    motion_fields = {
        "motion_lengths",
        "motion_dt",
        "motion_num_frames",
        "motion_weights",
    }
    result = {}
    for key, value in data.items():
        if value is None:
            continue
        if key in frame_fields:
            result[key] = value[start:end].clone()
        elif key in motion_fields:
            result[key] = value[motion : motion + 1].clone()
        elif key == "length_starts":
            result[key] = torch.zeros(1, dtype=value.dtype)
        elif key == "motion_files":
            result[key] = (value[motion],)
        else:
            raise ValueError(f"Unsupported packed motion field: {key}")
    selected = Path(directory) / "selected_motion.pt"
    torch.save(result, selected)
    request["render_motion_file"] = str(selected)
    request["render_motion_id"] = 0


def configure_video_inference(request, simulator, motion_lib, scene_lib, env):
    from protomotions.components.motion_lib import MotionFileSwitchMode

    simulator.num_envs = 1
    simulator.headless = True
    simulator.projectile.num_projectiles = 0
    # Training uses thousands of robots; reserve only single-robot collision buffers.
    for field in (
        "gpu_max_rigid_contact_count",
        "gpu_found_lost_pairs_capacity",
        "gpu_found_lost_aggregate_pairs_capacity",
        "gpu_max_rigid_patch_count",
    ):
        setattr(
            simulator.sim.physx, field, min(getattr(simulator.sim.physx, field), 2**18)
        )
    motion_lib.motion_file = request.get("render_motion_file", request["motion_file"])
    motion_lib.motion_file_shard_indices = None
    motion_lib.motion_file_switch_mode = MotionFileSwitchMode.FIXED
    scene_lib.scene_file = request["scene_file"]
    env.motion_manager.init_start_prob = 1.0
    env.motion_manager.subset_method = [
        request.get("render_motion_id", request["wandb_video_motion_id"])
    ]
    env.motion_manager.exclude_motion_ids = None
    env.motion_manager.exclude_motions_file = None


def blend_reference(base, overlay, opacity):
    """Keep the policy visible through reference geometry, independent of materials."""
    import numpy as np

    return (
        np.rint(
            base.astype(np.float32) * (1 - opacity)
            + overlay.astype(np.float32) * opacity
        )
        .clip(0, 255)
        .astype(np.uint8)
    )


class FollowCamera:
    """A dedicated USD camera / RGB render product; no desktop viewport required."""

    def __init__(self, simulator, width, height):
        import omni.replicator.core as rep
        from pxr import UsdGeom

        self.simulator = simulator
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import (
            ensure_rtx_hydra_engine_attached,
        )

        ensure_rtx_hydra_engine_attached()
        simulator._sim.set_setting("/isaaclab/video/enabled", True)
        self.width, self.height = width, height
        self.camera = UsdGeom.Camera.Define(
            simulator._sim.stage, "/World/PolicyVideoCamera"
        )
        self.camera.CreateFocalLengthAttr(24.0)
        self.camera.CreateHorizontalApertureAttr(24.0)
        self.camera.CreateVerticalApertureAttr(24.0 * height / width)
        self.camera.CreateClippingRangeAttr((0.05, 1000.0))
        self.transform = UsdGeom.Xformable(self.camera).AddTransformOp()
        # Headless inference already omits reference visualization; also hide debug markers.
        for path in ("/Visuals",):
            prim = simulator._sim.stage.GetPrimAtPath(path)
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
        self.product = rep.create.render_product(
            str(self.camera.GetPath()), (width, height)
        )
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach([self.product])

    def _follow(self):
        import numpy as np
        from pxr import Gf

        root = (
            self.simulator._get_simulator_root_state(0)
            .root_pos.detach()
            .cpu()
            .numpy()
            .reshape(-1, 3)[0]
        )
        eye = root + np.array([3.0, -4.0, 1.5])
        target = root + np.array([0.0, 0.0, 0.2])
        view = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*eye.tolist()), Gf.Vec3d(*target.tolist()), Gf.Vec3d(0, 0, 1)
        )
        self.transform.Set(view.GetInverse())

    def frame(self, reference=None):
        from protomotions.utils.reference_video import REFERENCE_OPACITY

        if reference is None:
            return self._frame()
        # Composite two identical simulation instants to guarantee fractional
        # opacity across RTX presets. No physics advances between these passes.
        reference.set_visible(False)
        try:
            self._frame()  # flush visibility to the render product
            base = self._frame().copy()
        finally:
            reference.set_visible(True)
        self._frame()
        overlay = self._frame()
        return blend_reference(base, overlay, REFERENCE_OPACITY)

    def _frame(self):
        import numpy as np

        self._follow()
        self.simulator._sim.render()
        rgb = np.asarray(self.rgb.get_data())
        if rgb.shape[:2] != (self.height, self.width):
            raise RuntimeError(f"Invalid RGB frame: {rgb.shape}")
        return np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)

    def warmup(self):
        # Let render-product shaders and annotators initialize without advancing physics.
        self._follow()
        for _ in range(12):
            self.simulator._sim.render()

    def close(self):
        self.rgb.detach([self.product])
        self.product.destroy()


def record_policy_video(agent, request, checkpoint):
    import imageio.v2 as imageio
    import torch

    from protomotions.utils.reference_video import REFERENCE_OPACITY, ReferenceRobot

    env = agent.env
    motion_id = request.get("render_motion_id", request["wandb_video_motion_id"])
    fps = request["wandb_video_fps"]
    dt = env.dt
    motion_ids = torch.tensor([motion_id], dtype=torch.long, device=agent.device)
    duration = min(
        request["wandb_video_duration"],
        float(env.motion_lib.get_motion_length(motion_ids)[0]),
    )
    if duration <= 0:
        raise ValueError("Selected motion is empty")
    agent.eval()
    agent.fabric.seed_everything(0)
    if agent.evaluator is not None:
        agent.evaluator._disable_perturbations()
    env.motion_manager.motion_ids[:] = motion_id
    env.motion_manager.motion_times[:] = 0
    obs, _ = env.reset(disable_motion_resample=True)
    reference = ReferenceRobot(env)
    camera = FollowCamera(
        env.simulator, request["wandb_video_width"], request["wandb_video_height"]
    )
    output = Path(request["output"])
    temporary = output.with_name("best_policy.partial.mp4")
    frames = 0
    camera_samples = []
    try:
        with torch.inference_mode(), imageio.get_writer(
            str(temporary),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
            ffmpeg_log_level="error",
        ) as writer:
            # Capture by simulation time, so changing FPS never changes playback speed.
            for step in range(1, math.ceil(duration / dt) + 1):
                agent.pre_collect_step(step - 1)
                obs = agent.add_agent_info_to_obs(obs)
                outputs = agent.model(agent.obs_dict_to_tensordict(obs))
                action = (
                    outputs["mean_action"]
                    if "mean_action" in outputs
                    else outputs["action"]
                )
                obs, _, dones, _, _ = env.step(action)
                elapsed = min(step * dt, duration)
                reference.update()
                if frames == 0:
                    # PhysX publishes reset transforms to rendering after its first step.
                    camera.warmup()
                if frames / fps <= elapsed + 1e-8 and frames < math.ceil(
                    duration * fps
                ):
                    frame = camera.frame(reference)
                    while frames / fps <= elapsed + 1e-8 and frames < math.ceil(
                        duration * fps
                    ):
                        writer.append_data(frame)
                        frames += 1
                    if frames % fps == 0:
                        log.info(
                            "Recorded %s frames (%.2f simulated seconds)",
                            frames,
                            elapsed,
                        )
                if len(camera_samples) < 2 or step % max(1, round(1 / dt)) == 0:
                    camera_samples.append(
                        env.simulator._get_simulator_root_state(0)
                        .root_pos.detach()
                        .cpu()
                        .tolist()
                    )
                if bool(dones[0]) or elapsed >= duration:
                    break
        temporary.replace(output)
        log.info("Best-policy video encoded: %s (%s frames)", output, frames)
        # Read only after simulation work to keep checkpoint metadata on CPU.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = {
            "trigger_epoch": request["trigger_epoch"],
            "best_epoch": state.get("epoch", 0),
            "best_score": state.get("best_evaluated_score"),
            "motion_id": request["wandb_video_motion_id"],
            "motion_file": request["motion_file"],
            "frames": frames,
            "fps": fps,
            "duration": frames / fps,
            "width": request["wandb_video_width"],
            "height": request["wandb_video_height"],
            "robot_positions": camera_samples,
            "reference_overlay": True,
            "reference_color": "blue",
            "reference_opacity": REFERENCE_OPACITY,
        }
        output.with_suffix(".json").write_text(json.dumps(metadata))
    finally:
        reference.close()
        camera.close()
        temporary.unlink(missing_ok=True)
