# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable steering review snapshots, command protocol and arrow geometry.

Simulator, torch and plotting imports stay out of the module-level CLI path.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import time

CASES = (
    "turn_left",
    "turn_right",
    "straight_low",
    "straight_cruise",
    "straight_high",
    "stop",
    "strafe_left",
    "backward",
)
VIEWS = (*CASES, "speed_stop")
FRAME_FIELDS = {
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
MOTION_FIELDS = {"motion_lengths", "motion_dt", "motion_num_frames", "motion_weights"}


@dataclass(frozen=True)
class SteeringProtocol:
    duration: float = 16.0
    switch_time: float = 6.0
    steady_start: float = 8.0
    low_speed: float = 1.2
    cruise_speed: float = 2.0
    high_speed: float = 4.0
    seed: int = 0
    fps: int = 15
    width: int = 960
    height: int = 720
    arrow_scale: float = 0.35
    min_height: float = 0.30
    min_upright: float = 0.50
    fall_hold: float = 0.25

    def validate(self):
        for key, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if not 0 < self.switch_time < self.steady_start < self.duration:
            raise ValueError("Require 0 < switch_time < steady_start < duration")
        if min(self.low_speed, self.cruise_speed, self.high_speed) < 0:
            raise ValueError("Command speeds must be nonnegative")
        if (
            min(self.fps, self.width, self.height, self.arrow_scale, self.fall_hold)
            <= 0
        ):
            raise ValueError(
                "FPS, image size, arrow scale and fall hold must be positive"
            )
        if self.width < 320 or self.height < 240 or self.width % 2 or self.height % 2:
            raise ValueError("Use even image dimensions of at least 320 x 240")
        if self.seed < 0 or self.min_height < 0 or not -1 <= self.min_upright <= 1:
            raise ValueError("Invalid seed or posture thresholds")
        return self


def command_for_case(case, t, protocol):
    """Return movement angle, facing angle (relative to initial yaw), speed."""
    if case not in CASES:
        raise ValueError(f"Unknown steering case {case!r}")
    angle = facing = 0.0
    speed = protocol.cruise_speed
    if case == "straight_low":
        speed = protocol.low_speed
    elif case == "straight_high":
        speed = protocol.high_speed
    elif case == "strafe_left":
        angle, speed = math.pi / 2, protocol.low_speed
    elif case == "backward":
        angle, speed = math.pi, protocol.low_speed
    elif t >= protocol.switch_time:
        if case == "turn_left":
            angle = facing = math.pi / 2
        elif case == "turn_right":
            angle = facing = -math.pi / 2
        elif case == "stop":
            speed = 0.0
    return angle, facing, speed


def focus_environment(view, elapsed, repeats, motion_index, duration):
    if view not in VIEWS or not 0 <= motion_index < repeats:
        raise ValueError("Invalid view or video motion index")
    case = (
        ("straight_high" if elapsed < duration / 2 else "stop")
        if view == "speed_stop"
        else view
    )
    return CASES.index(case) * repeats + motion_index


def velocity_arrow_points(root_xy, direction, z, speed, scale=0.35):
    """Three line segments; shaft length is exactly speed * scale, zero is hidden.

    The arrow head also shrinks with short arrows, so it never extends behind
    the tail. Facing arrows can use this geometry with speed=1, scale=1.
    """
    if not all(math.isfinite(v) for v in [*root_xy, *direction, z, speed, scale]):
        raise ValueError("Arrow arguments must be finite")
    if speed < 0 or scale <= 0:
        raise ValueError("Arrow speed must be nonnegative and scale positive")
    if speed == 0:
        return []
    norm = math.hypot(*direction)
    if norm <= 0:
        raise ValueError("A moving arrow requires a nonzero direction")
    dx, dy = direction[0] / norm, direction[1] / norm
    length = speed * scale
    start = (float(root_xy[0]), float(root_xy[1]), float(z))
    end = (start[0] + dx * length, start[1] + dy * length, start[2])
    head, half_width = min(0.25, length * 0.25), min(0.14, length * 0.14)
    back = (end[0] - dx * head, end[1] - dy * head, z)
    left = (back[0] - dy * half_width, back[1] + dx * half_width, z)
    right = (back[0] + dy * half_width, back[1] - dx * half_width, z)
    return [start, end, end, left, end, right]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_copy(source, target, attempts=3):
    """Reject a checkpoint changed while copied; never overwrite the source."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError("Snapshot destination must differ from source")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    try:
        for _ in range(attempts):
            before = source.stat()
            shutil.copyfile(source, temporary)
            after = source.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) == (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                temporary.replace(target)
                return before
        raise RuntimeError(f"Source changed during {attempts} copies: {source}")
    finally:
        temporary.unlink(missing_ok=True)


def select_motions(source, ids, target):
    import torch

    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Provide distinct initial motion IDs")
    data = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    if min(ids) < 0 or max(ids) >= len(data["motion_lengths"]):
        raise ValueError("Initial motion ID is outside this physical .pt shard")
    selected = {}
    for key, value in data.items():
        if value is None:
            continue
        if key in FRAME_FIELDS:
            selected[key] = torch.cat(
                [
                    value[
                        int(data["length_starts"][i]) : int(data["length_starts"][i])
                        + int(data["motion_num_frames"][i])
                    ]
                    for i in ids
                ]
            ).clone()
        elif key in MOTION_FIELDS:
            selected[key] = value[ids].clone()
        elif key == "length_starts":
            counts = data["motion_num_frames"][ids]
            selected[key] = torch.cat(
                [torch.zeros(1, dtype=value.dtype), counts[:-1].cumsum(0)]
            ).to(value.dtype)
        elif key == "motion_files":
            selected[key] = tuple(value[i] for i in ids)
        else:
            raise ValueError(f"Unsupported packed motion field: {key}")
    torch.save(selected, target)


def prepare_snapshot(
    checkpoint, motion_file, motion_ids, directory, prior_checkpoint=None
):
    import torch

    checkpoint, motion_file = Path(checkpoint).resolve(), Path(motion_file).resolve()
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            "Snapshot already exists; use a new directory or render the prepared snapshot"
        )
    frozen = checkpoint.parent / "resolved_configs_inference.pt"
    if not frozen.is_file():
        raise FileNotFoundError(f"Missing frozen inference config: {frozen}")
    cfg = torch.load(frozen, map_location="cpu", weights_only=False)
    if "steering" not in cfg["env"].control_components:
        raise ValueError("The checkpoint does not configure a steering task")
    prior = Path(
        prior_checkpoint or cfg["agent"].pretrained_modules["prior"].checkpoint_path
    ).resolve()
    for path in [
        checkpoint,
        checkpoint.parent / "resolved_configs.pt",
        motion_file,
        prior,
        prior.parent / "resolved_configs.pt",
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    policy = directory / "policy"
    before = stable_copy(checkpoint, policy / "last.ckpt")
    state = torch.load(
        policy / "last.ckpt", map_location="cpu", weights_only=False, mmap=True
    )
    reference_keys = sum("reference_prior." in k for k in state.get("model", {}))
    if not reference_keys:
        raise ValueError(
            "Use a full Stage 3 checkpoint with frozen reference state, not adapter-only inference_last.ckpt"
        )
    for name in [
        "resolved_configs.pt",
        "resolved_configs_inference.pt",
        "resolved_configs.yaml",
        "resolved_configs_inference.yaml",
        "experiment_config.py",
        "config.yaml",
    ]:
        if (checkpoint.parent / name).is_file():
            shutil.copy2(checkpoint.parent / name, policy / name)
    stable_copy(prior, directory / "prior/prior.ckpt")
    for name in [
        "resolved_configs.pt",
        "resolved_configs_inference.pt",
        "experiment_config.py",
        "config.yaml",
    ]:
        if (prior.parent / name).is_file():
            shutil.copy2(prior.parent / name, directory / "prior" / name)
    select_motions(motion_file, motion_ids, directory / "selected_motions.pt")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(directory))] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    manifest = {
        "schema_version": 1,
        "source_host": socket.gethostname(),
        "source_checkpoint": str(checkpoint),
        "source_prior": str(prior),
        "source_motion_file": str(motion_file),
        "source_checkpoint_mtime_ns": before.st_mtime_ns,
        "snapshot_time": time.time(),
        "epoch": state.get("epoch"),
        "step_count": state.get("step_count"),
        "reference_keys": reference_keys,
        "initial_motion_ids": list(motion_ids),
        "files": files,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_snapshot(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported steering snapshot schema")
    for relative, expected in manifest["files"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Snapshot file must remain inside the package")
        if (
            path.stat().st_size != expected["bytes"]
            or file_sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"Snapshot checksum mismatch: {relative}")
    return manifest
