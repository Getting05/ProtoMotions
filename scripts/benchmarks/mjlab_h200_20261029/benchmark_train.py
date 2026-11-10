# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instrument native training without replacing its learner or physics path."""

import json
import math
import os
from pathlib import Path
import runpy
import time
import torch

rank = int(os.environ.get("LOCAL_RANK", "0"))
folder = Path(os.environ["MJLAB_BENCH_DIR"])
folder.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
torch.cuda.set_device(rank)
budget = float(os.environ.get("MJLAB_TORCH_BUDGET_GIB", "4.5")) * 1024**3
torch.cuda.set_per_process_memory_fraction(
    budget / torch.cuda.get_device_properties(rank).total_memory, rank
)
logfile = folder / f"rank{rank}.jsonl"


def emit(event, **values):
    with logfile.open("a") as f:
        f.write(
            json.dumps(
                dict(event=event, rank=rank, monotonic=time.monotonic(), **values),
                allow_nan=False,
            )
            + "\n"
        )


def scalar(v):
    if isinstance(v, torch.Tensor):
        return float(v.detach().float().mean().item())
    return float(v)


class Metrics:
    def on_fit_start(self, agent):
        self.pending = None
        self.fit_start = time.perf_counter()
        agent._bench_done_count = torch.zeros((), device=agent.device)
        agent._bench_terminated_count = torch.zeros((), device=agent.device)
        emit(
            "fit_start",
            world_size=agent.fabric.world_size,
            num_envs=agent.num_envs,
            num_steps=agent.num_steps,
            batch_size=agent.config.batch_size,
            num_mini_epochs=agent.num_mini_epochs,
            motions=agent.motion_lib.num_motions(),
            torch_budget_gib=budget / 1024**3,
            torch_version=torch.__version__,
        )

    def finish_wall(self, now):
        if self.pending is not None:
            epoch, start = self.pending
            emit("iteration_wall", epoch=epoch, seconds=now - start)
            self.pending = None

    def before_play_steps(self, agent):
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.finish_wall(now)
        agent._bench_iter_start = now
        agent._bench_done_count.zero_()
        agent._bench_terminated_count.zero_()
        torch.cuda.reset_peak_memory_stats()

    def after_train(self, agent):
        torch.cuda.synchronize()
        q = agent.env.simulator._sim.data.qpos
        dq = agent.env.simulator._sim.data.qvel
        assert (
            torch.isfinite(q).all() and torch.isfinite(dq).all()
        ), "Nonfinite physics state"
        if agent.current_epoch % 10 == 0:
            assert all(
                torch.isfinite(p).all() for p in agent.model.parameters()
            ), "Nonfinite model parameter"
        now = time.perf_counter()
        self.pending = (agent.current_epoch, agent._bench_iter_start)
        emit(
            "iteration",
            epoch=agent.current_epoch,
            step_count=agent.step_count,
            rollout_seconds=agent._bench_rollout_seconds,
            update_seconds=agent._bench_update_seconds,
            compute_seconds=now - agent._bench_iter_start,
            done_count=scalar(agent._bench_done_count),
            terminated_count=scalar(agent._bench_terminated_count),
            end_rollout_max_abs_dof_vel=float(dq[:, 6:].abs().max()),
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            losses=agent._bench_losses,
        )

    def on_fit_end(self, agent):
        torch.cuda.synchronize()
        self.finish_wall(time.perf_counter())
        emit(
            "fit_end",
            epoch=agent.current_epoch,
            step_count=agent.step_count,
            fit_seconds=time.perf_counter() - self.fit_start,
        )


from lightning.fabric import Fabric  # noqa: E402 - configure CUDA budget before Fabric
from protomotions.agents.base_agent.agent import BaseAgent  # noqa: E402 - configure CUDA first

original_init = Fabric.__init__


def instrument_init(self, *args, **kwargs):
    kwargs["callbacks"] = [*(kwargs.get("callbacks") or []), Metrics()]
    original_init(self, *args, **kwargs)


Fabric.__init__ = instrument_init
original_optimize = BaseAgent.optimize_model


def measured_optimize(self):
    torch.cuda.synchronize()
    start = time.perf_counter()
    self._bench_rollout_seconds = start - self._bench_iter_start
    result = original_optimize(self)
    torch.cuda.synchronize()
    self._bench_update_seconds = time.perf_counter() - start
    self._bench_losses = {str(k): scalar(v) for k, v in result.items()}
    assert all(
        math.isfinite(v) for v in self._bench_losses.values()
    ), "Nonfinite training metric"
    return result


BaseAgent.optimize_model = measured_optimize
original_record = BaseAgent.record_rollout_step


def measured_record(
    self, next_obs, actions, rewards, dones, terminated, done_indices, extras, step
):
    self._bench_done_count.add_(dones.sum())
    self._bench_terminated_count.add_(terminated.sum())
    return original_record(
        self, next_obs, actions, rewards, dones, terminated, done_indices, extras, step
    )


BaseAgent.record_rollout_step = measured_record
emit(
    "process_start",
    pid=os.getpid(),
    visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
)
try:
    runpy.run_module("protomotions.train_agent", run_name="__main__")
except BaseException as error:
    if not isinstance(error, SystemExit) or error.code not in (None, 0):
        (folder / f"rank{rank}.failed.json").write_text(
            json.dumps(dict(type=type(error).__name__, message=str(error)))
        )
    raise
