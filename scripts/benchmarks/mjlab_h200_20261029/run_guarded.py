# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only controls the process group it creates; never signals existing workloads."""

import argparse
import json
import os
import signal
import statistics
import shutil
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--gpus", required=True)
p.add_argument("--label", required=True)
p.add_argument("--num-envs", type=int, default=256)
p.add_argument("--iterations", type=int, default=100)
p.add_argument("--batch-size", type=int, default=1024)
p.add_argument("--min-free-mib", type=int, default=8192)
p.add_argument("--max-growth-mib", type=int, default=4608)
p.add_argument("--timeout", type=int, default=1800)
p.add_argument(
    "--headroom-only",
    action="store_true",
    help="Use actual free memory plus Torch hard limit; log growth estimates without attributing other jobs to this process",
)
a = p.parse_args()
repo = Path("/data/workspace/ProtoMotions")
base = Path("/data/workspace/mjlab-build/realbench_20261029")
folder = base / a.label
folder.mkdir(parents=True, exist_ok=False)
gpus = [int(x) for x in a.gpus.split(",")]


def gpu_status():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=8,
    )
    return {
        int(v[0]): dict(total=int(v[1]), used=int(v[2]), free=int(v[3]), util=int(v[4]))
        for line in out.splitlines()
        if (v := line.replace(" ", "").split(","))
    }


for source in (Path(__file__), base / "benchmark_train.py"):
    shutil.copy2(source, folder / source.name)
baseline = gpu_status()
references = [g for g in baseline if g not in gpus]
for g in gpus:
    if baseline[g]["free"] < a.min_free_mib + a.max_growth_mib:
        raise SystemExit(f"GPU {g} insufficient headroom: {baseline[g]}")
(folder / "baseline.json").write_text(json.dumps(baseline, indent=2))
env = os.environ.copy()
env.update(
    CUDA_VISIBLE_DEVICES=a.gpus,
    MUJOCO_GL="disable",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    PYTHONUNBUFFERED="1",
    NCCL_CUMEM_HOST_ENABLE="0",
    TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
    MJLAB_BENCH_DIR=str(folder),
    MJLAB_TORCH_BUDGET_GIB="3.0",
)
command = [
    "nice",
    "-n",
    "15",
    "/opt/venvs/protomotions-mjlab/bin/python",
    str(base / "benchmark_train.py"),
    "--robot-name",
    "astro_p2",
    "--simulator",
    "mjlab",
    "--headless",
    "--num-envs",
    str(a.num_envs),
    "--batch-size",
    str(a.batch_size),
    "--motion-file",
    "/data/workspace/datasets/ASTRO-SUPERMIMIC/astro_supermimic_88883_20261022/validation_512/astro_supermimic_p2_validation_512.pt",
    "--experiment-path",
    "examples/experiments/mimic/fsq_reduced_coord_obs.py",
    "--experiment-name",
    a.label,
    "--ngpu",
    str(len(gpus)),
    "--no-gpc-exclude-failed-motions",
    "--training-max-iterations",
    str(a.iterations),
    "--overrides",
    "agent.fail_on_bad_grads=True",
    "agent.save_last_checkpoint_every=50",
]
(folder / "launch.json").write_text(
    json.dumps(
        dict(
            command=command,
            limits=vars(a),
            environment={
                k: env[k]
                for k in [
                    "CUDA_VISIBLE_DEVICES",
                    "MJLAB_TORCH_BUDGET_GIB",
                    "OMP_NUM_THREADS",
                    "NCCL_CUMEM_HOST_ENABLE",
                ]
            },
        ),
        indent=2,
    )
)
log = (folder / "train.log").open("w")
telemetry = (folder / "gpu.jsonl").open("w", buffering=1)
proc = subprocess.Popen(
    command,
    cwd=repo,
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
(folder / "pid").write_text(str(proc.pid))
started = time.monotonic()
abort = None
failures = 0
try:
    while proc.poll() is None:
        try:
            status = gpu_status()
            failures = 0
            shifts = [status[g]["used"] - baseline[g]["used"] for g in references]
            common_shift = statistics.median(shifts) if shifts else 0
            # Correct only coherent changes on a majority of unused reference GPUs.
            coherent = (
                shifts
                and sum(abs(x - common_shift) <= 256 for x in shifts) > len(shifts) / 2
            )
            if not coherent or abs(common_shift) < 128:
                common_shift = 0
            growth = {
                g: max(0, status[g]["used"] - baseline[g]["used"] - common_shift)
                for g in gpus
            }
            telemetry.write(
                json.dumps(
                    dict(
                        elapsed=time.monotonic() - started,
                        gpus=status,
                        common_external_shift_mib=common_shift,
                        estimated_owned_growth_mib=growth,
                    )
                )
                + "\n"
            )
            for g in gpus:
                if status[g]["free"] < a.min_free_mib:
                    abort = f"GPU {g} free memory below {a.min_free_mib} MiB"
                if not a.headroom_only and growth[g] > a.max_growth_mib:
                    abort = f"GPU {g} reference-corrected memory growth above {a.max_growth_mib} MiB"
        except Exception as error:
            failures += 1
            if failures >= 3:
                abort = f"GPU monitor unavailable: {error}"
        if list(folder.glob("rank*.failed.json")):
            abort = "Training rank reported an error"
        if time.monotonic() - started > a.timeout:
            abort = "Wall-time limit"
        if abort:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            break
        time.sleep(1)
finally:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    # Reap any remaining Fabric children from this owned process group only.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    log.close()
    telemetry.close()
result = dict(
    returncode=proc.returncode,
    abort_reason=abort,
    seconds=time.monotonic() - started,
    final_gpus=gpu_status(),
)
(folder / "result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
raise SystemExit(0 if proc.returncode == 0 and not abort else 1)
