# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
from pathlib import Path

repo = Path("/data/workspace/ProtoMotions")
base = Path("/data/workspace/mjlab-build/realbench_20261029")
commit = subprocess.check_output(
    ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "rev-parse", "HEAD"],
    text=True,
).strip()
info = dict(
    repo=str(repo),
    commit=commit,
    branch=subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={repo}",
            "-C",
            str(repo),
            "branch",
            "--show-current",
        ],
        text=True,
    ).strip(),
    scripts_sha256={
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in base.glob("*.py")
    },
    dependency_lock_sha256=hashlib.sha256(
        (repo / "requirements_mjlab.lock.txt").read_bytes()
    ).hexdigest(),
    guards={
        "torch_budget_GiB_per_process": 3,
        "single_minimum_free_GiB": 8,
        "final_dual_minimum_free_GiB": 4,
        "monitor_period_seconds": 1,
        "only_own_process_group_signalled": True,
        "gpu_process_memory_not_exposed_by_host": True,
    },
    early_stops={},
)
for name in [
    "mjlab_astro512_dual256_20261029",
    "mjlab_astro512_dual256_v2_20261029",
    "mjlab_astro512_dual256_v3_20261029",
]:
    p = base / name / "result.json"
    if p.exists():
        info["early_stops"][name] = json.loads(p.read_text())
(base / "provenance.json").write_text(json.dumps(info, indent=2))
print(commit)
