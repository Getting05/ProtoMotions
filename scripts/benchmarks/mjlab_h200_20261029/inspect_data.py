# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(2)
base = Path(
    "/data/workspace/datasets/ASTRO-SUPERMIMIC/astro_supermimic_88883_20261022/validation_512"
)
m = json.loads((base / "manifest.json").read_text())
p = base / m["output_file"]
d = torch.load(p, map_location="cpu", mmap=True, weights_only=True)
model = Path(
    "/data/workspace/ProtoMotions/protomotions/data/assets/astro_p2/mjcf/astro_p2_protomotions.xml"
)
assert hashlib.sha256(model.read_bytes()).hexdigest() == m["target_model_sha256"]
assert (
    len(d["motion_files"]) == 512 and int(d["motion_num_frames"].sum()) == m["frames"]
)
assert torch.equal(
    d["length_starts"], d["motion_num_frames"].cumsum(0) - d["motion_num_frames"]
)
assert d["contacts"].dtype == torch.bool and d["contacts"].any()
for k in ("gts", "grs", "gvs", "gavs", "dps", "dvs"):
    assert torch.isfinite(d[k]).all()
report = dict(
    motions=512,
    frames=m["frames"],
    families=m["family_counts"],
    dataset_sha256=m["sha256"],
    fps=sorted((1 / d["motion_dt"]).unique().tolist()),
    root_height_min=float(d["gts"][:, 0, 2].min()),
    root_height_max=float(d["gts"][:, 0, 2].max()),
    max_abs_joint_velocity=float(d["dvs"].abs().max()),
    left_foot_contact_fraction=float(d["contacts"][:, 6].float().mean()),
    right_foot_contact_fraction=float(d["contacts"][:, 12].float().mean()),
    tensor_bytes=sum(
        v.numel() * v.element_size() for v in d.values() if isinstance(v, torch.Tensor)
    ),
)
(base / "target_validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
