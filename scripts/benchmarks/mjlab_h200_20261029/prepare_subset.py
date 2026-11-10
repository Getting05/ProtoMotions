# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import torch

torch.set_num_threads(2)
base = Path(
    "/root/code/datasets/ASTRO-SUPERMIMIC/astro_supermimic_88883_20261022/p2_motionlib"
)
out = Path("/root/code/datasets/ASTRO-SUPERMIMIC/h200_mjlab_validation_20261029")
out.mkdir(exist_ok=True)
plan_path = base / "conversion/plan.json"
plan = json.loads(plan_path.read_text())
rows = []
for key, entries in plan["jobs"].items():
    for i, r in enumerate(entries):
        rows.append(
            dict(
                r,
                shard=int(key.split("_")[-1]),
                source_index=i,
                family=r["file"].split("/")[0],
            )
        )


def order(r):
    return hashlib.sha256(("mjlab-h200-20261029/" + r["file"]).encode()).hexdigest()


families = defaultdict(list)
for r in rows:
    families[r["family"]].append(r)
selected = {}
for family, items in sorted(families.items()):
    anchors = (
        sorted(items, key=lambda r: r["frames"])[:1]
        + sorted(items, key=lambda r: -r["frames"])[:1]
        + sorted(items, key=order)[:16]
    )
    for r in anchors:
        selected[r["file"]] = r
for r in sorted(rows, key=order):
    if len(selected) >= 512:
        break
    if r["file"] in selected:
        continue
    if sum(x["frames"] for x in selected.values()) + r["frames"] > 600000:
        continue
    selected[r["file"]] = r
assert len(selected) == 512
ordered = sorted(selected.values(), key=lambda r: (r["shard"], r["source_index"]))
chunks = defaultdict(list)
constants = {}
fields = None
for shard in range(8):
    chosen = [r for r in ordered if r["shard"] == shard]
    if not chosen:
        continue
    path = base / f"astro_supermimic_88883_astro_p2_{shard}.pt"
    data = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    n = len(data["motion_files"])
    frames = int(data["motion_num_frames"].sum())
    if fields is None:
        fields = set(data)
    assert set(data) == fields
    for r in chosen:
        i = r["source_index"]
        start = int(data["length_starts"][i])
        end = start + r["frames"]
        assert data["motion_files"][i] == "motions/" + r["file"]
        assert int(data["motion_num_frames"][i]) == r["frames"]
        for k, v in data.items():
            if k == "length_starts":
                continue
            if k == "motion_files":
                chunks[k].append(v[i])
                continue
            if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == frames:
                t = v[start:end].clone()
                assert torch.isfinite(t).all()
                chunks[k].append(t)
            elif isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == n:
                chunks[k].append(v[i : i + 1].clone())
            else:
                constants[k] = v
    print("EXTRACTED", shard, len(chosen), flush=True)
    del data
    gc.collect()
result = dict(constants)
for k, v in chunks.items():
    result[k] = tuple(v) if k == "motion_files" else torch.cat(v)
result["length_starts"] = (
    result["motion_num_frames"].cumsum(0) - result["motion_num_frames"]
)
assert result["contacts"].dtype == torch.bool and result["contacts"].any()
assert len(result["motion_files"]) == 512
assert (result["grs"].norm(dim=-1) - 1).abs().max() < 1e-4
assert (result["dps"][:, -1] == 0).all() and (result["dvs"][:, -1] == 0).all()
path = out / "astro_supermimic_p2_validation_512.pt"
torch.save(result, path)
digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
manifest = dict(
    version="astro_supermimic_88883_20261022",
    source_host="172.16.9.22",
    source_directory=str(base),
    source_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    source_shards=json.loads((base / "final_verification.json").read_text())["shards"],
    target_model_sha256=plan["target_hash"],
    selection_seed="mjlab-h200-20261029",
    selection="512 whole motions; shortest/longest and deterministic family anchors, then SHA256 order; <=600000 frames; no clipping or resampling",
    count=len(ordered),
    frames=int(result["motion_num_frames"].sum()),
    family_counts=dict(Counter(r["family"] for r in ordered)),
    output_file=path.name,
    bytes=path.stat().st_size,
    sha256=digest,
    contact_fraction=float(result["contacts"].float().mean()),
    motions=ordered,
    source_data_transformation="None: copies original float32/bool tensors, original fps and weights; recomputes only concatenation offsets",
)
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
(out / "SHA256SUMS").write_text(f"{digest}  {path.name}\n")
(out / "source_README.md").write_text((base / "README.md").read_text())
print(
    "SUBSET_READY",
    json.dumps(
        {k: v for k, v in manifest.items() if k not in ("motions", "source_shards")}
    ),
    flush=True,
)
