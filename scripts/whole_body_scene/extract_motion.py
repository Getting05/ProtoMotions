"""Memory-mapped MotionLib reader; motion identity is never inferred from rank."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import torch


@dataclass
class Motion:
    local_id: int
    key: str
    dt: float
    pos: np.ndarray
    rot: np.ndarray
    dof: np.ndarray

    def digest(self):
        h = hashlib.sha256(self.key.encode())
        h.update(np.float64(self.dt).tobytes())
        for x in (self.pos, self.rot, self.dof):
            h.update(np.ascontiguousarray(x, dtype="<f4").tobytes())
        return h.hexdigest()


class MotionArchive:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.data = torch.load(path, weights_only=False, mmap=True, map_location="cpu")
        required = {"gts", "grs", "dps", "motion_num_frames", "length_starts", "motion_dt", "motion_files"}
        if not required.issubset(self.data):
            raise ValueError(f"Not a packaged MotionLib: missing {required-set(self.data)}")
        self.keys = list(self.data["motion_files"])
        if len(self.keys) != len(set(self.keys)):
            raise ValueError("Duplicate motion keys in shard")

    def __len__(self):
        return len(self.keys)

    def get(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        d = self.data
        start, count = int(d["length_starts"][index]), int(d["motion_num_frames"][index])
        sl = slice(start, start+count)
        a = [d[k][sl].numpy().astype(np.float64) for k in ("gts", "grs", "dps")]
        if count < 4 or any(not np.isfinite(x).all() for x in a):
            raise ValueError("Too short or nonfinite motion")
        dt = float(d["motion_dt"][index])
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid motion dt")
        if np.max(np.abs(np.linalg.norm(a[1], axis=-1)-1)) > 1e-3:
            raise ValueError("Invalid body quaternions")
        return Motion(index, self.keys[index], dt, *a)
