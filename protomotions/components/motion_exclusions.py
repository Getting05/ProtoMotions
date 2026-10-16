"""Persistent motion exclusions shared by reconstruction and every training stage.

Rules match source keys, never an unscoped shard-local integer. Original motion
IDs remain stable so existing scenes and checkpoint weight vectors stay aligned.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath

DEFAULT_EXCLUSIONS = Path(__file__).resolve().parents[1] / "data" / "motion_exclusions.json"


def normalize_key(key, include_mirrors=False):
    path = PurePosixPath(str(key).replace("\\", "/"))
    if include_mirrors and path.stem.endswith("_M"):
        path = path.with_name(path.stem[:-2] + path.suffix)
    return str(path).lstrip("/")


class MotionExclusions:
    def __init__(self, extra_files=None):
        self.rules = []
        self.files = []
        seen = {}
        for path in dict.fromkeys([str(DEFAULT_EXCLUSIONS), *(extra_files or [])]):
            path = Path(path).expanduser().resolve()
            raw = path.read_bytes()  # Missing/malformed policy must fail closed.
            data = json.loads(raw)
            if data.get("schema_version") != 1 or not isinstance(data.get("exclusions"), list):
                raise ValueError(f"Invalid motion exclusion file: {path}")
            self.files.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
            for entry in data["exclusions"]:
                rule = dict(entry)
                if not all(isinstance(rule.get(k), str) and rule[k].strip()
                           for k in ("id", "motion_key_suffix", "reason")):
                    raise ValueError(f"Exclusion needs id, motion_key_suffix and reason: {path}")
                if type(rule.get("include_mirrors", False)) is not bool:
                    raise ValueError("include_mirrors must be boolean")
                rule["motion_key_suffix"] = normalize_key(rule["motion_key_suffix"], rule.get("include_mirrors", False))
                if ".." in PurePosixPath(rule["motion_key_suffix"]).parts:
                    raise ValueError("Motion key suffix must not contain '..'")
                if rule["id"] in seen:
                    if seen[rule["id"]] != rule:
                        raise ValueError(f"Conflicting motion exclusion id: {rule['id']}")
                    continue
                seen[rule["id"]] = rule
                self.rules.append(rule)

    def matches(self, key):
        result = []
        for rule in self.rules:
            normalized = normalize_key(key, rule.get("include_mirrors", False))
            suffix = rule["motion_key_suffix"]
            if normalized == suffix or normalized.endswith("/" + suffix):
                result.append(rule)
        return result

    def partition(self, keys):
        allowed, excluded = [], []
        for index, key in enumerate(keys):
            matches = self.matches(key)
            if matches:
                excluded.append({"motion_local_id": index, "motion_key": str(key),
                                 "rule_ids": [r["id"] for r in matches],
                                 "reasons": [r["reason"] for r in matches]})
            else:
                allowed.append(index)
        return allowed, excluded

    def require_allowed(self, key):
        matches = self.matches(key)
        if matches:
            raise ValueError(f"Excluded motion {key}: " + "; ".join(r["reason"] for r in matches))


def allowed_motion_ids(library, device=None):
    """Source IDs eligible for training/evaluation, with lightweight test support."""
    import torch
    device = device if device is not None else getattr(library, "device", "cpu")
    ids = torch.arange(library.num_motions(), device=device)
    mask = getattr(library, "motion_excluded_mask", None)
    if mask is not None and len(mask) == len(ids):
        ids = ids[~mask.to(device)]
    return ids


def require_allowed_motion_ids(library, motion_ids):
    import torch
    if not getattr(library, "has_motion_exclusions", False):
        return
    ids = torch.as_tensor(motion_ids, device=library.motion_excluded_mask.device, dtype=torch.long)
    if bool(((ids < 0) | (ids >= library.num_motions())).any()):
        raise ValueError("Motion ID outside the library")
    bad = ids[library.motion_excluded_mask[ids]]
    if bad.numel():
        keys = [library.motion_files[i] for i in bad.unique().cpu().tolist()]
        raise ValueError(f"Excluded motion requested explicitly: {keys}")


def apply_motion_exclusions(library):
    import torch
    policy = MotionExclusions(getattr(library.config, "motion_exclusion_files", None))
    if getattr(library, "motion_files", None) is None:
        raise ValueError("Motion library lacks source motion_files keys; cannot enforce exclusions. Repackage with source keys.")
    if len(library.motion_files) != library.num_motions():
        raise ValueError("Motion source key count does not match the library; cannot enforce exclusions")
    allowed, excluded = policy.partition(library.motion_files)
    library.motion_exclusion_report = {"policy_files": policy.files, "excluded": excluded}
    library.motion_excluded_mask = torch.zeros(library.num_motions(), dtype=torch.bool, device=library.device)
    library.excluded_motion_ids = torch.tensor([e["motion_local_id"] for e in excluded], dtype=torch.long, device=library.device)
    library.has_motion_exclusions = bool(excluded)
    library.motion_excluded_mask[library.excluded_motion_ids] = True
    if excluded:
        library.motion_weights[library.excluded_motion_ids] = 0
        print(f"Motion exclusion policy: blocked {len(excluded)} / {library.num_motions()} motions; source IDs preserved")
    if library.num_motions() and not allowed:
        raise ValueError("Every motion in this library is excluded; supply an eligible shard")
