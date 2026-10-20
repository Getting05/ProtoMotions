"""Portable source-motion exclusions; packed-library indices remain unchanged."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DEFAULT_GPC_EXCLUSIONS = "protomotions/data/motion_exclusions/p2_gpc_tracking_failed_20261020.json"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_manifest_path(path):
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and not candidate.is_file():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve(strict=True)


def source_key(source, roots):
    """Match whole dataset-root components, retaining subdirectories and mirrors."""
    source = "/" + str(source).replace("\\", "/").lstrip("/")
    for root in roots:
        marker = "/" + root.strip("/") + "/"
        if marker in source:
            return root.strip("/") + "/" + source.split(marker, 1)[1]
    return None


@lru_cache(maxsize=16)
def _read_manifest(path, mtime_ns, size):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported motion exclusion schema: {path}")
    roots, sources = data["source_path_roots"], data["excluded_sources"]
    if not roots or not all(isinstance(root, str) and root.strip('/') for root in roots):
        raise ValueError(f"Invalid source_path_roots: {path}")
    if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
        raise ValueError(f"Invalid excluded_sources: {path}")
    keys = [source_key(source, roots) for source in sources]
    if None in keys or len(keys) != len(set(keys)) or len(keys) != data["num_excluded"]:
        raise ValueError(f"Invalid, duplicate, or inconsistent excluded_sources: {path}")
    return tuple(roots), frozenset(keys)


def source_excluded_ids(path, motion_files):
    resolved = resolve_manifest_path(path)
    stat = resolved.stat()
    roots, excluded = _read_manifest(str(resolved), stat.st_mtime_ns, stat.st_size)
    ids = [i for i, source in enumerate(motion_files) if source_key(source, roots) in excluded]
    print(f"Motion Manager: source blacklist {resolved}: matched {len(ids)}/{len(motion_files)} motions ({len(excluded)} manifest entries)")
    return ids


def configure_gpc_training_exclusions(args, env_config, agent_config, override=None):
    """Called after fresh/warm-start/resume config loading, before configs are saved."""
    if override is None:
        override = getattr(args, "gpc_exclude_failed_motions", None)
    manager = getattr(env_config, "motion_manager", None)
    if manager is None:
        return
    model = getattr(agent_config, "model", None)
    actor = getattr(model, "actor", None)
    mu = getattr(actor, "mu_model", None)
    targets = [str(getattr(obj, "_target_", "")) for obj in (agent_config, model, actor, mu)]
    experiment = "/" + str(getattr(args, "experiment_path", "")).replace("\\", "/")
    family = "/gpc/" in experiment or any(
        ".fsq.FSQAutoEncoder" in target or ".latent_prior_" in target or ".agents.peft." in target
        for target in targets
    )
    enabled = getattr(args, "robot_name", None) == "p2" and family
    if override is not None:
        enabled = override
    current = getattr(manager, "exclude_motion_sources_file", None)
    if enabled:
        manager.exclude_motion_sources_file = current or DEFAULT_GPC_EXCLUSIONS
        # Missing or invalid manifests must stop startup rather than silently train unfiltered.
        resolved = resolve_manifest_path(manager.exclude_motion_sources_file)
        stat = resolved.stat()
        _read_manifest(str(resolved), stat.st_mtime_ns, stat.st_size)
        print(f"GPC training source exclusion enabled: {manager.exclude_motion_sources_file}")
    elif override is False and current == DEFAULT_GPC_EXCLUSIONS:
        manager.exclude_motion_sources_file = None


def clear_bundled_training_exclusions(config):
    """Inference bundles must still allow reviewing a failed clip explicitly."""
    manager = getattr(config, "motion_manager", None)
    if manager is not None and getattr(manager, "exclude_motion_sources_file", None) == DEFAULT_GPC_EXCLUSIONS:
        manager.exclude_motion_sources_file = None
