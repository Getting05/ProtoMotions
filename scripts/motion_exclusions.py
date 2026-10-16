"""Manage key-based exclusions; audit shards without loading their frame tensors.

python -m scripts.motion_exclusions add --motion-file motions.pt --ids 4,7 --reason 'bad retarget'
python -m scripts.motion_exclusions audit --motion-file motions_slurmrank.pt validation.pt
python -m scripts.motion_exclusions filter-manifest --manifest scene_output/manifest.jsonl
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from protomotions.components.motion_exclusions import DEFAULT_EXCLUSIONS, MotionExclusions, normalize_key


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_keys(path):
    import torch
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)["motion_files"]


def expand_files(paths):
    from protomotions.components.motion_lib import _discover_shard_files
    for path in paths:
        if "slurmrank" in Path(path).name:
            yield from _discover_shard_files(path).values()
        else:
            yield path


def filter_manifest(path, policy):
    """Remove excluded rows from active manifest; retain artifacts as audit history."""
    import fcntl
    path = Path(path).resolve()
    with open(path.parent / ".writer.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = path.read_bytes()
        records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        keep, removed = [], []
        for row in records:
            rules = policy.matches(row["motion_key"])
            if rules:
                removed.append({**row, "status": "excluded", "training_eligible": False,
                                "exclusion_rule_ids": [r["id"] for r in rules]})
            else:
                keep.append(row)
        if removed:
            backup = path.with_name(path.name + ".before_exclusions." + hashlib.sha256(raw).hexdigest()[:12])
            if not backup.exists(): backup.write_bytes(raw)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep)); tmp.replace(path)
        previous = path.parent / "exclusion_audit.json"
        old = json.loads(previous.read_text()).get("excluded", []) if previous.exists() else []
        merged = {r["motion_key"]: r for r in [*old, *removed]}
        audit = {"policy_files": policy.files, "active_records": len(keep), "excluded": list(merged.values())}
        write_json(previous, audit)
        summary_path = path.parent / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            summary["active_after_exclusions"] = len(keep)
            summary["excluded_motion_ids"] = sorted(r["motion_local_id"] for r in merged.values())
            summary["statuses"] = {s: sum(r["status"] == s for r in keep) for s in {r["status"] for r in keep}}
            write_json(summary_path, summary)
        return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add")
    add.add_argument("--file", default=str(DEFAULT_EXCLUSIONS))
    add.add_argument("--motion-file", required=True); add.add_argument("--ids", required=True)
    add.add_argument("--reason", required=True)
    add.add_argument("--exact-only", action="store_true", help="Do not include terminal _M mirrors")
    audit = sub.add_parser("audit")
    audit.add_argument("--motion-file", nargs="+", required=True)
    audit.add_argument("--exclude-file", action="append", default=[])
    audit.add_argument("--output")
    filt = sub.add_parser("filter-manifest")
    filt.add_argument("--manifest", required=True); filt.add_argument("--exclude-file", action="append", default=[])
    args = parser.parse_args()
    if args.command == "add":
        import fcntl
        if not args.reason.strip(): raise ValueError("An exclusion reason is required")
        path = Path(args.file).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
        keys = read_keys(args.motion_file)
        ids = list(dict.fromkeys(int(v) for v in args.ids.split(",")))
        if not ids or min(ids) < 0 or max(ids) >= len(keys): raise ValueError("Invalid source-shard motion ID")
        with open(str(path)+".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            data = json.loads(path.read_text()) if path.exists() else {"schema_version":1,"exclusions":[]}
            if data.get("schema_version") != 1: raise ValueError("Unsupported exclusion schema")
            for mid in ids:
                suffix = "/".join(Path(normalize_key(keys[mid], not args.exact_only)).parts[-2:])
                identity = f"{suffix}|mirrors={not args.exact_only}"
                entry = {"id":"motion-"+hashlib.sha256(identity.encode()).hexdigest()[:12],
                         "motion_key_suffix":suffix,"include_mirrors":not args.exact_only,"reason":args.reason,
                         "source_reference":{"file":str(Path(args.motion_file).resolve()),"motion_local_id":mid,"motion_key":keys[mid]}}
                if not any(r["motion_key_suffix"] == suffix and r.get("include_mirrors",False)==entry["include_mirrors"] for r in data["exclusions"]):
                    data["exclusions"].append(entry)
            write_json(path,data)
        print(json.dumps({"file":str(path),"rules":len(data["exclusions"])},indent=2))
    elif args.command == "audit":
        policy = MotionExclusions(args.exclude_file); results = []
        for path in expand_files(args.motion_file):
            keys = read_keys(path); allowed, excluded = policy.partition(keys)
            results.append({"motion_file":str(path),"total":len(keys),"allowed":len(allowed),"excluded":excluded})
        report = {"policy_files":policy.files,"shards":results}
        if args.output: write_json(args.output,report)
        print(json.dumps(report,indent=2,ensure_ascii=False))
    else:
        report = filter_manifest(args.manifest,MotionExclusions(args.exclude_file))
        print(json.dumps({"active_records":report["active_records"],"excluded_motion_ids":[r["motion_local_id"] for r in report["excluded"]]}))


if __name__ == "__main__": main()
