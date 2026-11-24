#!/usr/bin/env bash
# Run on the 5090 host. The rootless container mounts /data/atom5/code as /root/code.
set -euo pipefail

run_host=$1
run_ctr="/root/code/${run_host#/data/atom5/code/}"
experiment=/root/code/ProtoMotions-GPC-FSQ93-stage2/examples/experiments/gpc/fsq93_terrain
docker_sock="unix:///run/user/$(id -u)/docker.sock"
container=physics-drift-supermimic93-5090
mkdir -p "$run_host/bench"
exec 9>"$run_host/launcher.lock"
flock -n 9
test ! -e "$run_host/train/stage3-latest.pt"

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
    echo "GPUs are occupied"
    exit 3
fi

candidate() {
    local envs=$1 minibatch=$2 label="e${1}-m${2}"
    local out="$run_host/bench/$label" monitor rc=0
    mkdir -p "$out"
    echo "BENCH_START $label $(date -Is)"
    nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits --loop-ms=750 > "$out/gpu.csv" 2>&1 &
    monitor=$!
    docker -H "$docker_sock" exec "$container" "$experiment/run_onestep_5090.sh" \
        --output "$run_ctr/bench/$label" --envs "$envs" --minibatch "$minibatch" \
        --iterations 2 --steps 32 --benchmark > "$out/launcher.log" 2>&1 || rc=$?
    kill "$monitor" 2>/dev/null || true
    wait "$monitor" 2>/dev/null || true
    echo "$rc" > "$out/status"
    echo "BENCH_END $label rc=$rc $(date -Is)"
    if ((rc == 0)); then
        tail -1 "$out/train.jsonl"
    else
        grep -iE 'out of memory|outofmemory|error|killed' "$out/launcher.log" | tail -8 || true
    fi
}

candidate 1024 4096
candidate 2048 4096
candidate 2048 8192
candidate 3072 8192
candidate 4096 8192
candidate 4096 4096

python3 - "$run_host" <<'PY'
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
records = []
for path in sorted(root.joinpath("bench").glob("e*-m*")):
    status = int(path.joinpath("status").read_text())
    record = {"label": path.name, "exit_code": status}
    peaks = defaultdict(int)
    with path.joinpath("gpu.csv").open() as stream:
        for row in csv.reader(stream):
            if len(row) == 4 and row[1].strip().isdigit() and row[2].strip().isdigit():
                index, memory = int(row[1]), int(row[2])
                peaks[index] = max(peaks[index], memory)
    record["max_board_mib"] = max(peaks.values(), default=0)
    record["peak_board_memory_mib_by_gpu"] = dict(sorted(peaks.items()))
    if status == 0:
        lines = path.joinpath("train.jsonl").read_text().splitlines()
        values = [json.loads(line) for line in lines]
        if len(values) == 2:
            record["warmup"] = values[0]
            record["measured"] = values[1]
    records.append(record)

# Leave at least ~1.5 GiB under the 32607 MiB board capacity for late contacts.
eligible = [r for r in records if r.get("measured") and
            len(r["peak_board_memory_mib_by_gpu"]) == 8 and r["max_board_mib"] <= 31000]
winner = max(eligible, key=lambda r: r["measured"]["transitions_per_second"], default=None)
summary = {"selection": "fastest measured complete PPO iteration after one warmup iteration",
           "board_memory_eligibility_ceiling_mib": 31000, "winner": winner, "candidates": records}
root.joinpath("benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
if winner is None:
    raise SystemExit("No successful benchmark candidate with headroom")
data = winner["measured"]
root.joinpath("selected.txt").write_text(f'{data["envs_per_gpu"]} {data["minibatch"]}\n')
print(f'WINNER {winner["label"]}: {data["transitions_per_second"]:.1f} transitions/s, '
      f'{winner["max_board_mib"]} MiB peak')
PY

read -r envs minibatch < "$run_host/selected.txt"
mkdir -p "$run_host/train"
echo "TRAIN_START e${envs}-m${minibatch} $(date -Is)"
docker -H "$docker_sock" exec "$container" "$experiment/run_onestep_5090.sh" \
    --output "$run_ctr/train" --envs "$envs" --minibatch "$minibatch" > "$run_host/train/launcher.log" 2>&1
test -s "$run_host/train/stage3-latest.pt"
echo "TRAIN_DONE $(date -Is)"
