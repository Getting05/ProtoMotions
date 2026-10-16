"""Process explicit MotionLib shards and pack matching scene/motion shards.

Each shard has its own resumable output/lock. No source split is changed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion-pattern",required=True,help="Filename containing slurmrank")
    p.add_argument("--output",required=True)
    p.add_argument("--shards",default="0,1,2,3,4,5,6,7")
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--limit-per-shard",type=int)
    p.add_argument("--support-frames",type=int,default=0)
    p.add_argument("--pack",action="store_true")
    args=p.parse_args()
    if "slurmrank" not in Path(args.motion_pattern).name: p.error("Pattern filename must contain slurmrank")
    if args.workers<1:p.error("workers must be positive")
    shards=[int(x) for x in args.shards.split(",")]
    if len(shards)!=len(set(shards)) or any(x<0 for x in shards):p.error("Invalid shard IDs")
    paths={i:Path(args.motion_pattern).with_name(Path(args.motion_pattern).name.replace("slurmrank",str(i))) for i in shards}
    for path in paths.values():
        if not path.is_file():raise FileNotFoundError(path)
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    def run(index):
        out=root/f"shard_{index}"
        command=[sys.executable,"-u","-m","scripts.whole_body_scene.run","--motion-file",str(paths[index]),
                 "--output",str(out),"--support-frames",str(args.support_frames)]
        if args.limit_per_shard is not None:command.extend(["--limit",str(args.limit_per_shard)])
        with open(root/f"shard_{index}.log","a") as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            if args.pack:
                subprocess.run([sys.executable,"-m","scripts.whole_body_scene.pack","--manifest",str(out/"manifest.jsonl"),
                                "--output",str(root/"packed"/f"scenes_{index}.pt")],stdout=log,stderr=subprocess.STDOUT,check=True)
        return index
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index in executor.map(run,shards):print(f"Completed shard {index}",flush=True)


if __name__=="__main__":main()
