"""Split a ProtoMotions MotionLib into complete, frame-balanced motion shards."""
import argparse
import gc
import json
import os
from pathlib import Path
import torch
from protomotions.components.motion_lib import MotionLib, MotionLibConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('output_dir', type=Path)
    p.add_argument('--shards', type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(1)
    src = torch.load(a.input, map_location='cpu', weights_only=False, mmap=True)
    n = len(src['motion_num_frames'])
    assert 0 < a.shards <= n
    frame_fields = {'gts','grs','gvs','gavs','dvs','dps','contacts','lrs','goal_states'}
    motion_fields = {'motion_lengths','motion_dt','motion_num_frames','motion_weights'}
    assert not set(src) - frame_fields - motion_fields - {'motion_files','length_starts'}, set(src)
    groups = [[] for _ in range(a.shards)]
    totals = [0] * a.shards
    counts = src['motion_num_frames'].tolist()
    for i in sorted(range(n), key=lambda i: (-counts[i], i)):
        j = min(range(a.shards), key=lambda j: (totals[j], len(groups[j]), j))
        groups[j].append(i)
        totals[j] += counts[i]
    assert sorted(i for group in groups for i in group) == list(range(n))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [a.output_dir / f'amass_astro_p2_{j:02d}.pt' for j in range(a.shards)]
    if any(f.exists() for f in destinations):
        raise FileExistsError('Shard outputs already exist; use a new directory')
    report = dict(source=str(a.input), strategy='Whole motions, longest first into smallest frame total; deterministic',
                  source_motions=n, source_frames=sum(counts), shards=[])
    for j, group in enumerate(groups):
        group.sort()
        ids = torch.tensor(group)
        frames = torch.cat([torch.arange(int(src['length_starts'][i]), int(src['length_starts'][i])+counts[i]) for i in group])
        data = {}
        for k, v in src.items():
            if v is None: data[k] = None
            elif k in frame_fields: data[k] = v[frames]
            elif k in motion_fields: data[k] = v[ids]
            elif k == 'motion_files': data[k] = tuple(v[i] for i in group)
        lengths = data['motion_num_frames']
        data['length_starts'] = lengths.cumsum(0) - lengths
        dest = destinations[j]
        tmp = dest.with_suffix('.partial.pt')
        torch.save(data, tmp)
        del data
        gc.collect()
        lib = MotionLib(MotionLibConfig(motion_file=str(tmp)), device='cpu')
        assert lib.num_motions() == len(group)
        for k, v in src.items():
            if v is None: continue
            actual = getattr(lib, k)
            if k in frame_fields:
                for start in range(0, len(frames), 100000):
                    assert torch.equal(actual[start:start+100000], v[frames[start:start+100000]]), k
            elif k in motion_fields: assert torch.equal(actual, v[ids]), k
            elif k == 'motion_files': assert actual == tuple(v[i] for i in group)
        assert int(lib.length_starts[-1]+lib.motion_num_frames[-1]) == len(frames)
        query_ids = torch.arange(len(group)).repeat_interleave(3)
        times = lib.motion_lengths.repeat_interleave(3) * torch.tensor([0.,.5,1.]).repeat(len(group))
        state = lib.get_motion_state(query_ids, times)
        for k in ('rigid_body_pos','rigid_body_rot','rigid_body_vel','rigid_body_ang_vel','dof_pos','dof_vel'):
            assert torch.isfinite(getattr(state,k)).all(), k
        del lib, state
        gc.collect()
        os.replace(tmp, dest)
        row = dict(index=j, file=str(dest), motions=len(group), frames=len(frames),
                   bytes=dest.stat().st_size, original_motion_indices=group,
                   validation='Exact source tensor equality, MotionLib reload, start/mid/end interpolation PASS')
        report['shards'].append(row)
        print({k:v for k,v in row.items() if k!='original_motion_indices'}, flush=True)
    assert sum(s['motions'] for s in report['shards']) == n
    assert sum(s['frames'] for s in report['shards']) == sum(counts)
    report['validation'] = 'PASS: all source motions covered exactly once, all source frames preserved'
    (a.output_dir/'sharding_manifest.json').write_text(json.dumps(report, indent=2))
    print('DONE', flush=True)


if __name__ == '__main__': main()
