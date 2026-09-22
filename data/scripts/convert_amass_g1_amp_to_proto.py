"""Convert AMP G1 NPZs using ProtoMotions' native FK and MotionLib APIs.

Run from the ProtoMotions repository. Source files are never modified.
Follows convert_g1_csv_to_proto.py and convert_pyroki_retargeted_robot_motions_to_proto.py.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import json
from pathlib import Path
import os
import numpy as np
import torch
from protomotions.components.pose_lib import (extract_kinematic_info,
    extract_transforms_from_qpos, extract_qpos_from_transforms,
    fk_from_transforms_with_velocities, compute_cartesian_velocity)
from contact_detection import compute_contact_labels_from_pos_and_vel

KIN = None

def init_worker():
    global KIN
    torch.set_num_threads(1)
    KIN = extract_kinematic_info('protomotions/data/assets/mjcf/g1_bm_box_feet.xml')

@torch.no_grad()
def convert(job):
    src, dst = map(Path, job)
    try:
        if dst.exists():
            m = torch.load(dst, weights_only=False)
            return dict(file=str(src), output=str(dst), frames=len(m['dof_pos']), fps=float(m['fps']), status='existing')
        with np.load(src, allow_pickle=False) as d:
            names = d['dof_names'].astype(str).tolist()
            bodies = d['body_names'].astype(str).tolist()
            if len(set(names)) != len(names) or set(names) != set(KIN.dof_names):
                raise ValueError('DOF names do not match G1')
            idx = [names.index(n) for n in KIN.dof_names]
            root = bodies.index(KIN.body_names[0])
            fps = float(np.asarray(d['fps']).item())
            joints = torch.tensor(d['dof_positions'][:, idx], dtype=torch.float32)
            pos = torch.tensor(d['body_positions'][:, root], dtype=torch.float32)
            quat = torch.tensor(d['body_rotations'][:, root], dtype=torch.float32)
        if not np.isfinite(fps) or fps <= 0 or len(pos) < 2:
            raise ValueError('Invalid FPS or fewer than two frames')
        qpos = torch.cat([pos, quat, joints], -1)
        if not torch.isfinite(qpos).all():
            raise ValueError('Nonfinite input')
        if (torch.linalg.vector_norm(quat, dim=-1) - 1).abs().max() > 1e-3:
            raise ValueError('Invalid root quaternion norm')
        root_pos, rotations = extract_transforms_from_qpos(KIN, qpos)
        motion = fk_from_transforms_with_velocities(KIN, root_pos, rotations,
            fps=fps, compute_velocities=True, velocity_max_horizon=3)
        motion.dof_pos = extract_qpos_from_transforms(KIN, root_pos, rotations)[:, 7:]
        if not torch.allclose(torch.sin(motion.dof_pos), torch.sin(joints), atol=1e-4):
            raise ValueError('Joint conversion changed pose')
        motion.dof_vel = compute_cartesian_velocity(joints.unsqueeze(1), fps=fps).squeeze(1)
        translations = motion.fix_height_per_frame(height_offset=0.02)
        delta = torch.zeros_like(motion.rigid_body_vel[:, :1])
        delta[:-1] = (translations[1:] - translations[:-1]).unsqueeze(1) * fps
        motion.rigid_body_vel += delta
        motion.fix_height(height_offset=0.04)
        motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
            motion.rigid_body_pos, motion.rigid_body_vel, vel_thres=0.15, height_thresh=0.1).bool()
        motion.local_rigid_body_rot = None
        for field in ('dof_pos', 'dof_vel', 'rigid_body_pos', 'rigid_body_rot', 'rigid_body_vel', 'rigid_body_ang_vel'):
            if not torch.isfinite(getattr(motion, field)).all():
                raise ValueError('Nonfinite output: ' + field)
        assert motion.rigid_body_pos.shape == (len(pos), KIN.num_bodies, 3)
        assert (torch.linalg.vector_norm(motion.rigid_body_rot, dim=-1)-1).abs().max() < 1e-3
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix('.tmp')
        torch.save(motion.to_dict(), tmp)
        os.replace(tmp, dst)
        return dict(file=str(src), output=str(dst), frames=len(pos), fps=fps, status='ok')
    except Exception as e:
        return dict(file=str(src), status='error', error=repr(e))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--package', action='store_true')
    args = p.parse_args()
    files = sorted(args.input_dir.rglob('*.npz'))
    if args.limit: files = files[:args.limit]
    if not files: raise RuntimeError('No NPZ files')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(args.output_dir/'motions'/f.relative_to(args.input_dir).with_suffix('.motion'))) for f in files]
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=init_worker) as pool:
        for i, row in enumerate(pool.map(convert, jobs), 1):
            rows.append(row)
            if i % 100 == 0 or i == len(files):
                print(f'{i}/{len(files)} {dict(Counter(r["status"] for r in rows))}', flush=True)
    (args.output_dir/'conversion_manifest.json').write_text(json.dumps(rows, indent=2))
    errors = [r for r in rows if r['status']=='error']
    summary = dict(input_files=len(files), converted=len(rows)-len(errors), errors=errors,
        frames=sum(r.get('frames', 0) for r in rows),
        duration_seconds=sum((r['frames']-1)/r['fps'] for r in rows if 'frames' in r),
        fps_counts=dict(Counter(str(r['fps']) for r in rows if 'fps' in r)),
        bodies=33, dofs=29, quaternion_order='xyzw', contact_labels='height/speed heuristic',
        motion_filter=False)
    (args.output_dir/'conversion_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if errors: raise RuntimeError('Conversion errors; inspect manifest before packaging')
    if args.package:
        from protomotions.components.motion_lib import MotionLib, MotionLibConfig
        torch.set_num_threads(1)
        lib = MotionLib(MotionLibConfig(motion_file=str(args.output_dir/'motions')), device='cpu')
        assert lib.num_motions() == len(files)
        out = args.output_dir/'amass_g1.pt'
        tmp = args.output_dir/'amass_g1.partial.pt'
        lib.save_to_file(tmp)
        os.replace(tmp, out)
        print('DONE: ' + str(out), flush=True)

if __name__ == '__main__':
    main()
