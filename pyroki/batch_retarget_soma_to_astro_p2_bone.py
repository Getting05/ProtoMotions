#!/usr/bin/env python3
"""SOMA77 BVH/NPZ or SOMA23 ProtoMotions .motion -> P2 using bone calibration."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml

import batch_retarget_to_astro_p2_bone as common
from calibration_viewer.calibration import CalibrationParams
from calibration_viewer.motion import RootTrajectoryParams

ROOT = common.REPO_ROOT
VERSION = 'soma-p2-bone-ground-hands-v6'
BVH77_NAMES = ('Hips', 'Spine1', 'Spine2', 'Chest', 'Neck1', 'Neck2', 'Head', 'HeadEnd', 'Jaw', 'LeftEye', 'RightEye', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumbEnd', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex4', 'LeftHandIndexEnd', 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle4', 'LeftHandMiddleEnd', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4', 'LeftHandRingEnd', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky4', 'LeftHandPinkyEnd', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumbEnd', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex4', 'RightHandIndexEnd', 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle4', 'RightHandMiddleEnd', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4', 'RightHandRingEnd', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4', 'RightHandPinkyEnd', 'LeftLeg', 'LeftShin', 'LeftFoot', 'LeftToeBase', 'LeftToeEnd', 'RightLeg', 'RightShin', 'RightFoot', 'RightToeBase', 'RightToeEnd')
TPOSE_OFFSETS = ROOT/'data/soma/standard_t_pose_global_offsets_rots.p'

MJCF = ROOT/'protomotions/data/assets/mjcf/soma23_humanoid.xml'
DEFAULT_CONFIG = Path(__file__).with_name('soma_p2_bone_calibration.yaml')
# User-confirmed calibration (2026-09-28). Normal runs do not fit parameters.
APPROVED_CALIBRATION = json.loads(r'''{
  "format_version": 2,
  "human": {
    "format": "soma",
    "layout": "soma23",
    "units": "m",
    "axes": [
      "-y",
      "x",
      "z"
    ],
    "skeleton": "soma23"
  },
  "robot": {
    "format": "urdf",
    "base_link": "pelvis",
    "keypoint_links": {
      "pelvis": "pelvis",
      "left_hip": "left_hip_pitch_link",
      "right_hip": "right_hip_pitch_link",
      "left_knee": "left_knee_link",
      "right_knee": "right_knee_link",
      "left_ankle": "left_ankle_roll_link",
      "right_ankle": "right_ankle_roll_link",
      "left_foot": "left_foot_link",
      "right_foot": "right_foot_link",
      "left_shoulder": "left_shoulder_pitch_link",
      "right_shoulder": "right_shoulder_pitch_link",
      "left_elbow": "left_elbow_link",
      "right_elbow": "right_elbow_link",
      "left_wrist": "left_wrist_yaw_link",
      "right_wrist": "right_wrist_yaw_link",
      "neck": "head_link"
    },
    "t_pose_joint_positions": {
      "left_hip_pitch_joint": 0.0,
      "left_hip_roll_joint": 0.0,
      "left_hip_yaw_joint": 0.0,
      "left_knee_joint": 0.0,
      "left_ankle_pitch_joint": 0.0,
      "left_ankle_roll_joint": 0.0,
      "right_hip_pitch_joint": 0.0,
      "right_hip_roll_joint": 0.0,
      "right_hip_yaw_joint": 0.0,
      "right_knee_joint": 0.0,
      "right_ankle_pitch_joint": 0.0,
      "right_ankle_roll_joint": 0.0,
      "waist_yaw_joint": 0.0,
      "waist_pitch_joint": 0.0,
      "waist_roll_joint": 0.0,
      "left_shoulder_pitch_joint": 0.0,
      "left_shoulder_roll_joint": 1.5708,
      "left_shoulder_yaw_joint": 0.0,
      "left_elbow_joint": 1.5708,
      "left_wrist_roll_joint": 0.0,
      "left_wrist_pitch_joint": 0.0,
      "left_wrist_yaw_joint": 0.0,
      "right_shoulder_pitch_joint": 0.0,
      "right_shoulder_roll_joint": -1.5708,
      "right_shoulder_yaw_joint": 0.0,
      "right_elbow_joint": 1.5708,
      "right_wrist_roll_joint": 0.0,
      "right_wrist_pitch_joint": 0.0,
      "right_wrist_yaw_joint": 0.0,
      "head_joint": 0.0
    }
  },
  "calibration": {
    "mode": "bone",
    "upper_scale": [
      1.0,
      1.0,
      1.0
    ],
    "lower_scale": [
      1.0,
      1.0,
      1.0
    ],
    "shoulder_offset": 0.0,
    "elbow_offset": 0.0,
    "bone_scales": {
      "torso": 0.950070275306741,
      "upper_arm": 0.9994727232391987,
      "forearm": 0.779078544121117,
      "thigh": 0.6997741356596586,
      "shank": 0.73
    },
    "joint_offsets": {
      "shoulder": [
        0.012198193251247168,
        -0.04679651151977265,
        -0.01806821703437234
      ],
      "elbow": [
        2.0606839926650682e-06,
        -0.029354110386513747,
        -2.487063596916636e-06
      ],
      "hip": [
        -0.02292282036415529,
        0.011549422528873539,
        0.085794197531323
      ]
    },
    "asymmetric": false
  },
  "root_trajectory": {
    "scale": [
      1.0,
      1.0,
      1.0
    ],
    "offset": [
      0.0,
      0.0,
      0.0
    ]
  }
}''')
DEFAULT_INPUT = Path('/data/chenguanting/datasets/BONES-SEED/soma_uniform')
DEFAULT_OUTPUT = Path('/data/chenguanting/datasets/BONES-SEED/p2_bone_pyroki')

SOMA23 = ('Hips','Spine1','Spine2','Chest','Neck1','Neck2','Head',
          'RightShoulder','RightArm','RightForeArm','RightHand',
          'LeftShoulder','LeftArm','LeftForeArm','LeftHand',
          'RightLeg','RightShin','RightFoot','RightToeBase',
          'LeftLeg','LeftShin','LeftFoot','LeftToeBase')
# SOMASkeleton77 body selection, matching convert_soma23_to_proto.py.
SELECT77 = (0,1,2,3,4,5,6,39,40,41,42,11,12,13,14,72,73,74,75,67,68,69,70)
LANDMARKS = ('Hips','LeftLeg','RightLeg','LeftShin','RightShin',
             'LeftFoot','RightFoot','LeftToeBase','RightToeBase',
             'LeftArm','RightArm','LeftForeArm','RightForeArm','LeftHand','RightHand')
KEY_IDS = [SOMA23.index(n) for n in LANDMARKS]
FOOT_IDS = [SOMA23.index(n) for n in ('LeftFoot','LeftToeBase','RightFoot','RightToeBase')]
RAW_BODY_BASIS = common.basis_for(['z','x','y'])
MJCF_BODY_BASIS = common.basis_for(['-y','x','z'])
# Matches ProtoMotions' raw SOMA -> .motion world transform (rot2).
RAW_WORLD_BASIS = common.basis_for(['-x','z','y'])


def load_skeleton(path=MJCF):
    root = ET.parse(path).getroot().find('worldbody/body')
    names, parents, offsets = [], [], []
    def visit(node, parent):
        if any(k in node.attrib for k in ('quat','euler','axisangle','xyaxes','zaxis')):
            raise ValueError('SOMA MJCF reference rotations require an explicit adapter')
        index = len(names)
        names.append(node.attrib['name']); parents.append(parent)
        offsets.append(np.fromstring(node.get('pos','0 0 0'),sep=' '))
        for child in node.findall('body'): visit(child,index)
    visit(root,-1)
    if tuple(names) != SOMA23:
        raise ValueError('SOMA MJCF body topology/order differs from the validated SOMA23 layout')
    offsets = np.asarray(offsets)
    rest = offsets.copy()
    for i in range(1,len(rest)): rest[i] += rest[parents[i]]
    return np.asarray(parents), rest @ MJCF_BODY_BASIS.T


def bone_transform(points, parents, params):
    """SOMA's own 23-node tree, in pelvis-local forward/left/up coordinates."""
    points = np.asarray(points,dtype=float)
    if points.shape[-2:] != (23,3) or not np.isfinite(points).all():
        raise ValueError('Expected finite [...,23,3] SOMA points')
    params.validate()
    if params.mode != 'bone': raise ValueError('SOMA requires bone calibration')
    result = points.copy()
    for i,name in enumerate(SOMA23[1:],1):
        side = 'left' if name.startswith('Left') else 'right' if name.startswith('Right') else None
        part = name.removeprefix('Left').removeprefix('Right')
        group = {'ForeArm':'upper_arm','Hand':'forearm','Shin':'thigh','Foot':'shank'}.get(part)
        if name in ('Spine1','Spine2','Chest','Neck1','Neck2'): group = 'torso'
        scale = params.bone_scales.get(group,1.)
        if params.asymmetric and side and group:
            scale = params.bone_scales.get(f'{side}_{group}',scale)
        parent = parents[i]
        result[...,i,:] = result[...,parent,:] + scale*(points[...,i,:]-points[...,parent,:])
        joint = {'Arm':'shoulder','ForeArm':'elbow','Leg':'hip'}.get(part)
        if joint:
            result[...,i,:] += np.asarray(params.joint_offsets.get(joint,[0,0,0])) * [1,1 if side=='left' else -1,1]
    return result


def load_config(path, mjcf):
    if path is None:
        cfg = json.loads(json.dumps(APPROVED_CALIBRATION))
        params = CalibrationParams.from_mapping(cfg['calibration'])
        root = RootTrajectoryParams.from_mapping(cfg.get('root_trajectory'))
        basis = common.basis_for(cfg['human']['axes'])
    else:
        cfg,params,root,basis = common.load_calibration(path)
    human=cfg['human']
    valid_layout=human.get('format')=='soma23' or (human.get('format')=='soma' and human.get('layout',human.get('skeleton'))=='soma23')
    if not valid_layout or human.get('units','m')!='m' or not np.allclose(basis,MJCF_BODY_BASIS):
        raise ValueError('Require a SOMA23 calibration with axes [-y,x,z]; SMPL calibration is not interchangeable')
    expected = cfg.get('source_skeleton',{}).get('sha256')
    if expected and hashlib.sha256(Path(mjcf).read_bytes()).hexdigest() != expected:
        raise ValueError('Source MJCF changed since calibration; regenerate the SOMA calibration')
    return cfg,params,root


def fit_calibration(args,parents,rest):
    """Generate a reproducible initial fit; a static pose is not dynamic validation."""
    from calibration_viewer.robot import RobotModel
    with open(args.robot_calibration) as stream: template = yaml.safe_load(stream)
    model = RobotModel.load(args.urdf_path,base_link='pelvis',
                            keypoint_links=template['robot']['keypoint_links'],mesh_dir=args.mesh_dir)
    model.update(template['robot']['t_pose_joint_positions'])
    target = model.keypoints()
    human = rest[KEY_IDS]-rest[0]
    robot = np.stack([target[n]-target['pelvis'] for n in common.KEYPOINT_NAMES])
    initial = CalibrationParams.defaults('bone')
    # Anatomical length priors anchor the underconstrained single-pose fit.
    pairs = {'upper_arm':[(9,11),(10,12)],'forearm':[(11,13),(12,14)],
             'thigh':[(1,3),(2,4)],'shank':[(3,5),(4,6)]}
    for group,edges in pairs.items():
        initial.bone_scales[group] = float(np.mean([np.linalg.norm(robot[b]-robot[a])/np.linalg.norm(human[b]-human[a]) for a,b in edges]))
    chest = rest[SOMA23.index('Chest'),2]-rest[0,2]
    shoulder_height = human[[9,10],2].mean()
    initial.bone_scales['torso'] = float(np.clip((robot[[9,10],2].mean()-(shoulder_height-chest))/chest,.3,2.))
    x0 = initial.vector()
    def residual(x,regularize=True):
        out = bone_transform(rest,parents,initial.with_vector(x))[KEY_IDS]
        delta = (out-out[0]-robot).ravel()
        if regularize: delta=np.r_[delta,.08*(x[:5]-x0[:5]),.02*x[5:]]
        return delta
    solved = least_squares(residual,np.clip(x0,np.r_[np.full(5,.3),np.full(9,-.2)],np.r_[np.full(5,2),np.full(9,.2)]),
                           bounds=(np.r_[np.full(5,.3),np.full(9,-.2)],np.r_[np.full(5,2),np.full(9,.2)]))
    params = initial.with_vector(solved.x)
    report = dict(method='regularized single neutral-pose fit with anatomical length priors',
                  success=bool(solved.success),rmse_m=float(np.sqrt(np.mean(residual(solved.x,False)**2))),
                  raw_jacobian_rank=int(np.linalg.matrix_rank(solved.jac[:45],tol=1e-7)),parameter_count=len(x0),
                  bound_hits=int(np.count_nonzero(solved.active_mask)),
                  note='Initial calibration only: torso scale and shoulder offset are coupled in one pose; verify representative motions.')
    cfg = dict(format_version=2,human=dict(format='soma23',axes=['-y','x','z']),robot=template['robot'],
               calibration=params.serializable(),root_trajectory=dict(scale=[1.,1.,1.],offset=[0.,0.,0.]),
               source_skeleton=dict(path=str(args.soma_mjcf),sha256=hashlib.sha256(args.soma_mjcf.read_bytes()).hexdigest()),
               fit_report=report)
    if not solved.success: raise RuntimeError(f'Calibration failed: {solved.message}')
    args.calibration_config.parent.mkdir(parents=True,exist_ok=True)
    args.calibration_config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    print(json.dumps(report,indent=2));print(f'Saved {args.calibration_config}')


def validate_rotations(rotations):
    r = common.finite_array(rotations,'rotations')
    if r.shape[-2:] != (3,3) or not np.allclose(r.swapaxes(-1,-2)@r,np.eye(3),atol=2e-4) or not np.allclose(np.linalg.det(r),1,atol=2e-4):
        raise ValueError('Orientations must be proper rotation matrices')
    # Remove float32 drift before the body-frame calibration.
    return Rotation.from_matrix(r.reshape(-1,3,3)).as_matrix().reshape(r.shape)



def load_bvh_motion(path,args,parents):
    """BONES-SEED BVH -> neutral SOMA23 FK, using the repository T-pose offsets.

    BVH uses bone-aligned zero frames, not SOMA's canonical T pose. Retain
    the converter's absolute Hips translation convention (centimeters).
    """
    import torch
    sys.path.insert(0,str(ROOT/'data/scripts'))
    from bvh import Bvh
    text=path.read_text()
    mocap=Bvh(text if text.endswith('\n') else text+'\n',backend='np')
    names=mocap.get_joints_names()
    if names != ['Root',*BVH77_NAMES] and names != list(BVH77_NAMES):
        raise ValueError('BVH must use the BONES-SEED SOMA77 hierarchy/order')
    if len(mocap.frames)!=mocap.nframes or mocap.nframes<2:
        raise ValueError('BVH frame count does not match its header')
    data=mocap.np_data_array
    if not np.isfinite(data).all(): raise ValueError('Nonfinite BVH motion channels')
    if names[0]=='Root':
        channels=mocap.joint_channels('Root')
        wrapper=np.asarray(mocap.frames_joint_channels('Root',channels))
        if not np.allclose(wrapper,0.,atol=1e-6) or not np.allclose(mocap.joint_offset('Root'),0.):
            raise ValueError('Animated BVH Root wrapper is unsupported; refusing to discard its transform')
    global_rot=np.empty((mocap.nframes,77,3,3))
    for i,name in enumerate(BVH77_NAMES):
        channels=mocap.joint_channels(name)
        rot=[c for c in channels if c.endswith('rotation')]
        if len(rot)!=3 or len({c[0] for c in rot})!=3:
            raise ValueError(f'Invalid rotation channels for {name}')
        angles=np.asarray(mocap.frames_joint_channels(name,rot))
        local=Rotation.from_euler(''.join(c[0] for c in rot),angles,degrees=True).as_matrix()
        if i==0:
            global_rot[:,i]=local
        else:
            parent=BVH77_NAMES.index(mocap.joint_parent(name).name)
            global_rot[:,i]=global_rot[:,parent]@local
        pos_channels=[c for c in channels if c.endswith('position')]
        if i and pos_channels:
            raise ValueError(f'Unexpected non-root translation channels on {name}')
    root_channels=mocap.joint_channels('Hips')
    if not all(c in root_channels for c in ('Xposition','Yposition','Zposition')):
        raise ValueError('BVH Hips must provide XYZ translation channels')
    root_pos=np.asarray(mocap.frames_joint_channels('Hips',['Xposition','Yposition','Zposition']))*.01
    offset_path=getattr(args,'tpose_offsets',TPOSE_OFFSETS)
    offsets=torch.load(offset_path,map_location='cpu',weights_only=True)
    if not isinstance(offsets,torch.Tensor) or tuple(offsets.shape)!=(77,3,3):
        raise ValueError('T-pose offsets must be a tensor [77,3,3]')
    offsets=validate_rotations(offsets.detach().cpu().numpy())
    tpose_rot=global_rot@offsets.swapaxes(-1,-2)
    world=RAW_WORLD_BASIS if args.raw_world_axes=='proto' else common.basis_for(args.raw_world_axes.split(','))
    rotations=validate_rotations(world@tpose_rot[:,SELECT77]@RAW_BODY_BASIS.T)
    _,rest=load_skeleton(getattr(args,'soma_mjcf',MJCF))
    points=np.empty((len(rotations),23,3));points[:,0]=root_pos@world.T
    for i in range(1,23):
        points[:,i]=points[:,parents[i]]+np.einsum('tij,j->ti',rotations[:,parents[i]],rest[i]-rest[parents[i]])
    frame_time=float(mocap.frame_time)
    if not np.isfinite(frame_time) or frame_time<=0: raise ValueError('Invalid BVH Frame Time')
    fps=1./frame_time
    # Typical BVH writes 120 Hz as 0.008333; undo only decimal-print rounding.
    nearest=round(fps)
    if nearest>0 and abs(fps-nearest)/nearest<1e-4: fps=float(nearest)
    if args.input_fps is not None: fps=float(args.input_fps)
    if not np.isfinite(fps) or fps<=0: raise ValueError('Invalid FPS')
    axes='BVH cm -> m; bone-frame -> T-pose; world '+('[-x,z,y]' if args.raw_world_axes=='proto' else args.raw_world_axes)
    return points,rotations,None,fps,axes,'height+speed heuristic (BVH has no contacts)'


def load_motion(path,args,parents):
    if path.suffix.lower()=='.bvh': return load_bvh_motion(path,args,parents)
    if path.suffix == '.motion':
        import torch
        sys.path.insert(0,str(ROOT))
        from protomotions.simulator.base_simulator.simulator_state import StateConversion
        # Only this inspected enum is allowlisted; never use weights_only=False.
        with torch.serialization.safe_globals([StateConversion]):
            data=torch.load(path,map_location='cpu',weights_only=True)
        if data.get('state_conversion') != StateConversion.COMMON:
            raise ValueError('.motion requires COMMON body ordering')
        data={k:v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v for k,v in data.items()}
        points=common.finite_array(data['rigid_body_pos'],'rigid_body_pos')
        quat=common.finite_array(data['rigid_body_rot'],'rigid_body_rot')
        if points.ndim!=3 or points.shape[1:]!=(23,3) or quat.shape!=points.shape[:-1]+(4,):
            raise ValueError('Expected SOMA23 .motion positions [T,23,3] and XYZW quaternions [T,23,4]')
        if not np.allclose(np.linalg.norm(quat,axis=-1),1,atol=2e-3): raise ValueError('Invalid quaternion norms')
        rotations=Rotation.from_quat(quat.reshape(-1,4)).as_matrix().reshape(-1,23,3,3)
        rotations=rotations @ MJCF_BODY_BASIS.T
        world_axes='ProtoMotions world XYZ; local SOMA frame -Y forward/Z up'
        contacts=data.get('rigid_body_contacts')
        contacts=None if contacts is None else np.asarray(contacts)[:,FOOT_IDS]
        contact_source='SOMA23 per-body contacts' if contacts is not None else 'height+speed heuristic'
    else:
        with np.load(path,allow_pickle=False) as f: data=dict(f)
        points=common.finite_array(data['posed_joints'],'posed_joints')
        if points.ndim!=3 or points.shape[1:] not in ((77,3),(23,3)):
            raise ValueError('Raw SOMA NPZ requires posed_joints [T,77,3] or SOMA23 [T,23,3]')
        count=points.shape[1]
        selected=SELECT77 if count==77 else tuple(range(23))
        if 'joint_names' in data:
            names=[str(x) for x in data['joint_names']]
            if len(names)!=count or len(set(names))!=count: raise ValueError('Invalid joint_names')
            selected=tuple(names.index(n) for n in SOMA23)
        points=points[:,selected]
        if 'global_rot_mats' in data:
            raw_rot=common.finite_array(data['global_rot_mats'],'global_rot_mats')
            if raw_rot.shape!=(len(points),count,3,3): raise ValueError('global_rot_mats shape mismatch')
            rotations=raw_rot[:,selected]
        elif 'local_rot_mats' in data:
            local=common.finite_array(data['local_rot_mats'],'local_rot_mats')
            if local.shape!=(len(points),count,3,3): raise ValueError('local_rot_mats shape mismatch')
            local=local[:,selected];rotations=local.copy()
            for i in range(1,23): rotations[:,i]=rotations[:,parents[i]]@local[:,i]
        else: raise ValueError('Raw SOMA requires global_rot_mats or local_rot_mats')
        world=RAW_WORLD_BASIS if args.raw_world_axes=='proto' else common.basis_for(args.raw_world_axes.split(','))
        points=points@world.T
        rotations=world@rotations@RAW_BODY_BASIS.T
        world_axes='-x,z,y (ProtoMotions raw SOMA transform)' if args.raw_world_axes=='proto' else args.raw_world_axes
        contacts=None;contact_source='height+speed heuristic'
        if 'foot_contacts' in data and args.foot_contact_order:
            names=args.foot_contact_order.split(',')
            expected=['left_ankle','left_foot','right_ankle','right_foot']
            c=np.asarray(data['foot_contacts'])
            if len(names)!=4 or set(names)!=set(expected) or c.shape!=(len(points),4):
                raise ValueError('foot_contact_order must name each of left_ankle,left_foot,right_ankle,right_foot exactly once')
            contacts=c[:,[names.index(n) for n in expected]];contact_source='explicit raw foot_contact_order'
        elif 'foot_contacts' in data:
            contact_source='height+speed heuristic; ambiguous raw foot_contacts order unused'
    if len(points)<2: raise ValueError('Motion needs at least two frames')
    rotations=validate_rotations(rotations)
    fps=args.input_fps if args.input_fps is not None else data.get('fps',data.get('mocap_framerate'))
    if fps is None or not np.isfinite(float(fps)) or float(fps)<=0: raise ValueError('FPS missing/invalid; pass --input-fps')
    if contacts is not None:
        contacts=common.finite_array(contacts,'contacts')
        if contacts.shape!=(len(points),4) or np.any((contacts<0)|(contacts>1)): raise ValueError('Invalid contact array')
    return points,rotations,contacts,float(fps),world_axes,contact_source


def prepare_motion(path,args,parents,params,root,keypoint_names=None):
    points,rotations,contacts,fps,axes,contact_source=load_motion(path,args,parents)
    stop=None if args.max_frames is None else args.start_frame+args.max_frames
    frames=np.arange(len(points))[args.start_frame:stop:args.subsample_factor]
    if len(frames)<2: raise ValueError('Need at least two selected frames')
    points,rotations=points[frames],rotations[frames]
    rate=fps/args.subsample_factor
    roots=points[:,0].copy(); root_R=rotations[:,0]
    local=np.einsum('tjk,tkl->tjl',points-roots[:,None],root_R)
    calibrated=bone_transform(local,parents,params)@root_R.swapaxes(-1,-2)+root.apply(roots)[:,None]
    names=tuple(keypoint_names) if keypoint_names is not None else common.KEYPOINT_NAMES+('neck',)
    source_ids=[SOMA23.index(dict(zip(common.KEYPOINT_NAMES,LANDMARKS),neck='Neck1')[n]) for n in names]
    n=len(names)
    targets=np.empty((len(points),n+3,3)); orientations=np.empty((len(points),n+3,3,3))
    targets[:,:n]=calibrated[:,source_ids];orientations[:,:n]=rotations[:,source_ids]
    for wi,ai,sign in ((names.index('left_wrist'),n,1),(names.index('right_wrist'),n+1,-1)):
        # SOMA hand's longitudinal rest direction is native +/-X (canonical +/-Y).
        targets[:,ai]=targets[:,wi]+.11*np.einsum('tij,j->ti',orientations[:,wi],[0,sign,0])
        orientations[:,ai]=orientations[:,wi]
    targets[:,n+2]=targets[:,0]+np.einsum('tij,j->ti',root_R,[.18,0,0]);orientations[:,n+2]=root_R
    shift=0.
    if args.align_ground:
        shift=-float(np.percentile(calibrated[:,FOOT_IDS,2],2))+float(root.offset[2]);targets[...,2]+=shift
    feet=points[:,FOOT_IDS]
    ground=float(np.percentile(feet[...,2],2)) if args.ground_height is None else args.ground_height
    if contacts is None:
        speed=np.linalg.norm(np.gradient(feet,axis=0)*rate,axis=-1)
        contacts=((feet[...,2]<ground+args.contact_height)&(speed<args.contact_speed)).astype(float)
    else: contacts=contacts[frames]
    left=common.smooth_contacts(contacts[:,:2].mean(axis=1,keepdims=True))
    right=common.smooth_contacts(contacts[:,2:].mean(axis=1,keepdims=True))
    metadata=dict(source_path=str(path.resolve()),source_frames=frames.tolist(),source_fps=fps,fps=rate,
                  input_axes=axes,keypoint_names=list(names),source_keypoint_names=[SOMA23[i] for i in source_ids],source_skeleton='SOMA23 FK from T-pose-corrected BVH' if path.suffix.lower()=='.bvh' else 'SOMA23 selected from original positions',contact_source=contact_source,
                  estimated_ground_height_m=ground,ground_alignment_m=shift)
    return targets,orientations,left,right,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=DEFAULT_INPUT,help='SOMA BVH/NPZ/.motion or directory; defaults to all raw BONES-SEED BVHs')
    p.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--calibration-config',type=Path,help='Optional YAML override; default uses the user-confirmed parameters embedded in this script')
    p.add_argument('--fit-calibration',action='store_true',help='Generate a SOMA-specific neutral-pose initial calibration, then exit')
    p.add_argument('--robot-calibration',type=Path,default=Path(common.DEFAULT_CALIBRATION),help='Only robot mapping/reference joint pose is used for fitting')
    p.add_argument('--tpose-offsets',type=Path,default=TPOSE_OFFSETS,help='SOMA77 BVH bone-frame to T-pose rotation tensor')
    p.add_argument('--soma-mjcf',type=Path,default=MJCF)
    p.add_argument('--input-fps',type=float,help='Required when raw NPZ has no FPS metadata')
    p.add_argument('--raw-world-axes',default='proto',help='proto: match existing raw SOMA -> .motion converter; or signed XYZ permutation')
    p.add_argument('--foot-contact-order',help='Explicit order for raw NPZ foot_contacts; otherwise infer contacts')
    p.add_argument('--limit-motions',type=int,help='Process only the first N selected motions, for an explicit bounded validation')
    p.add_argument('--list-inputs',action='store_true',help='Print input count and first/last path without loading motions or running IK')
    p.add_argument('--start-frame',type=int,default=0)
    p.add_argument('--max-frames',type=int)
    p.add_argument('--subsample-factor',type=int,default=1)
    p.add_argument('--align-ground',action='store_true')
    p.add_argument('--ground-height',type=float)
    p.add_argument('--contact-height',type=float,default=.08)
    p.add_argument('--contact-speed',type=float,default=.2)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--save-targets',action='store_true')
    p.add_argument('--skip-existing',action='store_true')
    p.add_argument('--num-shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--urdf-path',default=str(ROOT/'protomotions/data/assets/astro_p2/urdf/astro_p2_retarget.urdf'))
    p.add_argument('--mesh-dir',default=str(ROOT/'protomotions/data/assets/astro_p2/meshes'))
    args=p.parse_args()
    parents,rest=load_skeleton(args.soma_mjcf)
    if args.fit_calibration:
        if args.calibration_config is None:
            p.error('--fit-calibration requires an explicit --calibration-config output path; confirmed defaults are not refitted')
        fit_calibration(args,parents,rest);return
    if args.input is None or args.output_dir is None: p.error('--input and --output-dir are required')
    if args.start_frame<0 or args.subsample_factor<1 or (args.max_frames is not None and args.max_frames<2): p.error('Invalid frame selection')
    if args.num_shards<1 or not 0<=args.shard_index<args.num_shards: p.error('Invalid shard selection')
    if not all(np.isfinite(v) and v>0 for v in (args.contact_height,args.contact_speed)): p.error('Invalid contact thresholds')
    if args.ground_height is not None and not np.isfinite(args.ground_height): p.error('ground-height must be finite')
    cfg,params,root=load_config(args.calibration_config,args.soma_mjcf)
    keypoint_names=common.KEYPOINT_NAMES+(('neck',) if 'neck' in cfg['robot']['keypoint_links'] else ())
    if args.input.is_file():
        paths=[args.input]; input_root=args.input.parent
    elif args.input.is_dir():
        input_root=args.input/'bvh' if (args.input/'bvh').is_dir() else args.input
        paths=sorted(x for x in input_root.rglob('*') if x.suffix.lower() in ('.bvh','.npz','.motion') and args.output_dir.resolve() not in x.resolve().parents)
    else: p.error('Input not found')
    total_count=len(paths)
    paths=paths[args.shard_index::args.num_shards]
    if args.limit_motions is not None:
        if args.limit_motions<1: p.error('--limit-motions must be positive')
        paths=paths[:args.limit_motions]
    if not paths: p.error('No motions selected')
    if any(x.suffix.lower() not in ('.bvh','.npz','.motion') for x in paths): p.error('Only .bvh, .npz and .motion are supported')
    print(f'Input: {input_root}; total={total_count}; selected={len(paths)}; shard={args.shard_index}/{args.num_shards}',flush=True)
    print('Calibration: '+(str(args.calibration_config) if args.calibration_config else 'embedded user-confirmed parameters'),flush=True)
    if args.list_inputs:
        print(json.dumps(dict(input_root=str(input_root),total=total_count,selected=len(paths),first=str(paths[0]),last=str(paths[-1])),indent=2));return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('input','output_dir','skip_existing','num_shards','shard_index','limit_motions','list_inputs')}
    digest=hashlib.sha256(json.dumps(dict(options=options,calibration=cfg),sort_keys=True).encode())
    for f in (Path(__file__),Path(common.__file__),Path(__file__).with_name('batch_retarget_to_astro_p2_from_keypoints.py'),
              args.calibration_config,args.soma_mjcf,args.tpose_offsets,ROOT/'data/scripts/bvh.py',Path(args.urdf_path),
              common.CALIBRATION_PACKAGE_ROOT/'calibration.py',common.CALIBRATION_PACKAGE_ROOT/'motion.py'):
        if f is not None: digest.update(f.read_bytes())
    solver=None; failures=[]
    for path in paths:
        output=args.output_dir/path.relative_to(input_root).parent/(path.name+'_retargeted.npz')
        target=output.with_name(path.name+'_targets.npz');output.parent.mkdir(parents=True,exist_ok=True)
        stamp=f'{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}'
        signature=hashlib.sha256(digest.digest()+stamp.encode()).hexdigest()
        try:
            check=target if args.prepare_only else output
            if args.skip_existing and check.exists():
                with np.load(check,allow_pickle=False) as previous:
                    if 'signature' in previous and str(previous['signature'].item())==signature:
                        print(f'Skipping current: {path}',flush=True);continue
            targets,rotations,left,right,metadata=prepare_motion(path,args,parents,params,root,keypoint_names)
            meta_json=json.dumps(dict(metadata,calibration=cfg,calibration_source=str(args.calibration_config) if args.calibration_config else 'embedded user-confirmed calibration 2026-09-29',options=options))
            if args.prepare_only or args.save_targets:
                np.savez_compressed(target,positions=targets,orientations=rotations,left_foot_contacts=left,
                                    right_foot_contacts=right,fps=metadata['fps'],metadata_json=meta_json,signature=signature)
            if args.prepare_only:
                print(f'Prepared {len(targets)} frames: {target}',flush=True);continue
            if solver is None: solver=common.initialize_solver(args,cfg,keypoint_names)
            result,report=common.solve_motion(solver,targets,rotations,left,right,metadata['fps'],args)
            positions,quaternions,joints=result
            temp=output.with_suffix('.tmp.npz')
            np.savez_compressed(temp,retarget_version=VERSION,signature=signature,base_frame_pos=positions,
                                base_frame_wxyz=quaternions,joint_angles=joints,joint_names=np.asarray(solver[1].joints.actuated_names),
                                fps=metadata['fps'],source_fps=metadata['source_fps'],subsample_factor=args.subsample_factor,
                                source_frames=np.asarray(metadata['source_frames']),left_foot_contacts=left,right_foot_contacts=right,
                                metadata_json=meta_json)
            os.replace(temp,output)
            output.with_suffix('.report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(f'Saved {output}: {json.dumps(report)}',flush=True)
        except Exception as error:
            import traceback
            traceback.print_exc();failures.append(dict(path=str(path),error=str(error)))
    manifest=args.output_dir/f'run_shard_{args.shard_index}.json'
    manifest.write_text(json.dumps(dict(version=VERSION,selected=len(paths),failures=failures,options=options),indent=2)+'\n')
    if failures: raise SystemExit(f'{len(failures)} motions failed; see {manifest}')


if __name__=='__main__': main()
