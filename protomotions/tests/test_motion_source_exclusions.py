"""Training blacklist behavior across roots, shards, forced resets and resume."""
import json
from types import SimpleNamespace as NS

import pytest
import torch

from protomotions.envs.motion_manager.config import MotionManagerConfig
from protomotions.envs.motion_manager.motion_manager import MotionManager
from protomotions.utils.motion_exclusions import (
    DEFAULT_GPC_EXCLUSIONS, clear_bundled_training_exclusions,
    configure_gpc_training_exclusions, source_excluded_ids,
)


def manifest(tmp_path, sources=None):
    path=tmp_path/'excluded.json'
    sources=sources or ['BONES-SEED/p2_retarget/bvh/230201/a_M.pkl']
    path.write_text(json.dumps(dict(schema_version=1,num_excluded=len(sources),
        source_path_roots=['BONES-SEED/p2_retarget/bvh'],excluded_sources=sources)))
    return str(path)


def library():
    return NS(motion_files=('/remote/BONES-SEED/p2_retarget/bvh/230201/a.pkl',
                          '/remote/BONES-SEED/p2_retarget/bvh/230201/a_M.pkl',
                          '/remote/BONES-SEED/p2_retarget/bvh/230202/a_M.pkl'),
              motion_weights=torch.ones(3), motion_lengths=torch.ones(3)*2,
              motion_file='repacked.pt', num_motions=lambda:3)


def manager(path, **kwargs):
    return MotionManager(MotionManagerConfig(exclude_motion_sources_file=path,init_start_prob=0),
                         3,.02,torch.device('cpu'),library(),**kwargs)


def test_sources_are_root_independent_and_do_not_merge_mirrors_or_same_basenames(tmp_path):
    path=manifest(tmp_path)
    assert source_excluded_ids(path,library().motion_files)==[1]
    repacked=[library().motion_files[1].replace('/remote/','/container/copied/'),library().motion_files[0]]
    assert source_excluded_ids(path,repacked)==[0]


def test_sampling_curriculum_resume_and_explicit_resets_cannot_reintroduce_failures(tmp_path):
    mm=manager(manifest(tmp_path))
    assert mm.motion_weights.tolist()==[1,0,1]
    mm.update_sampling_weights(torch.ones(3))
    mm.load_state_dict(dict(motion_file_name='repacked.pt',motion_weights=torch.ones(3)))
    assert mm.motion_weights.tolist()==[1,0,1]
    assert not (mm.sample_n_motion_ids(10000)==1).any()
    mm.sample_motions(torch.arange(3))
    assert not (mm.motion_ids==1).any()
    with pytest.raises(ValueError,match='source-blacklisted'):
        mm.sample_motions(torch.tensor([0]),torch.tensor([1]))
    with pytest.raises(ValueError,match='source-blacklisted'):
        manager(manifest(tmp_path),fixed_motion_ids_per_env=torch.tensor([0,1,-1]))


def test_missing_or_malformed_manifest_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        manager(str(tmp_path/'missing.json'))
    path=manifest(tmp_path)
    data=json.loads(open(path).read());data['num_excluded']=10
    open(path,'w').write(json.dumps(data))
    with pytest.raises(ValueError,match='inconsistent'):
        manager(path)


@pytest.mark.parametrize('target,experiment',[
    ('protomotions.agents.common.fsq.FSQAutoEncoder','custom_stage1.py'),
    ('protomotions.agents.supervised.latent_prior_model.DiscreteAutoregressiveLatentPriorModel','custom_stage2.py'),
    ('protomotions.agents.peft.actor.DiscretePriorPEFTActor','custom_stage3.py'),
    ('unused','examples/experiments/gpc/prior_reduced_coord_obs.py'),
])
def test_all_p2_gpc_stages_default_on_for_fresh_or_old_frozen_configs(target,experiment):
    # Namespace models an old unpickled config lacking the new field.
    env=NS(motion_manager=NS())
    args=NS(robot_name='p2',experiment_path=experiment)
    agent=NS(model=NS(_target_=target,actor=NS(mu_model=NS(_target_=target))))
    configure_gpc_training_exclusions(args,env,agent)
    assert env.motion_manager.exclude_motion_sources_file==DEFAULT_GPC_EXCLUSIONS
    configure_gpc_training_exclusions(args,env,agent,False)
    assert env.motion_manager.exclude_motion_sources_file is None


def test_unrelated_training_and_inference_stay_independent():
    for robot,experiment in [('g1','examples/experiments/gpc/prior.py'),('p2','examples/experiments/mimic/mlp.py')]:
        env=NS(motion_manager=MotionManagerConfig())
        configure_gpc_training_exclusions(NS(robot_name=robot,experiment_path=experiment),env,NS())
        assert env.motion_manager.exclude_motion_sources_file is None
    env=NS(motion_manager=MotionManagerConfig(exclude_motion_sources_file=DEFAULT_GPC_EXCLUSIONS))
    clear_bundled_training_exclusions(env)
    assert env.motion_manager.exclude_motion_sources_file is None


def test_bundled_manifest_contains_exactly_the_2633_failed_sources():
    from protomotions.utils.motion_exclusions import resolve_manifest_path
    data=json.loads(resolve_manifest_path(DEFAULT_GPC_EXCLUSIONS).read_text())
    assert data['num_excluded']==len(set(data['excluded_sources']))==2633
    assert sum(v for k,v in data['provenance']['counts'].items() if k.endswith('/train'))==2403


def test_every_bundled_source_matches_after_host_root_changes():
    from protomotions.utils.motion_exclusions import resolve_manifest_path
    data=json.loads(resolve_manifest_path(DEFAULT_GPC_EXCLUSIONS).read_text())
    copied=["/different/host/mount/"+source for source in data['excluded_sources']]
    assert source_excluded_ids(DEFAULT_GPC_EXCLUSIONS,copied)==list(range(2633))


def test_saved_opt_out_survives_resume_and_explicit_cli_opt_in_wins():
    env=NS(motion_manager=MotionManagerConfig())
    args=NS(robot_name='p2',experiment_path='examples/experiments/gpc/prior.py',gpc_exclude_failed_motions=False)
    configure_gpc_training_exclusions(args,env,NS())
    assert env.motion_manager.exclude_motion_sources_file is None
    configure_gpc_training_exclusions(args,env,NS(),True)
    assert env.motion_manager.exclude_motion_sources_file==DEFAULT_GPC_EXCLUSIONS
