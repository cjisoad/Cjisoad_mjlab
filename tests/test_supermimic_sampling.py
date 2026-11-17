"""Coverage allocation, coherent inherited resets, and real-frame accounting."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch


def make_sampler(monkeypatch, n=120):
    from src.tasks.legmanip_distil.supermimic.sampling import CoverageIntent, CoverageIntentCfg
    import src.tasks.legmanip_distil.supermimic.sampling as sampling
    monkeypatch.setattr(sampling, '_validated_high_paths', lambda offsets,nominal,zero: offsets, raising=False)
    import src.tasks.leg_manip.mdp.foot_workspace as workspace
    low = np.array([[x,y,z] for x in (-.10,0,.10) for y in (-.07,0,.07) for z in (.08,.16,.27)])
    high = np.array([[x,y,z] for x in (.02,.12,.24) for y in (-.08,0,.08) for z in (.32,.48,.66)])
    low=np.concatenate([low,[[0.,0.,.35]]])
    high=np.concatenate([high,[[.12,0.,.25]]])
    monkeypatch.setattr(workspace, 'build_foot_workspace', lambda: SimpleNamespace(
        nominal_biped_offset=np.array([.1,0,.45]), zero=np.zeros(3),
        offsets=np.concatenate([low,high]), segment=np.repeat([0,1],[len(low),len(high)])))
    default_root = torch.zeros(n,13); default_root[:,2]=.3; default_root[:,3]=1
    robot = SimpleNamespace(data=SimpleNamespace(default_root_state=default_root,
        default_joint_pos=torch.zeros(n,12)), roots=default_root.clone(), joints=torch.zeros(n,12))
    def write_root(value,env_ids): robot.roots[env_ids]=value
    def write_joints(pos,vel,env_ids): robot.joints[env_ids]=pos
    robot.write_root_state_to_sim=write_root
    robot.write_joint_state_to_sim=write_joints
    robot.clear_state=lambda env_ids: None
    action=SimpleNamespace(executed_position=torch.zeros(n,12),executed_velocity=torch.zeros(n,12),
        _raw_action=torch.zeros(n,12),previous_action=torch.zeros(n,12),last_request=torch.zeros(n,12),joint_ids=slice(None))
    action.raw_action=action._raw_action
    env=SimpleNamespace(num_envs=n,device='cpu',step_dt=.02,common_step_counter=0,
        scene={'robot':robot},cfg=SimpleNamespace(commands={}))
    class Scene(dict): pass
    env.scene=Scene(env.scene); env.scene.env_origins=torch.zeros(n,3)
    env.action_manager=SimpleNamespace(get_term=lambda name:action,action=torch.zeros(n,12))
    term=CoverageIntent(CoverageIntentCfg(),env)
    # A deterministic coherent bank witness avoids disk/motion dependencies in unit tests.
    bank_root=default_root.clone(); bank_root[:,2]=.55
    term.entry_bank=SimpleNamespace(sample=lambda count,**kwargs:torch.arange(count), data={
        'root_state':bank_root,'joint_pos':torch.full((n,12),.2),'joint_vel':torch.full((n,12),.3),
        'raw_action':torch.full((n,12),.4),'previous_action':torch.full((n,12),.5),
        'limiter_position':torch.full((n,12),.6),'limiter_velocity':torch.full((n,12),.7)})
    return term,env


def test_training_uses_coverage_but_play_and_legacy_scenarios_stay_available():
    from src.tasks.legmanip_distil.supermimic.env import supermimic_env_cfg
    from src.tasks.legmanip_distil.supermimic.sampling import CoverageIntentCfg
    from src.tasks.legmanip_distil.intent import IntentCommandCfg
    assert isinstance(supermimic_env_cfg().commands['intent'],CoverageIntentCfg)
    assert supermimic_env_cfg().commands['intent'].coverage_enabled
    play=supermimic_env_cfg(play=True).commands['intent']
    assert type(play) is IntentCommandCfg and not play.automatic


def test_all_tasks_have_independent_durations_coherent_biped_reset_and_stage(monkeypatch):
    from src.tasks.legmanip_distil.supermimic.sampling import TASK_NAMES, TARGET_FRACTIONS
    from src.tasks.legmanip_distil.intent import STAGES
    torch.manual_seed(3)
    term,env=make_sampler(monkeypatch,600)
    term._resample_command(torch.arange(env.num_envs))
    assert TASK_NAMES==('quad_walk','biped_walk','tripod_operation','biped_operation','combined','transition')
    assert tuple(TARGET_FRACTIONS)==(.2,.2,.2,.15,.15,.1)
    assert term.task_id.dtype==torch.long and set(term.task_id.tolist())==set(range(6))
    pure=term.task_id!=5
    assert ((term.task_duration[pure]>=10)&(term.task_duration[pure]<=20)).all()
    assert (term.command[:,1]==0).all()
    assert not term.twist[:,1].any()
    assert (term.starts_biped[term.task_id==1]).all()
    assert (term.starts_biped[term.task_id==3]).all()
    assert not term.starts_biped[(term.task_id==0)|(term.task_id==2)|(term.task_id==5)].any()
    combined=term.starts_biped[term.task_id==4].float().mean()
    assert .3<float(combined)<.7
    biped=term.starts_biped
    assert (term.entry_indices[biped]>=0).all()
    assert (env.scene['robot'].roots[biped,2]==.55).all()
    assert (env.action_manager.get_term('joint_pos').executed_position[biped]==.6).all()
    names=term.stage_name_ids
    assert (names[term.task_id==0]==STAGES.index('quad_walk')).all()
    assert (names[term.task_id==1]==STAGES.index('biped_walk')).all()
    assert (names[term.task_id==2]==STAGES.index('tripod_operation')).all()
    assert (names[term.task_id==3]==STAGES.index('biped_operation')).all()
    assert not term.command[term.task_id==2,:3].any()
    assert not term.command[term.task_id==3,:3].any()


def test_real_frames_are_counted_once_and_deficits_raise_allocation(monkeypatch):
    term,env=make_sampler(monkeypatch,600)
    term._resample_command(torch.arange(600))
    term.task_id[:]=0
    env.common_step_counter+=1
    term.record_frames(); term.record_frames()
    assert term.actual_frames.tolist()==[600,0,0,0,0,0]
    weights=term.allocation_weights()
    assert torch.isfinite(weights).all() and (weights>0).all()
    assert weights[0]<weights[1]
    term._resample_command(torch.arange(600))
    assert (term.task_id==0).sum()<(term.task_id==1).sum()
    metrics=term.coverage_metrics()
    assert metrics['achieved/quad_walk']==1.
    assert all(isinstance(v,float) for v in metrics.values())
    state=term.state_dict(); term.set_training_progress(9000)
    term.load_state_dict(state)
    assert term.training_progress.item()==0
    assert term.actual_frames.tolist()==[600,0,0,0,0,0]


def test_curriculum_has_stops_backward_turns_and_retains_easy_at_full_progress(monkeypatch):
    term,env=make_sampler(monkeypatch,6000)
    term.set_training_progress(10000)
    term._resample_command(torch.arange(6000))
    walking=(term.task_id==0)|(term.task_id==1)|(term.task_id==4)
    command=term.command[walking,:3]
    assert (command[:,0]>=-.2).all() and (command[:,0]<=.6).all()
    assert (command[:,2]>=-1).all() and (command[:,2]<=1).all()
    assert ((command[:,0]==0)&(command[:,2]==0)).any()
    assert (command[:,0]<0).any() and (command[:,2]<0).any() and (command[:,2]>0).any()
    assert term.easy_episode[walking].float().mean()>=.15
    assert not command[:,1].any()


def test_operation_holds_then_smoothly_retargets_high_to_high_and_finishes(monkeypatch):
    term,env=make_sampler(monkeypatch,600)
    term._resample_command(torch.arange(600))
    operation=(term.task_id==3)&~term.r_return_episode
    assert operation.any()
    assert ((term.foot_hold[operation]>=2)&(term.foot_hold[operation]<=5)).all()
    assert (term.foot_source[operation,2]>.25).all()
    assert (term.foot_goal[operation,2]>.25).all()
    first=term.command.clone()
    for _ in range(50):
        env.common_step_counter+=1; term._update_command()
    torch.testing.assert_close(term.command[operation,3:],first[operation,3:])
    max_change=0.
    for _ in range(350):
        old=term.command.clone(); env.common_step_counter+=1; term._update_command()
        max_change=max(max_change,float((term.command[operation,3:]-old[operation,3:]).abs().max()))
    assert max_change<.025
    assert not torch.equal(term.command[operation,3:],first[operation,3:])
    term.task_elapsed[operation]=term.task_duration[operation]
    env.common_step_counter+=1; term._update_command()
    assert term.finished[operation].all()


def test_legacy_cycle_is_exact_and_partial_reset_keeps_other_intent(monkeypatch):
    from src.tasks.legmanip_distil.intent import STAGES
    term,env=make_sampler(monkeypatch,4)
    term.cfg.scenario='cycle'
    term._resample_command(torch.arange(4))
    assert (term.scenario==2).all() and (term.stage_name_ids==STAGES.index('settle')).all()
    term.elapsed[:]=.6
    env.common_step_counter+=1; term._update_command()
    assert (term.stage_name_ids==STAGES.index('lift')).all()
    before=term.command[1:].clone()
    term._resample_command(torch.tensor([0]))
    torch.testing.assert_close(term.command[1:],before)


def test_stratification_and_failures_count_actual_frames_without_shadow_updates(monkeypatch):
    from src.tasks.legmanip_distil.intent import STAGES
    term,env=make_sampler(monkeypatch,1200)
    term.set_training_progress(10000)
    term._resample_command(torch.arange(1200))
    # Real distributions use the actual commanded target, including a held point.
    env.common_step_counter+=1
    term.record_frames()
    before=term.actual_frames.clone()
    failed=torch.zeros(1200,dtype=torch.bool); failed[0]=True
    task=int(term.task_id[0]); stage=int(term.stage_name_ids[0])
    term.record_failure(failed); term.record_failure(failed)
    assert term.failure_stage_counts.sum()==1
    assert term.failure_stage_counts[task,stage]==1
    for _ in range(100):
        env.common_step_counter+=1; term._update_command()
    torch.testing.assert_close(term.actual_frames,before)
    assert term.workspace_frames[0].sum()>0 and term.workspace_frames[1].sum()>0
    assert (term.workspace_frames[0]>0).all() and (term.workspace_frames[1]>0).all()
    metrics=term.coverage_metrics()
    assert metrics['failure_stage/'+('quad_walk','biped_walk','tripod_operation','biped_operation','combined','transition')[task]+'/'+STAGES[stage]]==1.


def test_commands_change_on_independent_clocks_and_repeated_compute_does_not_advance(monkeypatch):
    term,env=make_sampler(monkeypatch,600)
    term._resample_command(torch.arange(600))
    before=term.command.clone()
    term._update_command(); term._update_command()
    torch.testing.assert_close(term.command,before)
    assert not term.task_elapsed.any()
    walking=(term.task_id==0)|(term.task_id==1)|(term.task_id==4)
    for _ in range(225):
        env.common_step_counter+=1; term._update_command()
    assert not torch.equal(term.command[walking,:3],before[walking,:3])
    assert ((term.command_duration>=1)&(term.command_duration<=4))[term.task_id!=5].all()
    assert not term.command[:,1].any()


def test_disabled_sampler_keeps_original_script_and_state_validates_shape(monkeypatch):
    from src.tasks.legmanip_distil.intent import STAGES
    term,env=make_sampler(monkeypatch,20)
    term.cfg.coverage_enabled=False
    term.cfg.scenario='biped'
    term._resample_command(torch.arange(20))
    assert term.starts_biped.all()
    assert (term.stage_name_ids==STAGES.index('biped_settle')).all()
    assert not term.allocated_episodes.any()
    with pytest.raises(ValueError,match='shape mismatch'):
        term.load_state_dict({'actual_frames':torch.zeros(5)})


def test_real_environment_task_resets_teacher_ownership_and_pre_reset_capture():
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.legmanip_distil.supermimic.env import supermimic_env_cfg
    cfg=supermimic_env_cfg(); cfg.scene.num_envs=8
    env=ManagerBasedRlEnv(cfg,device='cpu')
    try:
        term=env.command_manager.get_term('intent')
        for task in range(6):
            weights=torch.zeros(6); weights[task]=1.
            term.allocation_weights=lambda weights=weights:weights
            env.reset()
            assert (term.task_id==task).all()
            owner=env.command_manager.get_term('twist').owner
            assert (owner[term.starts_biped]==3).all()
            assert (owner[~term.starts_biped]==0).all()
            assert (term.entry_indices[term.starts_biped]>=0).all()
            frames=term.actual_frames.sum().item()
            env.step(torch.zeros(8,12))
            assert term.actual_frames.sum().item()==frames+8
            assert torch.isfinite(term.command).all()
            assert not term.command[:,1].any()
    finally:
        env.close()


def test_allocation_uses_realized_episode_lengths_to_repair_short_failure_tasks(monkeypatch):
    term,_=make_sampler(monkeypatch,10)
    # Both tasks achieved their frame targets, but task 0 fails after one frame.
    # It needs more allocations than task 1 to keep the same achieved coverage.
    term.actual_frames[:]=torch.tensor([2000,2000,2000,1500,1500,1000])
    term.allocated_episodes[:]=torch.tensor([2000,20,20,15,15,10])
    weights=term.allocation_weights()
    assert weights[0]>weights[1]*4



def test_easy_foot_targets_remain_near_canonical_after_curriculum_widens(monkeypatch):
    term,env=make_sampler(monkeypatch,6000)
    term.set_training_progress(10000)
    term._resample_command(torch.arange(6000))
    for high in (False,True):
        eligible=((term.task_id==2)|(term.task_id==3)|(term.task_id==4))&(term.starts_biped==high)
        easy=eligible&term.easy_episode
        full=eligible&~term.easy_episode
        center=term.nominal if high else torch.tensor([0.,0.,.1])
        assert easy.any() and full.any()
        easy_dist=(term.foot_goal[easy]-center).norm(dim=-1)
        full_dist=(term.foot_goal[full]-center).norm(dim=-1)
        assert easy_dist.mean()<full_dist.mean()*.85
    term.set_training_progress(0)
    ids=torch.arange(6000)
    term._resample_command(ids)
    initial=(term.foot_goal[~term.starts_biped]-torch.tensor([0.,0.,.1])).norm(dim=-1)
    term.set_training_progress(10000); term._resample_command(ids)
    final=(term.foot_goal[~term.starts_biped]-torch.tensor([0.,0.,.1])).norm(dim=-1)
    assert final.mean()>initial.mean()*1.15


def test_tripod_and_combined_quad_end_with_smooth_zero_return(monkeypatch):
    from src.tasks.legmanip_distil.intent import STAGES
    term,env=make_sampler(monkeypatch,600)
    term._resample_command(torch.arange(600))
    tail=((term.task_id==2)|(term.task_id==4))&~term.starts_biped
    term.task_elapsed[tail]=term.task_duration[tail]-3.
    env.common_step_counter+=1; term._update_command()
    assert (term.stage_name_ids[tail]==STAGES.index('return')).all()
    assert not term.command[tail,:3].any()
    for _ in range(149):
        env.common_step_counter+=1; term._update_command()
    torch.testing.assert_close(term.command[tail],torch.zeros_like(term.command[tail]),atol=1e-6,rtol=0.)
    assert term.finished[tail].all()


def test_workspace_counts_exclude_locomotion_nominal_and_transition_stages(monkeypatch):
    term,env=make_sampler(monkeypatch,100)
    term._resample_command(torch.arange(100))
    term.task_id[:]=1
    term.starts_biped[:]=True
    term._command[:,:3]=0.; term._command[:,3:]=term.nominal
    env.common_step_counter+=1; term.record_frames()
    assert not term.workspace_frames.any()


def test_operation_retarget_passes_through_its_certified_hub(monkeypatch):
    term,env=make_sampler(monkeypatch,600)
    term.set_training_progress(10000)
    term._resample_command(torch.arange(600))
    operation=(term.task_id==2)|(term.task_id==3)|(term.task_id==4)
    ids=operation.nonzero().flatten()
    term.foot_elapsed[ids]=term.foot_hold[ids]+term.foot_motion[ids]*.5
    term._write_direct_commands(ids)
    high=ids[term.starts_biped[ids]]
    low=ids[~term.starts_biped[ids]]
    torch.testing.assert_close(term.command[high,3:],term.nominal.expand(len(high),-1),atol=1e-6,rtol=0.)
    torch.testing.assert_close(term.command[low,3:],torch.zeros(len(low),3),atol=1e-6,rtol=0.)


def test_high_path_certification_rejects_unreachable_target_and_reuses_cache():
    from src.tasks.leg_manip.mdp.foot_workspace import build_foot_workspace
    from src.tasks.legmanip_distil.supermimic.sampling import _validated_high_paths
    workspace=build_foot_workspace()
    nominal=tuple(map(float,workspace.nominal_biped_offset))
    zero=tuple(map(float,workspace.zero))
    candidates=(nominal,(nominal[0],nominal[1],5.))
    kept=_validated_high_paths(candidates,nominal,zero)
    assert kept==(nominal,)
    hits=_validated_high_paths.cache_info().hits
    assert _validated_high_paths(candidates,nominal,zero)==kept
    assert _validated_high_paths.cache_info().hits==hits+1


def test_quad_return_has_long_zero_hold_and_waits_for_lower_completion(monkeypatch):
    term,env=make_sampler(monkeypatch,600)
    term._resample_command(torch.arange(600))
    tail=(term.task_id==2)|((term.task_id==4)&~term.starts_biped)
    teacher=SimpleNamespace(owner=torch.zeros(600,dtype=torch.long),phase=torch.ones(600,dtype=torch.long))
    env.cfg.commands={'twist':None}; env.command_manager=SimpleNamespace(get_term=lambda _:teacher)
    term.task_elapsed[tail]=term.task_duration[tail]-3.
    env.common_step_counter+=1; term._update_command()
    term.task_elapsed[tail]=term.task_duration[tail]-2.2
    env.common_step_counter+=1; term._update_command()
    assert not term.command[tail].any()
    term.task_elapsed[tail]=term.task_duration[tail]
    env.common_step_counter+=1; term._update_command()
    assert not term.finished[tail].any()
    teacher.phase[:]=0
    env.common_step_counter+=1; term._update_command()
    assert term.finished[tail].all()


def test_biped_keyboard_r_tail_waits_for_safe_down_return_or_timeout(monkeypatch):
    from src.tasks.legmanip_distil.intent import STAGES
    term,env=make_sampler(monkeypatch,600)
    term._resample_command(torch.arange(600))
    assert term.r_return_episode.any()
    biped=term.starts_biped&(term.task_id!=5)
    term.r_return_episode[biped]=True
    teacher=SimpleNamespace(owner=torch.full((600,),3,dtype=torch.long),phase=torch.ones(600,dtype=torch.long))
    env.cfg.commands={'twist':None}; env.command_manager=SimpleNamespace(get_term=lambda _:teacher)
    term.task_elapsed[biped]=term.task_duration[biped]-3.
    env.common_step_counter+=1; term._update_command()
    assert not term.command[biped].any()
    assert (term.stage_name_ids[biped]==STAGES.index('return')).all()
    term.task_elapsed[biped]=term.task_duration[biped]
    env.common_step_counter+=1; term._update_command()
    assert not term.finished[biped].any()
    ids=biped.nonzero().flatten(); ready=ids[:len(ids)//2]; stuck=ids[len(ids)//2:]
    teacher.owner[ready]=0; teacher.phase[ready]=0
    env.common_step_counter+=1; term._update_command()
    assert term.finished[ready].all() and not term.finished[stuck].any()
    term.task_elapsed[stuck]=term.task_duration[stuck]+5.1
    env.common_step_counter+=1; term._update_command()
    assert term.timed_out[stuck].all()


def test_turns_in_place_and_combined_tripod_supported_velocity_limits(monkeypatch):
    term,env=make_sampler(monkeypatch,6000)
    term.set_training_progress(10000); term._resample_command(torch.arange(6000))
    walk=(term.task_id==0)|(term.task_id==1)
    velocity=term.command[walk,:3]
    assert ((velocity[:,0]==0)&(velocity[:,2]>0)).any()
    assert ((velocity[:,0]==0)&(velocity[:,2]<0)).any()
    tripod=(term.task_id==4)&~term.starts_biped
    assert (term.command[tripod,0].abs()<=.35).all()
    assert (term.command[tripod,2].abs()<=.4).all()
    assert (velocity[:,0]>.35).any() and (velocity[:,2].abs()>.4).any()


def test_original_banks_survive_fixed_legacy_scenarios_and_task_labels_follow_stage(monkeypatch):
    from src.tasks.legmanip_distil.intent import STAGES
    term,env=make_sampler(monkeypatch,100)
    # The inherited parent bank has no path certification filter applied.
    assert len(term.high_bank)==28 and len(term.low_bank)==28
    term.cfg.scenario='quad'; term._resample_command(torch.arange(100))
    assert (term.task_id==0).all()
    term.elapsed[:]=.6; env.common_step_counter+=1; term._update_command()
    assert (term.stage_name_ids==STAGES.index('quad_walk')).all()
    assert (term.task_id==0).all()
    term.cfg.scenario='biped'; term._resample_command(torch.arange(100))
    assert (term.task_id==1).all()
    term.stage[:]=2; term.elapsed[:]=0.
    env.common_step_counter+=1; term._update_command()
    assert (term.task_id==3).all()


def test_actual_stage_and_owner_frames_are_separate_from_episode_task_quotas(monkeypatch):
    from src.tasks.legmanip_distil.intent import STAGES
    term,env=make_sampler(monkeypatch,100)
    term._resample_command(torch.arange(100))
    env.cfg.commands={'twist':None}
    teacher=SimpleNamespace(owner=torch.tensor([0,2,3,4]).repeat(25))
    env.command_manager=SimpleNamespace(get_term=lambda _:teacher)
    term.task_id[:]=1
    term.scenario[:]=1; term.stage[:]=3  # DOWN physical stage within biped task.
    env.common_step_counter+=1; term.record_frames()
    assert term.stage_frames.sum()==term.actual_frames.sum()==100
    assert term.owner_frames.sum()==term.actual_frames.sum()
    metrics=term.coverage_metrics()
    assert metrics['achieved/biped_walk']==1.
    assert metrics['stage/down']==1.
    assert metrics['owner/down']==.25
    state=term.state_dict()
    assert 'stage_frames' in state and 'owner_frames' in state
