"""Gradient evidence and balanced losses do not alter training state."""
import torch

def test_opposed_and_zero_gradients_and_weighted_norms_without_side_effects():
    from src.tasks.legmanip_distil.supermimic.diagnostics import GradientMonitor
    p=torch.nn.Parameter(torch.tensor([1.,2.]))
    monitor=GradientMonitor()
    metrics=monitor.measure({'a':p.sum(),'b':-2*p.sum(),'zero':p.sum()*0}, [('rnn.weight',p)])
    assert abs(metrics['all/cosine/a__b']+1)<1e-6
    assert metrics['all/conflict_rate/a__b']==1
    assert 'all/cosine/a__zero' not in metrics
    assert abs(metrics['all/norm/b']/metrics['all/norm/a']-2)<1e-6
    assert p.grad is None
    (p.square().sum()).backward()
    torch.testing.assert_close(p.grad,torch.tensor([2.,4.]))

def test_task_balancing_is_by_actual_frames_not_episode_counts():
    from src.tasks.legmanip_distil.supermimic.diagnostics import task_balanced_mean
    error=torch.tensor([[1.,3.,9.],[1.,3.,9.]],requires_grad=True)
    task=torch.tensor([[0,0,1],[0,0,1]])
    result=task_balanced_mean(error,task)
    assert result==5.5
    result.backward()
    assert error.grad[0,2]==2*error.grad[0,0]

def test_model_validation_uses_disjoint_envs_and_preserves_terminal_mask():
    from src.tasks.legmanip_distil.supermimic.training import TransitionReplay
    replay=TransitionReplay(8)
    for t in range(4):
        s=torch.zeros(4,37);s[:,3]=1
        replay.add(dict(state=s,next_state=s,targets=torch.zeros(4,4,12),
                        contacts=torch.zeros(4,4),collision=torch.zeros(4,1),
                        owner=torch.zeros(4,dtype=torch.long),task=torch.arange(4),
                        done=torch.tensor([False,False,False,t==1])))
    train=replay.sample(100,4,allowed_envs=torch.tensor([0,1,2]))
    valid=replay.sample(100,4,allowed_envs=torch.tensor([3]))
    assert (train['env_ids']<3).all()
    assert (valid['env_ids']==3).all()
    assert valid['valid'][:2].all() and not valid['valid'][2:].any()

def test_teacher_probability_has_permanent_floor():
    from src.tasks.legmanip_distil.supermimic.training import SuperMimicAlgorithm
    obj=object.__new__(SuperMimicAlgorithm)
    obj.teacher_probability=None;obj.warmup_updates=1000;obj.decay_updates=2000;obj.num_updates=60000
    obj.teacher_probability_floor=.1
    assert obj.beta==.1

def test_lambda_returns_cut_invalid_future_without_losing_last_valid_bootstrap():
    from src.tasks.legmanip_distil.supermimic.learning import real_lambda_returns
    reward=torch.tensor([[1.],[1000.],[2.]])
    values=torch.tensor([[10.],[20.],[30.]])
    valid=torch.tensor([[True],[False],[False]])
    target=real_lambda_returns(reward,values,torch.zeros(3,1,dtype=torch.bool),valid,.9,.95)
    torch.testing.assert_close(target[0],torch.tensor([10.]))

def test_replay_boundary_does_not_change_physical_terminal_flags():
    from src.tasks.legmanip_distil.supermimic.training import TransitionReplay
    replay=TransitionReplay(4)
    for t in range(3):
        s=torch.zeros(1,37);s[:,3]=1
        replay.add(dict(state=s,next_state=s,targets=torch.zeros(1,4,12),
                        contacts=torch.zeros(1,4),collision=torch.zeros(1,1),owner=torch.zeros(1,dtype=torch.long),
                        done=torch.tensor([False]),boundary=torch.tensor([t==0])))
    sample=replay.sample(2,3)
    assert sample['valid'][0].all() and not sample['valid'][1:].any()
    assert not sample['done'].any()

def test_horizon_gate_requires_valid_endpoint_evidence():
    from types import SimpleNamespace
    from src.tasks.legmanip_distil.supermimic.learning import validate_dynamics
    class Replay:
        steps=[0]*4
        def sample(self,*args,**kwargs):
            s=torch.zeros(4,2,37);s[:,:,3]=1
            return dict(state=s,next_state=s,targets=torch.zeros(4,2,4,12),
                        contacts=torch.zeros(4,2,4),owner=torch.zeros(4,2,dtype=torch.long),
                        valid=torch.tensor([[True,True],[False,False],[False,False],[False,False]]))
    class Model:
        delta_scale=torch.ones(36)
        def __call__(self,s,t):
            return dict(state=s,contact_logits=torch.full((len(s),4),-20.))
    a=SimpleNamespace(replay=Replay(),world_model=Model(),validation_envs=torch.tensor([1]),
        sm_cfg=SimpleNamespace(max_horizon=4,horizon=1,validation_batch_size=2,validation_error_scale=.5,reliable_validations=3),
        validation_scores=torch.zeros(6),contact_scores=torch.zeros(6),
        validation_counts=torch.full((6,),3),active_horizon=1)
    validate_dynamics(a)
    assert a.active_horizon==1 and a.validation_counts[0]==0

def test_imagination_obeys_clock_and_scenario_end():
    from types import SimpleNamespace
    from src.tasks.legmanip_distil.supermimic.learning import imagined_termination
    intent=SimpleNamespace(finished=torch.tensor([False,True,False]),timed_out=torch.zeros(3,dtype=torch.bool))
    env=SimpleNamespace(episode_length_buf=torch.tensor([4,1,5]),step_dt=.02,
        cfg=SimpleNamespace(episode_length_s=.1),
        command_manager=SimpleNamespace(get_term=lambda _:intent))
    assert imagined_termination(env).tolist()==[False,True,True]

def test_imagined_done_matches_pre_command_scenario_and_ceil_clock():
    from types import SimpleNamespace
    from src.tasks.legmanip_distil.supermimic.learning import imagined_termination
    intent=SimpleNamespace(finished=torch.tensor([True]),timed_out=torch.tensor([False]))
    env=SimpleNamespace(episode_length_buf=torch.tensor([5]),step_dt=.02,
        cfg=SimpleNamespace(episode_length_s=.101),
        command_manager=SimpleNamespace(get_term=lambda _:intent))
    assert not imagined_termination(env,scenario_done_before=torch.tensor([False])).any()
    assert imagined_termination(env,scenario_done_before=torch.tensor([True])).all()
    env.episode_length_buf[:]=6
    assert imagined_termination(env,scenario_done_before=torch.tensor([False])).all()

def test_missing_task_cannot_reuse_old_validation_successes():
    from types import SimpleNamespace
    from src.tasks.legmanip_distil.supermimic.learning import validate_dynamics
    class Replay:
        steps=[0]*4
        def sample(self,*args,**kwargs):
            s=torch.zeros(4,2,37);s[:,:,3]=1
            return dict(state=s,next_state=s,targets=torch.zeros(4,2,4,12),
                        contacts=torch.zeros(4,2,4),owner=torch.zeros(4,2,dtype=torch.long),
                        valid=torch.ones(4,2,dtype=torch.bool))
    class Model:
        delta_scale=torch.ones(36)
        def __call__(self,s,t):
            return dict(state=s,contact_logits=torch.full((len(s),4),-20.))
    a=SimpleNamespace(replay=Replay(),world_model=Model(),validation_envs=torch.tensor([1]),
        sm_cfg=SimpleNamespace(max_horizon=4,horizon=1,validation_batch_size=2,validation_error_scale=.5,reliable_validations=3),
        validation_scores=torch.zeros(6),contact_scores=torch.zeros(6),
        validation_counts=torch.full((6,),3),active_horizon=1)
    validate_dynamics(a)
    assert not a.validation_counts[1:].any()
    assert a.active_horizon==1
