"""Acceptance tests for in-place stepping and fixed-endpoint position hold."""
from types import SimpleNamespace
from unittest.mock import patch
import torch

from src.tasks.transition_t.entry_evaluation import EntryTrials
from src.tasks.transition_t.rewards import SustainedStandingReward
from src.tasks.transition_t.standing import standing_conditions, endpoint_metrics


def values(n=1, xy=.01):
  return {name: torch.full((n,), value) for name, value in dict(
    orientation_error=.1, height_error=.01, xy_error=xy, feet_rms=.3,
    linear_speed=.6, angular_speed=1.2, joint_speed=25.,
    rear_supported=False, front_clear=False).items()}


def test_stepping_and_speed_do_not_interrupt_exactly_one_second():
  trials = EntryTrials(1, .02, device='cpu')
  completed = torch.ones(1, dtype=torch.bool); healthy = ~completed
  for _ in range(49):
    trials.record(completed, values(), healthy, healthy)
    assert not trials.finished.any()
  trials.record(completed, values(), healthy, healthy)
  assert trials.standing_held.all()
  assert abs(trials.records()[0]['time'] - 1.) < 1e-9


def test_xy_drift_resets_continuous_window_even_with_good_contacts():
  trials = EntryTrials(1, .02, device='cpu')
  completed = torch.ones(1, dtype=torch.bool); healthy = ~completed
  m = values(); m['rear_supported'][:] = True
  for _ in range(25): trials.record(completed, m, healthy, healthy)
  m['xy_error'][:] = .081
  trials.record(completed, m, healthy, healthy)
  assert float(trials.standing_duration[0]) == 0.
  m['xy_error'][:] = .01
  for _ in range(49): trials.record(completed, m, healthy, healthy)
  assert not trials.finished.any()
  trials.record(completed, m, healthy, healthy)
  assert trials.standing_held.all()


def test_each_acceptance_bound_and_diagnostic_contact_independence():
  m = values()
  assert all(bool(v.all()) for v in standing_conditions(m).values())
  for name, bad in [('xy_error', .081), ('height_error', .051), ('orientation_error', .251)]:
    m = values(); m[name][:] = bad
    assert not torch.stack(tuple(standing_conditions(m).values()), -1).all()


def test_fixed_endpoint_xy_cannot_follow_robot_drift():
  quat = torch.tensor([[1., 0., 0., 0.]])
  rotation = torch.tensor([[2.**-.5,0.,0.,2.**-.5]])
  origins = torch.tensor([[2.,3.,0.]])
  motion = SimpleNamespace(cfg=SimpleNamespace(body_names=['base_link','FL_foot','FR_foot','RL_foot','RR_foot']),
    motion=SimpleNamespace(time_step_total=2, body_pos_w=torch.tensor([[[0.,0.,0.]],[[.2,.1,.5]]])),
    motion_anchor_body_index=0, reference_rotation=rotation,
    reference_translation=torch.tensor([[3.,4.,0.]]),
    anchor_pos_w=torch.tensor([[3.,4.,.3]]), robot_anchor_pos_w=torch.tensor([[5.,7.2,.5]]),
    anchor_quat_w=quat, robot_anchor_quat_w=quat,
    robot_anchor_lin_vel_w=torch.zeros(1,3), robot_anchor_ang_vel_w=torch.zeros(1,3),
    body_pos_w=torch.zeros(1,5,3), body_pos_relative_w=torch.zeros(1,5,3))
  robot=SimpleNamespace(find_bodies=lambda *a,**k: ([0,1,2,3],None),
    data=SimpleNamespace(body_link_pos_w=torch.zeros(1,4,3),joint_vel=torch.zeros(1,12)))
  contact=SimpleNamespace(data=SimpleNamespace(force=torch.zeros(1,4,3),found=torch.zeros(1,4,dtype=torch.bool)))
  scene=Scene({'robot':robot,'feet_ground_contact':contact}, origins)
  env=SimpleNamespace(num_envs=1,scene=scene,
    command_manager=SimpleNamespace(get_term=lambda _:motion))
  robot.data.body_link_pos_w[:] = torch.tensor([2.,3.2,0.])
  m=endpoint_metrics(env)
  torch.testing.assert_close(m['xy_error'],torch.tensor([.1]),atol=1e-6,rtol=0.)
  # The cached relative target belongs to an old command update. Diagnostics
  # must align the same physical sample without mutating that cache.
  for leg in ('FL','FR','RL','RR'):
    torch.testing.assert_close(m[f'{leg}_aligned_error'],torch.zeros(1),atol=1e-6,rtol=0.)
  motion.robot_anchor_pos_w[:,0] += 1.
  torch.testing.assert_close(endpoint_metrics(env)['xy_error'],torch.tensor([1.1]),atol=1e-6,rtol=0.)


class Scene(dict):
  def __init__(self,entities,origins):
    super().__init__(entities)
    self.env_origins=origins


def test_reward_is_smooth_position_shaping_independent_of_contact_or_timer():
  from src.tasks.transition_t import rewards
  motion=SimpleNamespace(time_steps=torch.tensor([1]),motion=SimpleNamespace(time_step_total=2))
  env=SimpleNamespace(num_envs=1,device='cpu',common_step_counter=1,step_dt=.02,
    command_manager=SimpleNamespace(get_term=lambda _:motion),
    termination_manager=SimpleNamespace(terminated=torch.zeros(1,dtype=torch.bool)))
  reward=SustainedStandingReward(None,env)
  with patch.object(rewards,'endpoint_metrics',return_value=values(xy=.08)), \
    patch.object(rewards.terminations,'physical_failure',return_value=torch.zeros(1,dtype=torch.bool)):
    torch.testing.assert_close(reward(env),torch.exp(torch.tensor([-1.])))
    env.common_step_counter += 1
    torch.testing.assert_close(reward(env),torch.exp(torch.tensor([-1.])))
  env.common_step_counter += 1
  with patch.object(rewards,'endpoint_metrics',return_value=values(xy=.081)), \
    patch.object(rewards.terminations,'physical_failure',return_value=torch.zeros(1,dtype=torch.bool)):
    assert 0. < float(reward(env)[0]) < .368


def test_radius_sensitivity_and_window_diagnostics():
  completed=torch.ones(1,dtype=torch.bool); healthy=~completed
  for radius, succeeds in ((.05,False),(.08,True),(.10,True)):
    trials=EntryTrials(1,.02,device='cpu',position_radius=radius,timeout_s=1.)
    m=values(xy=.07)
    m['horizontal_speed']=torch.tensor([.4])
    m['rear_any_contact']=torch.tensor([True])
    m['FR_aligned_error']=torch.tensor([.03])
    for _ in range(50): trials.record(completed,m,healthy,healthy)
    assert bool(trials.standing_held[0]) == succeeds
    d=trials.records()[0]['diagnostics']
    assert d['post_reference_samples'] == 50
    assert d['rear_both_support_fraction'] == 0.
    assert d['rear_any_contact_fraction'] == 1.
    assert abs(d['horizontal_speed_mean']-.4) < 1e-6
    assert abs(d['FR_aligned_error_mean']-.03) < 1e-6
    assert d['failed_gate_samples']['position'] == (0 if succeeds else 50)
