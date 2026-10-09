"""Limits apply to commanded trajectories, including braking and reversals."""
import importlib
import importlib.util
from types import SimpleNamespace
import unittest

import torch


def limiter_type():
  name = 'src.tasks.pedipulation.mdp.target_limiter'
  assert importlib.util.find_spec(name) is not None, 'joint target limiter is missing'
  return importlib.import_module(name).JointTargetLimiter


class JointTargetLimiterTests(unittest.TestCase):
  def test_large_steps_and_reversals_bound_velocity_and_acceleration(self):
    cls = limiter_type()
    dt = .005
    speed = torch.tensor([2., 6.], dtype=torch.float64)
    acceleration = torch.tensor([30., 120.], dtype=torch.float64)
    limiter = cls(torch.zeros(3, 2, dtype=torch.float64), speed.tolist(), acceleration.tolist(), dt)
    last_position, last_velocity = limiter.position.clone(), limiter.velocity.clone()
    for index in range(400):
      target = torch.full((3, 2), 10. if index < 80 else -10., dtype=torch.float64)
      position = limiter.step(target).clone()
      velocity = (position-last_position)/dt
      torch.testing.assert_close(velocity, limiter.velocity, atol=1e-11, rtol=1e-11)
      assert (velocity.abs() <= speed+1e-10).all()
      assert ((velocity-last_velocity).abs() <= acceleration*dt+1e-10).all()
      last_position, last_velocity = position, velocity
    assert (limiter.velocity < 0).all()

  def test_fixed_target_settles_without_unbounded_small_oscillations(self):
    limiter = limiter_type()(torch.zeros(1, 2, dtype=torch.float64), (2., 6.), (30., 120.), .005)
    target = torch.tensor([[.037, -.713]], dtype=torch.float64)
    for _ in range(400):
      limiter.step(target)
    torch.testing.assert_close(limiter.position, target, atol=1e-7, rtol=0.)
    torch.testing.assert_close(limiter.velocity, torch.zeros_like(target), atol=1e-6, rtol=0.)

  def test_random_jumps_and_crossing_current_position_keep_acceleration_bound(self):
    limiter = limiter_type()(torch.zeros(16, 12), (2.,)*12, (30.,)*12, .005)
    generator = torch.Generator().manual_seed(122)
    for _ in range(400):
      previous = limiter.velocity.clone()
      target = torch.randn(16, 12, generator=generator)*2.
      limiter.step(target)
      assert (limiter.velocity.abs() <= 2.+1e-6).all()
      assert ((limiter.velocity-previous).abs() <= .15+1e-6).all()

  def test_partial_restore_preserves_other_worlds_and_initial_velocity(self):
    limiter = limiter_type()(torch.ones(3, 2), (2., 6.), (30., 120.), .005)
    limiter.step(torch.full((3, 2), 4.))
    before_position, before_velocity = limiter.position.clone(), limiter.velocity.clone()
    limiter.reset(torch.tensor([[.3, -.4]]), torch.tensor([[.2, -.5]]), env_ids=torch.tensor([1]))
    torch.testing.assert_close(limiter.position[[0, 2]], before_position[[0, 2]])
    torch.testing.assert_close(limiter.velocity[[0, 2]], before_velocity[[0, 2]])
    torch.testing.assert_close(limiter.position[1], torch.tensor([.3, -.4]))
    torch.testing.assert_close(limiter.velocity[1], torch.tensor([.2, -.5]))

  def test_none_limits_pass_support_targets_through_while_limiting_fr(self):
    limiter = limiter_type()(torch.zeros(2, 3), (None, 2., None), (None, 30., None), .005)
    first = torch.tensor([[3., 3., -4.], [-2., -2., 5.]])
    torch.testing.assert_close(limiter.step(first)[:, [0, 2]], first[:, [0, 2]])
    assert limiter.velocity[:, 1].abs().max() <= .15+1e-6
    second = -first
    torch.testing.assert_close(limiter.step(second)[:, [0, 2]], second[:, [0, 2]])
    assert limiter.position[:, 1].abs().max() < .01
    contract = limiter.contract()
    assert contract['velocity_limits'] == [None, 2., None]
    assert contract['acceleration_limits'] == [None, 30., None]

  def test_invalid_limits_timestep_and_nonfinite_inputs_fail_before_state_changes(self):
    cls = limiter_type()
    for limits, acceleration, dt in (((0., 2.), (30., 30.), .005),
        ((2.,), (30., 30.), .005), ((2., 2.), (30., float('nan')), .005),
        ((2., 2.), (30., 30.), 0.)):
      with self.subTest(limits=limits, acceleration=acceleration, dt=dt), self.assertRaises(ValueError):
        cls(torch.zeros(1, 2), limits, acceleration, dt)
    limiter = cls(torch.zeros(1, 2), (2., 2.), (30., 30.), .005)
    with self.assertRaises(ValueError):
      limiter.step(torch.tensor([[float('nan'), 1.]]))
    torch.testing.assert_close(limiter.position, torch.zeros(1, 2))
    with self.assertRaises(ValueError):
      limiter.reset(torch.zeros(1, 2), torch.tensor([[3., 0.]]))


class LimitedActionTests(unittest.TestCase):
  def action(self, *, limited=True, scale=.25):
    from src.tasks.pedipulation.mdp.actions import PedipulationPositionAction, PedipulationPositionActionCfg
    from src.tasks.pedipulation.constants import JOINT_NAMES
    targets = []
    robot = SimpleNamespace(find_joints=lambda *args, **kwargs: (list(range(12)), JOINT_NAMES),
      data=SimpleNamespace(default_joint_pos=torch.ones(2, 12), joint_pos=torch.ones(2, 12),
        joint_vel=torch.zeros(2, 12)),
      set_joint_position_target=lambda target, **kwargs: targets.append(target.clone()))
    manager = SimpleNamespace(action=torch.zeros(2, 12), prev_action=torch.zeros(2, 12),
      prev_prev_action=torch.zeros(2, 12))
    env = SimpleNamespace(num_envs=2, device='cpu', physics_dt=.005, scene={'robot': robot},
      cfg=SimpleNamespace(decimation=4), action_manager=manager)
    cfg = PedipulationPositionActionCfg(entity_name='robot', delay=False, scale=scale)
    if limited:
      cfg.target_velocity_limits = (2.,)*12
      cfg.target_acceleration_limits = (30.,)*12
    action = PedipulationPositionAction(cfg, env)
    action.reset()
    return action, robot, manager, targets

  def test_nonfinite_or_nonpositive_action_scale_is_rejected(self):
    for scale in (float('inf'), float('nan'), 0., -.25):
      with self.subTest(scale=scale), self.assertRaises(ValueError):
        self.action(scale=scale)

  def test_applied_targets_and_action_history_use_the_limited_command(self):
    action, _, manager, targets = self.action()
    action.process_actions(torch.full((2, 12), 10.))
    previous = torch.ones(2, 12)
    for _ in range(4):
      action.apply_actions()
      assert ((targets[-1]-previous).abs() <= 2.*.005+1e-6).all()
      torch.testing.assert_close(action.raw_action, (targets[-1]-action.default_angles)/.25)
      torch.testing.assert_close(manager.action, action.raw_action)
      previous = targets[-1]
    assert action.raw_action.abs().max() < 1.
    actual = action.raw_action.clone()
    action.process_actions(torch.full((2, 12), -10.))
    torch.testing.assert_close(action.previous_action, actual)

  def test_delay_holds_previous_request_without_repeatedly_braking(self):
    action, _, _, targets = self.action()
    action.cfg.delay = True
    requested = torch.full((2, 12), 40.)
    for _ in range(100):
      action.process_actions(requested)
      action.delay_steps.fill_(3)
      for _ in range(4):
        action.apply_actions()
    assert (targets[-1]-1.).min() > 3.8
    before = action.target_limiter.velocity.clone()
    action.process_actions(-requested)
    action.delay_steps.fill_(3)
    for _ in range(3):
      action.apply_actions()
      torch.testing.assert_close(action.target_limiter.velocity, before)
    action.apply_actions()
    assert (action.target_limiter.velocity < before).all()
    assert (action.target_limiter.velocity-before).abs().max() <= .15+1e-6

  def test_reset_initializes_from_final_physical_pose_and_only_resets_selected_world(self):
    action, robot, _, targets = self.action()
    action.process_actions(torch.ones(2, 12)); action.apply_actions()
    before = action.target_limiter.position[0].clone()
    action.reset(torch.tensor([1]))
    robot.data.joint_pos[1] = -1.2
    action.process_actions(torch.zeros(2, 12)); action.apply_actions()
    assert (targets[-1][1]+1.2).abs().max() < .001
    assert (targets[-1][0]-before).abs().max() < .01

  def test_disabled_action_keeps_existing_target_and_history_semantics(self):
    action, _, manager, targets = self.action(limited=False)
    action.process_actions(torch.full((2, 12), 2.)); action.apply_actions()
    torch.testing.assert_close(targets[-1], torch.full((2, 12), 1.5))
    torch.testing.assert_close(action.raw_action, torch.full((2, 12), 2.))
    torch.testing.assert_close(manager.action, torch.zeros(2, 12))


if __name__ == '__main__':
  unittest.main()
