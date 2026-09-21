"""CPU tests of the source task's mathematical and layout contract."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from stand_test_support import import_stand


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
  path = ROOT / relative
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class StandContractTests(unittest.TestCase):
  def test_task_modules_exist(self):
    for name in ('rewards', 'observations', 'actions', 'commands', 'events'):
      self.assertTrue((ROOT / f'src/tasks/stand/mdp/{name}.py').exists(), name)

  def test_reward_math_matches_source(self):
    path = ROOT / 'src/tasks/stand/mdp/rewards.py'
    if not path.exists():
      self.fail('Rear-stand reward terms have not been migrated')
    rewards = load_module('stand_reward_test', 'src/tasks/stand/mdp/rewards.py')
    robot = SimpleNamespace(data=SimpleNamespace(
      projected_gravity_b=torch.tensor([[-1., 0., 0.], [0., 0., -1.]]),
      root_link_pos_w=torch.tensor([[0., 0., .52], [0., 0., .32]]),
      root_link_lin_vel_b=torch.tensor([[0., 0., -.3], [.2, .1, .1]]),
      root_link_ang_vel_b=torch.tensor([[.4, 0., 0.], [0., .2, -.3]]),
      site_pos_w=torch.tensor([[[0., 0., .67], [0., 0., .67]],
                               [[0., 0., .5], [0., 0., .6]]]),
    ))
    state = SimpleNamespace(height_score=torch.tensor(0.))
    env = SimpleNamespace(scene={'robot': robot}, num_envs=2, device='cpu',
      action_manager=SimpleNamespace(get_term=lambda _: state),
      command_manager=SimpleNamespace(get_command=lambda _: torch.tensor([[.3, 0., .4], [0., 0., 0.]])))
    torch.testing.assert_close(rewards.handstand_orientation(env), torch.tensor([0., 2.]))
    expected = torch.exp(-torch.tensor([0., .2]) * 5)
    torch.testing.assert_close(rewards.base_height(env), expected)
    torch.testing.assert_close(state.height_score, expected.mean())
    torch.testing.assert_close(rewards.tracking_lin_vel(env), torch.zeros(2))
    state.height_score = torch.tensor(.8)
    torch.testing.assert_close(rewards.tracking_lin_vel(env), torch.exp(-torch.tensor([0., .02]) / .25))
    torch.testing.assert_close(rewards.handstand_feet_height_exp(env, SimpleNamespace(name='robot', site_ids=[0, 1])), torch.exp(-torch.tensor([0., .24]) * 10))

  def test_configuration_contract(self):
    task = import_stand()
    cfg = task.stand_env_cfg()
    self.assertEqual(cfg.scene.num_envs, 4096)
    self.assertEqual(cfg.scene.terrain.terrain_type, 'plane')
    self.assertEqual(cfg.curriculum, {})
    self.assertEqual(cfg.decimation * cfg.sim.mujoco.timestep, .02)
    self.assertEqual(cfg.episode_length_s, 20.)
    self.assertEqual(list(cfg.rewards), sorted(cfg.rewards))
    self.assertEqual(len(cfg.rewards), 23)
    self.assertEqual(cfg.rewards['handstand_feet_height_exp'].weight, 5.)
    self.assertEqual(cfg.rewards['collision'].weight, -2.)
    self.assertTrue(cfg.terminations['time_out'].time_out)
    self.assertEqual(cfg.events['physics'].mode, 'startup')
    self.assertNotIn('push_robot', task.stand_env_cfg(play=True).events)

  def test_observation_order_and_shared_noise(self):
    import_stand()
    from src.tasks.stand.mdp import observations
    joint = torch.arange(12).float().reshape(1, 12)
    state = SimpleNamespace(joint_pos=joint, default_angles=joint * 0,
      joint_vel=joint + 1, raw_action=joint - 5,
      dr_observation=torch.arange(34).float().reshape(1, 34))
    robot = SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=torch.tensor([[4., 8., 12.]]),
      projected_gravity_b=torch.tensor([[-1., 0., 0.]]),
      root_link_lin_vel_b=torch.tensor([[.1, .2, .3]])))
    sensor = SimpleNamespace(data=SimpleNamespace(force=torch.tensor([[[0., 0., 2.], [0., 0., 0.]]])))
    env = SimpleNamespace(scene={'robot': robot, 'front_contact': sensor, 'rear_contact': sensor},
      action_manager=SimpleNamespace(get_term=lambda _: state),
      command_manager=SimpleNamespace(get_command=lambda _: torch.tensor([[.5, 0., 1.]])))
    clean = observations.policy_state(env, add_noise=False).clone()
    torch.testing.assert_close(clean[:, :9], torch.tensor([[1., 2., 3., -1., 0., 0., 1., 0., .25]]))
    torch.testing.assert_close(clean[:, 9:21], joint)
    torch.testing.assert_close(clean[:, 21:33], (joint + 1) * .05)
    torch.testing.assert_close(clean[:, 33:], joint - 5)
    noisy = observations.policy_state(env, add_noise=True)
    critic = torch.cat((observations.base_velocity(env), observations.shared_policy_state(env),
      observations.domain_parameters(env), observations.foot_contact(env)), -1)
    self.assertEqual(tuple(critic.shape), (1, 86))
    torch.testing.assert_close(critic[:, 3:48], noisy, rtol=0, atol=0)
    amplitude = clean.new_tensor((.05,) * 6 + (0.,) * 3 + (.01,) * 12 + (.075,) * 12 + (0.,) * 12)
    self.assertTrue(torch.all((noisy - clean).abs() <= amplitude + 1.e-6))

  def test_delay_switch_and_single_scaling(self):
    import_stand()
    from src.tasks.stand.mdp.actions import StandPositionAction, StandPositionActionCfg
    from src.tasks.stand.constants import JOINT_NAMES
    targets = []
    entity = SimpleNamespace(find_joints=lambda *args, **kwargs: (list(range(12)), JOINT_NAMES),
      data=SimpleNamespace(default_joint_pos=torch.ones(4, 12), joint_pos=torch.ones(4, 12), joint_vel=torch.zeros(4, 12)),
      set_joint_position_target=lambda target, **kwargs: targets.append(target.clone()))
    env = SimpleNamespace(num_envs=4, device='cpu', scene={'robot': entity}, cfg=SimpleNamespace(decimation=4))
    action = StandPositionAction(StandPositionActionCfg(entity_name='robot'), env)
    action.process_actions(torch.full((4, 12), 2.))
    action.delay_steps[:, 0] = torch.arange(4)
    action.motor_offsets.fill_(.01)
    for i in range(4):
      action.apply_actions()
      expected = torch.where(torch.arange(4)[:, None] <= i, 1.51, 1.01).expand(4, 12)
      torch.testing.assert_close(targets[-1], expected)
    action.reset(torch.tensor([2]))
    self.assertEqual(action.raw_action[2, 0].item(), 2.)
    self.assertEqual(action.raw_action[1, 0].item(), 2.)
    action.process_actions(torch.full((4, 12), 3.))
    self.assertEqual(action.previous_action[2, 0].item(), 2.)

  def test_heading_and_source_zero_mask_behavior(self):
    import_stand()
    from src.tasks.stand.mdp.commands import StandCommand, StandCommandCfg
    n = 2048
    quat = torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1)
    env = SimpleNamespace(num_envs=n, device='cpu', common_step_counter=0,
      step_dt=.02, episode_length_buf=torch.zeros(n, dtype=torch.long),
      scene={'robot': SimpleNamespace(data=SimpleNamespace(root_link_quat_w=quat))})
    command = StandCommand(StandCommandCfg(resampling_time_range=(10., 10.)), env)
    command._resample_command(torch.arange(n))
    command._update_command()
    self.assertTrue(torch.all(command.command[:, 0] >= -.2))
    self.assertTrue(torch.all(command.command[:, 0] <= .6))
    self.assertTrue(torch.all((command.command[:, 0] == 0) | (command.command[:, 0].abs() > .1)))
    self.assertTrue(torch.all(command.command[:, 1] == 0))
    torch.testing.assert_close(command.command[:, 2], (.5 * command.heading_target).clamp(-1, 1))
    command.heading_target.fill_(2.)
    command._command.zero_()
    command._update_command()
    self.assertTrue(torch.all(command.command[:, 2] == 1.))

  def test_command_resampling_uses_episode_steps_before_rewards(self):
    import_stand()
    from src.tasks.stand.mdp.commands import StandCommand, StandCommandCfg
    env = SimpleNamespace(num_envs=2, device='cpu', common_step_counter=1, step_dt=.02,
      episode_length_buf=torch.tensor([499, 500]),
      scene={'robot': SimpleNamespace(data=SimpleNamespace(root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1)))})
    command = StandCommand(StandCommandCfg(resampling_time_range=(10., 10.)), env)
    command.heading_target.fill_(1.)
    before = command.command.clone()
    self.assertEqual(command.command_counter.tolist(), [0, 1])
    self.assertAlmostEqual(before[0, 2].item(), .5, places=6)
    command.compute(.02)
    self.assertEqual(command.command_counter.tolist(), [0, 1])
    torch.testing.assert_close(before, command.command)

  def test_timeout_strict_boundary_and_contact_threshold(self):
    terms = load_module('stand_termination_test', 'src/tasks/stand/mdp/terminations.py')
    env = SimpleNamespace(episode_length_buf=torch.tensor([999, 1000, 1001]), max_episode_length=1000,
      scene={'base_contact': SimpleNamespace(data=SimpleNamespace(force=torch.tensor([[[1., 0., 0.]], [[0., 1.01, 0.]], [[0., 0., 0.]]])))})
    self.assertEqual(terms.time_out(env).tolist(), [False, False, True])
    self.assertEqual(terms.base_contact(env).tolist(), [False, True, False])

  def test_push_overwrites_only_requested_velocities_on_exact_steps(self):
    import_stand()
    from src.tasks.stand.mdp.events import push_robot
    writes = []
    initial = torch.full((2, 6), 4.)
    robot = SimpleNamespace(data=SimpleNamespace(root_link_vel_w=initial),
      write_root_link_velocity_to_sim=lambda velocity: writes.append(velocity.clone()))
    env = SimpleNamespace(scene={'robot': robot}, num_envs=2, device='cpu', step_dt=.02,
      common_step_counter=399)
    push_robot(env, None)
    self.assertFalse(writes)
    env.common_step_counter = 400
    push_robot(env, None)
    self.assertEqual(len(writes), 1)
    self.assertTrue((writes[0][:, :2].abs() <= .4).all())
    self.assertTrue((writes[0][:, 3:].abs() <= .6).all())
    torch.testing.assert_close(writes[0][:, 2], initial[:, 2])


if __name__ == '__main__':
  unittest.main()
