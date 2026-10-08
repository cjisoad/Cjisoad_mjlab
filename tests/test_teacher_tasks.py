"""Physical command and training contracts for the aligned teacher pair."""
from copy import deepcopy
from dataclasses import asdict
import unittest
import torch
import numpy as np
import src.tasks
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from src.tasks.leg_manip.constants import BIPED_LOW, TRIPOD_HIGH, BIPED_HIGH, DZ_MIN
from src.tasks.leg_manip.mdp.foot_workspace import build_foot_workspace, LOW, HIGH
from src.tasks.leg_manip.mdp import observations as leg_obs
from src.tasks.pedipulation.env_cfg import pedipulation_env_cfg
from src.tasks.loco_pedipulation.env_cfg import loco_pedipulation_env_cfg
from test_leg_manip_commands import make_env

TASKS = ('pedipulation_t', 'loco_pedipulation_t')

def command_fixture(task, count=8):
  env, _ = make_env(count=count)
  env.episode_length_buf = torch.zeros(count, dtype=torch.long)
  cfg = load_env_cfg(task)
  cfg.commands['twist'].resampling_time_range = (100., 100.)
  if task == 'loco_pedipulation_t':
    cfg.commands['twist'].initial_stage = 3
  term = cfg.commands['twist'].build(env)
  env.command_manager.get_term = lambda _: term
  env.command_manager.get_command = lambda _: term.command
  term.reset(torch.arange(count))
  return env, term

class TeacherConfigurationTests(unittest.TestCase):
  def test_both_teacher_tasks_are_registered(self):
    self.assertTrue(set(TASKS).issubset(list_tasks()), 'Both aligned teachers must be registered')

  def test_rewards_preserve_remaining_source_functions_order_weights_and_parameters(self):
    for task, factory in zip(TASKS, (pedipulation_env_cfg, loco_pedipulation_env_cfg)):
      cfg, original = load_env_cfg(task), factory()
      if task == 'pedipulation_t':
        self.assertIn('ang_xz', original.rewards)
        self.assertNotIn('ang_xz', cfg.rewards)
        original.rewards.pop('ang_xz')
      expected = tuple(original.rewards)
      if task == 'loco_pedipulation_t':
        expected += ('return_contact',)
      self.assertEqual(tuple(cfg.rewards), expected)
      for name, old in original.rewards.items():
        new = cfg.rewards[name]
        self.assertEqual(new.weight, old.weight, (task, name))
        params = dict(new.params)
        if task == 'pedipulation_t':
          self.assertEqual(params.pop('reward_name'), name)
          from src.tasks.teacher_common.rewards import source_reward_function
          self.assertIs(source_reward_function(name), old.func)
        else:
          self.assertIs(new.func, old.func)
        self.assertEqual(params, old.params)

  def test_biped_training_and_play_remove_only_euler_roll_penalty(self):
    source = pedipulation_env_cfg()
    expected = tuple(name for name in source.rewards if name != 'ang_xz')
    for play in (False, True):
      cfg = load_env_cfg('pedipulation_t', play=play)
      self.assertEqual(tuple(cfg.rewards), expected)
      self.assertEqual(len(cfg.rewards), 25)
      self.assertEqual(cfg.rewards['handstand_orientation'].weight, -1.)
      self.assertEqual(cfg.rewards['orientation_symmetry'].weight, -.5)

  def test_action_physics_and_network_contract_match_teacher_settings(self):
    from src.tasks.leg_manip.env_cfg import leg_manip_env_cfg
    common = leg_manip_env_cfg()
    for task in TASKS:
      cfg, rl = load_env_cfg(task), load_rl_cfg(task)
      self.assertEqual(asdict(cfg.actions['joint_pos']), asdict(common.actions['joint_pos']))
      expected_robot = deepcopy(common.scene.entities['robot'])
      if task == 'loco_pedipulation_t':
        expected_robot.init_state.joint_pos = loco_pedipulation_env_cfg().scene.entities['robot'].init_state.joint_pos
        for actuator in expected_robot.articulation.actuators:
          actuator.armature = .02 if 'calf' in actuator.target_names_expr[0] else .01
      self.assertEqual(cfg.scene.entities['robot'], expected_robot)
      self.assertIs(cfg.events['physics'].func, common.events['physics'].func)
      self.assertEqual(rl.actor.hidden_dims, (512, 256, 128))
      self.assertFalse(rl.actor.obs_normalization)
      self.assertEqual(rl.clip_actions, 10.)
      self.assertEqual(tuple(cfg.commands), ('twist',))

  def test_training_and_play_have_explicit_noise_modes(self):
    for task in TASKS:
      training, play = load_env_cfg(task), load_env_cfg(task, play=True)
      self.assertTrue(training.observations['actor'].terms['policy'].params['add_noise'])
      self.assertFalse(play.observations['actor'].terms['policy'].params['add_noise'])
      self.assertIn('physics', training.events)
      self.assertIn('physics', play.events)
      self.assertIn('push_robot', training.events)
      self.assertNotIn('push_robot', play.events)

class TeacherCommandTests(unittest.TestCase):
  def test_workspace_keeps_the_ten_centimeter_overlap(self):
    self.assertAlmostEqual(TRIPOD_HIGH-BIPED_LOW, .10)
    for task, low, high in ((TASKS[0], BIPED_LOW, BIPED_HIGH), (TASKS[1], DZ_MIN, TRIPOD_HIGH)):
      env, term = command_fixture(task)
      bank = term.target_bank
      self.assertTrue((bank[:, 2] >= low-1e-6).all())
      self.assertTrue((bank[:, 2] <= high+1e-6).all())
      self.assertTrue(((bank[:, 2] >= BIPED_LOW)&(bank[:, 2] <= TRIPOD_HIGH)).any())
      np.testing.assert_allclose(term.zero.cpu(), build_foot_workspace().zero, atol=1e-7)

  def test_identical_overlap_command_has_identical_goal_in_all_stances(self):
    from mjlab.utils.lab_api.math import quat_from_euler_xyz
    fixtures = [command_fixture(task) for task in TASKS]
    for pitch in (0., -.35, -.9, -torch.pi/2):
      goals = []
      for env, term in fixtures:
        data = env.scene['robot'].data
        data.root_link_quat_w[:] = quat_from_euler_xyz(torch.tensor([.12]), torch.tensor([pitch]), torch.tensor([.7]))
        data.root_link_pos_w[:] = torch.tensor((2., -1., .5))
        for dz in (BIPED_LOW, .30, TRIPOD_HIGH):
          term.set_command(torch.arange(env.num_envs), velocity=(.2, -.03, .1), target_offset=(.04, .01, dz))
          goal = term.foot_target_pos_w
          expected = data.root_link_pos_w + (term.basis_w @ (term.zero+term.command[:, 3:])[...,None]).squeeze(-1)
          torch.testing.assert_close(goal, expected)
          goals.append(goal.clone())
      for k in range(3):
        torch.testing.assert_close(goals[k], goals[k+3], atol=1e-7, rtol=0)

  def test_no_operation_biped_encodes_stance_without_disabling_default_FR_reward(self):
    env, term = command_fixture('pedipulation_t', count=256)
    term.cfg.foot_target_probability = 0.
    term._resample_command(torch.arange(env.num_envs))
    torch.testing.assert_close(term.command[:, 3:], term.nominal_biped_offset.expand(env.num_envs, 3))
    self.assertFalse(term.has_foot_target.any())
    term.set_command([0], target_offset=(0., 0., .30))
    self.assertTrue(term.has_foot_target[0])
    term.set_command([0], target_offset=term.nominal_biped_offset)
    self.assertFalse(term.has_foot_target[0])

  def test_clean_actor_equals_leg_manip_and_velocity_noise_is_small(self):
    for task in TASKS:
      env, term = command_fixture(task, count=256)
      term.set_command(torch.arange(env.num_envs), velocity=(.2, .01, -.1), target_offset=(.04, .01, .30))
      env.scene['robot'].data.root_link_lin_vel_w[:] = torch.tensor((.2, -.1, .03))
      callback = load_env_cfg(task).observations['actor'].terms['policy'].func
      clean = callback(env, add_noise=False)
      source = leg_obs.policy_state(env, add_noise=False)
      actor_dim = 54 if task == 'loco_pedipulation_t' else 51
      self.assertEqual(clean.shape, (256, actor_dim))
      torch.testing.assert_close(clean[:, :51], source)
      if task == 'loco_pedipulation_t':
        from src.tasks.loco_pedipulation.mdp.observations import gait_phase
        torch.testing.assert_close(clean[:, 51:53], gait_phase(env))
        torch.testing.assert_close(clean[:, 53:54], term.blend[:, None])
      noisy = callback(env, add_noise=True)
      self.assertEqual(noisy.shape, (256, actor_dim))
      self.assertLessEqual(float((noisy[:, 48:51]-leg_obs.base_velocity(env)).abs().max()), .040001)
      torch.testing.assert_close(leg_obs.shared_policy_state(env), noisy[:, :48])

  def test_manual_velocity_is_not_overwritten_by_heading_feedback(self):
    env, term = command_fixture('pedipulation_t')
    term.set_command([0], velocity=(.2, 0., .15))
    term.compute(env.step_dt)
    torch.testing.assert_close(term.command[0, :3], torch.tensor((.2, 0., .15)))
    term.release_manual([0])
    self.assertFalse(term.manual[0])

class TeacherSymmetryTests(unittest.TestCase):
  def test_stance_and_operation_are_distinguished_after_zero_conversion(self):
    from src.tasks.teacher_common.rl import teacher_symmetry_loss
    nominal = torch.tensor(build_foot_workspace().nominal_biped_offset, dtype=torch.float32)
    model = torch.nn.Linear(51, 12)
    obs = torch.randn(8, 51)
    obs[:, 9:12] = nominal
    loss = teacher_symmetry_loss(model, obs)
    loss.backward()
    self.assertGreater(float(model.weight.grad.abs().sum()), 0.)
    obs[:, 11] += .05
    self.assertEqual(float(teacher_symmetry_loss(model, obs).detach()), 0.)


class TeacherGuiTests(unittest.TestCase):
  def test_both_teacher_command_guis_can_be_created_and_set_overlap_target(self):
    from types import SimpleNamespace
    from contextlib import nullcontext
    class Gui:
      def add_folder(self, *args): return nullcontext()
      def add_checkbox(self, label, initial_value): return SimpleNamespace(value=initial_value)
      def add_slider(self, label, **kwargs): return SimpleNamespace(value=kwargs['initial_value'])
      def add_button(self, *args, **kwargs): return SimpleNamespace(on_click=lambda callback: callback)
    for task in TASKS:
      env, term = command_fixture(task)
      term.create_gui('twist', SimpleNamespace(gui=Gui()), lambda: 0)
      manual, velocity, offset, _ = term._gui
      manual.value = True
      for field, value in zip(velocity, (.2, 0., .1)): field.value = value
      for field, value in zip(offset, (.03, .02, .3)): field.value = value
      term._read_gui()
      torch.testing.assert_close(term.command[0, 3:], torch.tensor((.03, .02, .3)))
      term.reset(torch.tensor([0]))
      term._read_gui()
      self.assertTrue(term.manual[0], 'Manual GUI must reapply unchanged controls after reset')
      torch.testing.assert_close(term.command[0, 3:], torch.tensor((.03, .02, .3)))

class TeacherRewardCompatibilityTests(unittest.TestCase):
  def test_source_stance_rewards_do_not_treat_nominal_biped_as_an_operation(self):
    from src.tasks.teacher_common.rewards import pedipulation_reward
    env, term = command_fixture('pedipulation_t')
    action = env.action_manager.get_term('joint_pos')
    action.desired_angles = torch.zeros(12)
    action.height_score = torch.tensor(1.)
    action.joint_pos[:, 3:6] = .1
    term.set_command(torch.arange(env.num_envs), target_offset=term.nominal_biped_offset)
    expected = torch.full((env.num_envs,), .3)
    torch.testing.assert_close(pedipulation_reward(env, 'default_pos_front'), expected)
    torch.testing.assert_close(pedipulation_reward(env, 'default_pos_reward_FR'), (-expected).exp())
    term.set_command(torch.arange(env.num_envs), target_offset=(0., 0., .3))
    torch.testing.assert_close(pedipulation_reward(env, 'default_pos_front'), torch.zeros(env.num_envs))
    torch.testing.assert_close(pedipulation_reward(env, 'default_pos_reward_FR'), torch.zeros(env.num_envs))


class TeacherMirrorSemanticsTests(unittest.TestCase):
  def test_mirrored_stance_command_remains_stance_despite_nonzero_nominal_dy(self):
    from src.tasks.teacher_common.rl import teacher_symmetry_loss
    nominal = torch.tensor(build_foot_workspace().nominal_biped_offset, dtype=torch.float32)
    self.assertGreater(abs(float(nominal[1])), .02)
    inputs = []
    model = torch.nn.Linear(51, 12)
    def capture(obs):
      inputs.append(obs.clone())
      return model(obs)
    obs = torch.randn(8, 51)
    obs[:, 9:12] = nominal
    teacher_symmetry_loss(capture, obs)
    torch.testing.assert_close(inputs[1][:, 9:12], nominal.expand(8, 3))
    torch.testing.assert_close(inputs[1][:, 48:51], obs[:, 48:51]*torch.tensor((1., -1., 1.)))

if __name__ == '__main__':
  unittest.main()
