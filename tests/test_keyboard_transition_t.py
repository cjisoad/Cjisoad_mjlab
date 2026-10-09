"""Real-state and interface regressions for the actor75 keyboard handoff."""
import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import unittest
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import keyboard_teacher


def backend():
  assert importlib.util.find_spec('src.tasks.transition_t.keyboard') is not None, 'transition_t keyboard backend is missing'
  return importlib.import_module('src.tasks.transition_t.keyboard')


class KeyboardTransitionTests(unittest.TestCase):
  def test_teacher_selected_action_is_the_transition_observation_history(self):
    world = self.world(); world.reset(seed=42)
    action = world.action_manager.get_term('joint_pos')
    source = action.source_teacher
    try:
      action.source_teacher = SimpleNamespace(actions=lambda env, ids: torch.full((len(ids), 12), .7))
      world.action_manager.process_action(torch.zeros(2, 12))
      torch.testing.assert_close(world.action_manager.action, action.raw_action)
      world.action_manager.process_action(torch.zeros(2, 12))
      torch.testing.assert_close(world.action_manager.prev_action, action.previous_action)
    finally:
      action.source_teacher = source

  def test_teacher_and_transition_requests_use_one_continuous_limiter(self):
    world = self.world(); world.reset(seed=42)
    action = world.action_manager.get_term('joint_pos')
    term = world.command_manager.get_term('twist')
    old = action.source_teacher
    try:
      action.source_teacher = SimpleNamespace(actions=lambda env, ids: torch.full((len(ids), 12), 4.))
      world.action_manager.process_action(torch.zeros(2, 12))
      action.delay_steps.zero_()
      for _ in range(4):
        world.action_manager.apply_action()
      first = action.target_limiter.position.clone()
      velocity = action.target_limiter.velocity.clone()
      assert (first-action.default_angles).abs().max() > 0.
      term.owner[:] = backend().TRANSITION
      world.action_manager.process_action(torch.full((2, 12), -4.))
      world.action_manager.apply_action()
      assert ((action.target_limiter.velocity-velocity).abs()
        <= action.target_limiter.acceleration_limits*world.physics_dt+1e-6).all()
      torch.testing.assert_close(world.action_manager.action,
        (action.target_limiter.position-action.default_angles-action.motor_offsets)/action.cfg.scale)
      assert (action.target_limiter.position[:, 3:6]-first[:, 3:6]).abs().max() < .031
      torch.testing.assert_close(action.target_limiter.position[:, :3], first[:, :3]-2.)
      torch.testing.assert_close(action.target_limiter.position[:, 6:], first[:, 6:]-2.)
      for _ in range(100):
        world.action_manager.apply_action()
      assert ((action.target_limiter.position-first) < 0.).all()
    finally:
      action.source_teacher = old

  def test_training_and_keyboard_share_explicit_target_limits(self):
    from src.tasks.transition_t.env_cfg import transition_env_cfg
    training = transition_env_cfg().actions['joint_pos']
    playback = backend().keyboard_transition_env_cfg().actions['joint_pos']
    assert training.target_velocity_limits == playback.target_velocity_limits
    assert training.target_acceleration_limits == playback.target_acceleration_limits
    assert training.target_velocity_limits[3:6] == (2., 2., 2.)
    assert training.target_acceleration_limits[3:6] == (30., 30., 30.)
    assert all(value is None for value in training.target_velocity_limits[:3]+training.target_velocity_limits[6:])
    assert all(value is None for value in training.target_acceleration_limits[:3]+training.target_acceleration_limits[6:])

  def world(self):
    cls = type(self)
    if not hasattr(cls, 'env'):
      module = backend()
      from mjlab.envs import ManagerBasedRlEnv
      torch.set_num_threads(1)
      cls.env = ManagerBasedRlEnv(module.keyboard_transition_env_cfg(2), device='cpu')
    return cls.env

  @classmethod
  def tearDownClass(cls):
    if hasattr(cls, 'env'):
      cls.env.close()

  def test_cli_selects_new_transition_checkpoint(self):
    args = keyboard_teacher.parse_args(['loco_pedipulation_t', '--workspace', 'union',
      '--checkpoint-file', 'loco.pt', '--transition-checkpoint-file', 'transition.pt'])
    assert args.transition_checkpoint_file == Path('transition.pt')
    assert args.bridge_checkpoint_file is None

  def test_cli_requires_explicit_opt_in_for_old_model_limiter_experiments(self):
    args = keyboard_teacher.parse_args(['loco_pedipulation_t', '--workspace', 'union',
      '--checkpoint-file', 'loco.pt', '--transition-checkpoint-file', 'old.pt',
      '--allow-legacy-transition-limiter'])
    assert args.allow_legacy_transition_limiter
    with self.assertRaises(SystemExit):
      keyboard_teacher.parse_args(['pedipulation_t', '--checkpoint-file', 'biped.pt',
        '--allow-legacy-transition-limiter'])


  def test_cli_rejects_two_different_transition_backends(self):
    with self.assertRaises(SystemExit):
      keyboard_teacher.parse_args(['loco_pedipulation_t', '--workspace', 'union',
        '--checkpoint-file', 'loco.pt', '--transition-checkpoint-file', 'new.pt',
        '--bridge-checkpoint-file', 'old.pt'])


  def test_cli_rejects_new_transition_on_single_teacher_mode(self):
    with self.assertRaises(SystemExit):
      keyboard_teacher.parse_args(['pedipulation_t', '--workspace', 'biped',
        '--checkpoint-file', 'biped.pt', '--transition-checkpoint-file', 'new.pt'])


  def test_new_backend_keeps_training_observation_and_physics_contract(self):
    module = backend()
    from src.tasks.transition_t.env_cfg import transition_env_cfg
    original = transition_env_cfg(play=True)
    cfg = module.keyboard_transition_env_cfg(2)
    assert list(cfg.observations['actor'].terms) == list(original.observations['actor'].terms)
    assert list(cfg.observations['critic'].terms) == list(original.observations['critic'].terms)
    assert cfg.scene.entities['robot'] == original.scene.entities['robot']
    assert cfg.sim == original.sim
    assert cfg.actions['joint_pos'].delay is False
    assert set(cfg.commands) == {'motion', 'twist'}
    assert 'reset_scene_to_default' in cfg.events


  def test_physical_reset_starts_in_teacher_default_not_motion_first_frame(self):
    world = self.world()
    obs, _ = world.reset(seed=42)
    robot = world.scene['robot']
    assert obs['actor'].shape == (2, 75)
    assert obs['critic'].shape == (2, 192)
    torch.testing.assert_close(robot.data.joint_pos, robot.data.default_joint_pos)
    assert not torch.allclose(robot.data.joint_pos[0], world.command_manager.get_term('motion').motion.joint_pos[0])


  def test_motion_clock_does_not_run_while_source_teacher_owns_actions(self):
    world = self.world()
    world.reset(seed=42)
    motion = world.command_manager.get_term('motion')
    for _ in range(5):
      world.common_step_counter += 1
      motion._update_command()
    assert motion.time_steps.tolist() == [0, 0]


  def test_real_state_entry_keeps_pose_velocity_and_action_history(self):
    world = self.world()
    world.reset(seed=42)
    term = world.command_manager.get_term('twist')
    motion = world.command_manager.get_term('motion')
    action = world.action_manager.get_term('joint_pos')
    action._raw_action.fill_(.23)
    action.previous_action.fill_(-.41)
    action.delay_steps.fill_(2)
    action.motor_offsets.fill_(.002)
    before = [x.clone() for x in (world.sim.data.qpos, world.sim.data.qvel,
      action.raw_action, action.previous_action, action.delay_steps, action.motor_offsets)]
    reset_clock = motion._reset_step.clone()
    term.start_transition(torch.tensor([0]))
    for expected, actual in zip(before, (world.sim.data.qpos, world.sim.data.qvel,
        action.raw_action, action.previous_action, action.delay_steps, action.motor_offsets)):
      torch.testing.assert_close(expected, actual.clone(), atol=0, rtol=0)
    torch.testing.assert_close(motion._reset_step, reset_clock)
    assert motion.time_steps.tolist() == [0, 0]
    world.common_step_counter += 1
    motion._update_command()
    assert motion.time_steps.tolist() == [1, 0]
    assert term.owner[0] == backend().TRANSITION
    assert term.owner[1] == backend().TRIPOD


  def test_teacher_observations_rebase_previous_actions_without_clipping(self):
    world = self.world()
    world.reset(seed=42)
    module = backend()
    from src.tasks.pedipulation_bridge_t.experts import BIPED_DEFAULT_ANGLES, QUADRUPED_DEFAULT_ANGLES
    action = world.action_manager.get_term('joint_pos')
    action._raw_action.fill_(9.9)
    for dim, zero in ((51, BIPED_DEFAULT_ANGLES), (54, QUADRUPED_DEFAULT_ANGLES)):
      contract = {'actor_dim': dim, 'default_angles': list(zero)}
      obs = module.teacher_observations(world, contract, torch.tensor([0, 1]))
      assert obs.shape == (2, dim)
      expected = action.raw_action + (action.default_angles-torch.tensor(zero))/.25
      torch.testing.assert_close(obs[:, 36:48], expected)
      torch.testing.assert_close(obs[:, 12:24], world.scene['robot'].data.joint_pos-torch.tensor(zero))
    assert module.teacher_observations(world,
      {'actor_dim': 51, 'default_angles': list(BIPED_DEFAULT_ANGLES)}, torch.tensor([0]))[:, 36:48].max() > 10


  def test_world_feet_drift_is_distinct_from_standing_handoff(self):
    module = backend()
    metrics = {name: torch.tensor([value]) for name, value in dict(
      orientation_error=.03, height_error=.008, feet_rms=.144,
      linear_speed=.07, angular_speed=.09, rear_supported=True, front_clear=True,
      joint_speed=1.).items()}
    gates = module.standing_conditions(metrics)
    assert torch.stack(tuple(gates.values())).all()
    assert not module.strict_endpoint_candidate(torch.tensor([True]), metrics).any()


  def test_transition_timeout_does_not_force_biped_handoff(self):
    world = self.world()
    world.reset(seed=42)
    term = world.command_manager.get_term('twist')
    term.start_transition(torch.tensor([0]))
    term.bridge_elapsed[0] = 10.
    world.common_step_counter += 1
    term.advance()
    assert term.keyboard_timed_out[0]
    assert term.owner[0] == backend().TRANSITION
    world.reset(seed=42)
    assert term.owner.tolist() == [backend().TRIPOD, backend().TRIPOD]
    assert not term.keyboard_timed_out.any()

  def test_partial_reset_does_not_skip_other_world_timeout(self):
    world = self.world()
    world.reset(seed=42)
    term = world.command_manager.get_term('twist')
    term.start_transition(torch.tensor([0]))
    term.bridge_elapsed[0] = 10.
    world.common_step_counter += 1
    term.reset(torch.tensor([1]))
    term.advance()
    assert term.keyboard_timed_out[0]

  def test_keyboard_tolerates_two_bad_samples_without_credit_or_early_handoff(self):
    module = backend()
    world = self.world(); world.reset(seed=42)
    term = world.command_manager.get_term('twist')
    motion = world.command_manager.get_term('motion')
    term.start_transition(torch.tensor([0]))
    motion.time_steps[0] = motion.motion.time_step_total-1
    term.candidate_elapsed[0] = .96
    metrics = {name: torch.tensor([value, value]) for name, value in dict(
      orientation_error=.03, height_error=.008, feet_rms=.02,
      linear_speed=.07, angular_speed=.09, rear_supported=True, front_clear=True,
      joint_speed=1.).items()}
    metrics['rear_supported'][0] = False
    with patch.object(module, 'endpoint_metrics', return_value=metrics):
      for count in (1, 2):
        world.common_step_counter += 1; term.advance()
        self.assertAlmostEqual(float(term.candidate_elapsed[0]), .96, places=6)
        self.assertEqual(int(term.candidate_bad_samples[0]), count)
        self.assertEqual(int(term.owner[0]), module.TRANSITION)
        term.advance()
        self.assertEqual(int(term.candidate_bad_samples[0]), count)
      metrics['rear_supported'][0] = True
      world.common_step_counter += 1; term.advance()
      self.assertAlmostEqual(float(term.candidate_elapsed[0]), .98, places=6)
      self.assertEqual(int(term.candidate_bad_samples[0]), 0)
      self.assertEqual(int(term.owner[0]), module.TRANSITION)
      world.common_step_counter += 1; term.advance()
      self.assertEqual(int(term.owner[0]), module.BIPED)

  def test_keyboard_third_bad_sample_resets_and_partial_reset_is_local(self):
    module = backend()
    world = self.world(); world.reset(seed=42)
    term = world.command_manager.get_term('twist')
    motion = world.command_manager.get_term('motion')
    term.start_transition(torch.tensor([0, 1]))
    motion.time_steps[:] = motion.motion.time_step_total-1
    term.candidate_elapsed[:] = .5
    metrics = {name: torch.tensor([value, value]) for name, value in dict(
      orientation_error=.03, height_error=.008, feet_rms=.02,
      linear_speed=.07, angular_speed=.09, rear_supported=False, front_clear=True,
      joint_speed=1.).items()}
    with patch.object(module, 'endpoint_metrics', return_value=metrics):
      for _ in range(2):
        world.common_step_counter += 1; term.advance()
      term.reset(torch.tensor([1]))
      self.assertEqual(int(term.candidate_bad_samples[1]), 0)
      self.assertEqual(float(term.candidate_elapsed[1]), 0.)
      self.assertEqual(int(term.candidate_bad_samples[0]), 2)
      world.common_step_counter += 1; term.advance()
      self.assertEqual(float(term.candidate_elapsed[0]), 0.)
      self.assertEqual(int(term.candidate_bad_samples[0]), 3)

  def test_keyboard_physical_failure_cannot_be_debounced_into_handoff(self):
    module = backend()
    world = self.world(); world.reset(seed=42)
    term = world.command_manager.get_term('twist')
    motion = world.command_manager.get_term('motion')
    term.start_transition(torch.tensor([0]))
    motion.time_steps[0] = motion.motion.time_step_total-1
    term.candidate_elapsed[0] = .98
    metrics = {name: torch.tensor([value, value]) for name, value in dict(
      orientation_error=.03, height_error=.008, feet_rms=.02,
      linear_speed=.07, angular_speed=.09, rear_supported=True, front_clear=True,
      joint_speed=1.).items()}
    with patch.object(module, 'endpoint_metrics', return_value=metrics), \
      patch.object(module.terminations, 'physical_failure', return_value=torch.tensor([True, False])):
      world.common_step_counter += 1; term.advance()
    self.assertEqual(int(term.owner[0]), module.TRANSITION)
    self.assertEqual(float(term.candidate_elapsed[0]), 0.)

if __name__ == '__main__':
  unittest.main()
