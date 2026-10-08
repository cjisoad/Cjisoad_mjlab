"""One-shot tracking semantics and configuration contracts for Go2."""
from types import SimpleNamespace

import mujoco
import pytest
import torch

from src.tasks.pedipulation.robot import get_stand_spec
from src.tasks.transition_t.motion import BODY_NAMES, JOINT_NAMES


class Scene(dict):
  pass


class Robot:
  joint_names = JOINT_NAMES
  body_names = tuple(reversed(BODY_NAMES))

  def __init__(self, n):
    self.writes = []
    self.data = SimpleNamespace(
      body_link_pos_w=torch.zeros(n, 29, 3),
      body_link_quat_w=torch.zeros(n, 29, 4),
      body_link_lin_vel_w=torch.zeros(n, 29, 3),
      body_link_ang_vel_w=torch.zeros(n, 29, 3),
      joint_pos=torch.zeros(n, 12), joint_vel=torch.zeros(n, 12))
    self.data.body_link_quat_w[..., 0] = 1.

  def find_bodies(self, names, preserve_order=True):
    return [self.body_names.index(n) for n in names], names

  def write_root_state_to_sim(self, root, env_ids):
    self.writes.append(('root', root.clone(), env_ids.clone()))

  def write_joint_state_to_sim(self, pos, vel, env_ids):
    self.writes.append(('joint', pos.clone(), vel.clone(), env_ids.clone()))

  def clear_state(self, env_ids):
    self.writes.append(('clear', env_ids.clone()))


def make_command(n=3, **overrides):
  from src.tasks.transition_t.commands import TransitionCommand, TransitionCommandCfg
  spec = get_stand_spec()
  spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, .1])
  robot = Robot(n)
  scene = Scene(robot=robot)
  scene.env_origins = torch.zeros(n, 3)
  scene.env_origins[:, 0] = torch.arange(n) * 3.
  env = SimpleNamespace(scene=scene, num_envs=n, device='cpu', step_dt=.02,
    sim=SimpleNamespace(mj_model=spec.compile()),
    termination_manager=SimpleNamespace(terminated=torch.zeros(n, dtype=torch.bool),
      time_outs=torch.zeros(n, dtype=torch.bool)), episode_length_buf=torch.full((n,), 300))
  cfg = TransitionCommandCfg(sampling_mode='start', pose_range={}, velocity_range={},
    foot_range=(0., 0., 0.), **overrides)
  command = TransitionCommand(cfg, env)
  env.command_manager = SimpleNamespace(get_term=lambda name: command)
  return command, env, robot


def test_cfg_preserves_tracking_interface_and_go2_contract():
  from mjlab.tasks.tracking.mdp import MotionCommandCfg as FrameworkCfg
  from src.tasks.tracking.mdp.commands import MotionCommandCfg as LocalCfg
  from src.tasks.transition_t.env_cfg import transition_env_cfg, TRACKED_BODY_NAMES
  cfg = transition_env_cfg()
  command = cfg.commands['motion']
  assert isinstance(command, (FrameworkCfg, LocalCfg))
  assert isinstance(command, FrameworkCfg) and isinstance(command, LocalCfg)
  assert command.first_frame_probability == .25
  assert command.undisturbed_probability == .25
  assert len(TRACKED_BODY_NAMES) == 13
  assert cfg.decimation * cfg.sim.mujoco.timestep == .02
  assert cfg.episode_length_s == 8.
  assert cfg.actions['joint_pos'].scale == .25
  assert 'push_robot' not in cfg.events
  assert cfg.scene.entities['robot'].articulation.soft_joint_pos_limit_factor == .98
  assert cfg.rewards['motion_global_root_pos'].params['std'] == .15
  assert cfg.rewards['motion_body_pos'].params['std'] == .15
  for group in ('actor', 'critic'):
    assert 'sensor_name' not in cfg.observations[group].terms['base_lin_vel'].params
  play = transition_env_cfg(play=True)
  assert play.commands['motion'].sampling_mode == 'start'
  assert play.commands['motion'].undisturbed_probability == 1.
  assert play.episode_length_s == 8.


def test_named_mapping_and_endpoint_hold_never_write_state():
  cmd, env, robot = make_command()
  assert cmd.motion.body_pos_w.shape == (351, 13, 3)
  assert torch.equal(cmd.motion.body_pos_w[:, 0], torch.tensor(cmd.reference.body_pos_w[:, 0], dtype=torch.float32))
  cmd.time_steps[:] = 350
  for _ in range(20):
    cmd._update_command()
  assert not robot.writes
  assert (cmd.time_steps == 350).all()
  assert not cmd.joint_vel.any()
  assert not cmd.body_lin_vel_w.any()
  assert not cmd.body_ang_vel_w.any()
  assert torch.allclose(cmd.anchor_pos_w, cmd.motion.body_pos_w[-1, 0] + env.scene.env_origins)


def test_subset_reset_uses_local_sampler_then_world_origin():
  cmd, env, robot = make_command()
  cmd.time_steps[0] = 99
  cmd.reset_age_steps[:] = 17
  cmd._resample_command(torch.tensor([1]))
  assert cmd.time_steps[0] == 99
  assert cmd.reset_age_steps.tolist() == [17, 0, 17]
  root_write = next(w for w in robot.writes if w[0] == 'root')
  assert torch.allclose(root_write[1][0, :3], cmd.motion.body_pos_w[0, 0] + env.scene.env_origins[1])
  assert all(w[-1].tolist() == [1] for w in robot.writes)


def test_state_preserving_entry_aligns_only_xy_yaw_and_preserves_reset_age():
  from mjlab.utils.lab_api.math import quat_from_euler_xyz
  cmd, env, robot = make_command()
  cmd.reset_age_steps[:] = 30
  cmd.time_steps[:] = 67
  anchor = cmd.robot_anchor_body_index
  robot.data.body_link_pos_w[1, anchor] = torch.tensor([6., 2., .8])
  robot.data.body_link_quat_w[1, anchor] = quat_from_euler_xyz(torch.tensor(.3), torch.tensor(.2), torch.tensor(.6))
  before = cmd.anchor_pos_w[0].clone()
  cmd.start_from_current_state(torch.tensor([1]))
  assert not robot.writes
  assert cmd.time_steps.tolist() == [67, 0, 67]
  assert (cmd.reset_age_steps == 30).all()
  assert torch.allclose(cmd.anchor_pos_w[1, :2], torch.tensor([6., 2.]))
  assert torch.allclose(cmd.anchor_pos_w[1, 2], cmd.motion.body_pos_w[0, 0, 2])
  assert torch.equal(cmd.anchor_pos_w[0], before)
  assert not torch.allclose(cmd.anchor_quat_w[1], robot.data.body_link_quat_w[1, anchor])


def test_adaptive_sampling_counts_failures_but_not_timeouts():
  cmd, env, robot = make_command()
  env.termination_manager.terminated[:] = torch.tensor([True, False, False])
  env.termination_manager.time_outs[:] = torch.tensor([False, True, False])
  cmd.time_steps[:] = torch.tensor([10, 250, 100])
  cmd._adaptive_sampling(torch.tensor([0, 1, 2]))
  assert cmd._current_bin_failed.sum() == 1


def test_termination_grace_uses_true_reset_age_and_full_orientation():
  from src.tasks.transition_t.terminations import tracking_orientation_failure, physical_failure
  from mjlab.utils.lab_api.math import quat_from_euler_xyz
  cmd, env, robot = make_command()
  anchor = cmd.robot_anchor_body_index
  robot.data.body_link_quat_w[:, anchor] = quat_from_euler_xyz(torch.zeros(3), torch.zeros(3), torch.ones(3)*2.)
  cmd.reset_age_steps[:] = torch.tensor([0, 9, 10])
  failed = tracking_orientation_failure(env, 'motion', .8, .2)
  assert failed.tolist() == [False, False, True]
  robot.data.joint_pos[0, 0] = float('nan')
  assert physical_failure(env, 'motion')[0]


def test_control_period_must_match_reference():
  cmd, env, robot = make_command()
  env.step_dt = .01
  from src.tasks.transition_t.commands import TransitionCommand
  with pytest.raises(ValueError, match='control|period|50'):
    TransitionCommand(cmd.cfg, env)


def test_actor_and_critic_observation_shapes_are_75_and_192():
  from src.tasks.transition_t.env_cfg import transition_env_cfg
  from src.tasks.tracking.mdp import observations
  cmd, env, robot = make_command()
  # Command 24 + anchor pos3/orientation6 + velocities6 + joints24 + action12.
  actor_width = cmd.command.shape[-1] + observations.motion_anchor_pos_b(env, 'motion').shape[-1]
  actor_width += observations.motion_anchor_ori_b(env, 'motion').shape[-1] + 6 + 24 + 12
  critic_width = actor_width + observations.robot_body_pos_b(env, 'motion').shape[-1]
  critic_width += observations.robot_body_ori_b(env, 'motion').shape[-1]
  assert (actor_width, critic_width) == (75, 192)


def test_state_preserving_entry_rotates_all_reference_world_velocities():
  from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz
  cmd, env, robot = make_command()
  cmd.motion.body_lin_vel_w[0] = torch.tensor([1., 0., 0.])
  cmd.motion.body_ang_vel_w[0] = torch.tensor([0., 1., 0.])
  robot.data.body_link_quat_w[1, cmd.robot_anchor_body_index] = quat_from_euler_xyz(
    torch.tensor(0.), torch.tensor(0.), torch.tensor(1.))
  cmd.start_from_current_state(torch.tensor([1]))
  rotation = cmd.reference_rotation[1].expand(13, -1)
  assert torch.allclose(cmd.body_lin_vel_w[1], quat_apply(rotation, cmd.motion.body_lin_vel_w[0]))
  assert torch.allclose(cmd.body_ang_vel_w[1], quat_apply(rotation, cmd.motion.body_ang_vel_w[0]))
  assert not robot.writes


def test_base_contact_terminates_immediately_during_grace():
  from src.tasks.transition_t.terminations import physical_failure
  cmd, env, robot = make_command()
  force = torch.zeros(3, 3, 1, 3)
  force[0, 0, 0, 2] = 20.
  env.scene['base_ground_contact'] = SimpleNamespace(data=SimpleNamespace(force=force))
  assert cmd.reset_age_steps[0] == 0
  assert physical_failure(env).tolist() == [True, False, False]


def test_reset_frame_and_undisturbed_draw_are_independent():
  cmd, env, robot = make_command(n=128)
  cmd.cfg.sampling_mode = 'uniform'
  cmd.cfg.velocity_range = {'x': (-.1, .1)}
  # Velocity-only perturbations preserve the accepted kinematic state and need
  # no IK retries, allowing this test to check the two actual sampling draws.
  torch.manual_seed(37)
  cmd._resample_command(torch.arange(128))
  first = cmd.time_steps == 0
  disturbed = cmd.metrics['initial_disturbed'].bool()
  assert first.any() and (~first).any()
  assert (first & disturbed).any() and (first & ~disturbed).any()
  assert (~first & disturbed).any() and (~first & ~disturbed).any()


def test_reset_age_tracks_real_environment_steps_without_reset_frame_skip():
  cmd, env, robot = make_command()
  env.common_step_counter = 40
  cmd._resample_command(torch.tensor([1]))
  cmd._update_command()  # Automatic resets are followed by compute in this step.
  assert cmd.time_steps[1] == 0
  assert cmd.reset_age_steps[1] == 0
  env.common_step_counter = 50
  from src.tasks.transition_t.terminations import _active
  assert _active(env, cmd, .2)[1]
  cmd._update_command()
  assert cmd.reset_age_steps[1] == 10


def test_reset_relative_targets_use_written_pose_before_simulator_forward():
  cmd, env, robot = make_command()
  # MuJoCo body positions/quaternions retain the old episode until forward().
  robot.data.body_link_pos_w[:, cmd.robot_anchor_body_index] = torch.tensor([9., 7., .8])
  robot.data.body_link_quat_w[:, cmd.robot_anchor_body_index] = torch.tensor([0., 0., 0., 1.])
  cmd._resample_command(torch.tensor([1]))
  torch.testing.assert_close(cmd.body_pos_relative_w[1], cmd.body_pos_w[1])
  torch.testing.assert_close(cmd.body_quat_relative_w[1], cmd.body_quat_w[1])
