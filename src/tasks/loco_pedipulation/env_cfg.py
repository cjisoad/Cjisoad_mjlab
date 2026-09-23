"""Flat quadruped locomotion with optional right-front foot control."""

import math

from mjlab.managers import SceneEntityCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg

from .mdp import curriculums, observations, rewards
from .mdp.commands import LocoPedipulationCommandCfg


def loco_pedipulation_env_cfg(play=False):
  cfg = unitree_go2_flat_env_cfg(play=play)
  cfg.seed = 42
  cfg.scene.num_envs = 1 if play else 4096
  cfg.scene.env_spacing = 3.
  cfg.sim.nconmax = 96
  cfg.sim.njmax = 512
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.mujoco.iterations = 20
  cfg.commands = {'twist': LocoPedipulationCommandCfg(
    debug_vis=play, initial_stage=3 if play else 0, curriculum_enabled=not play)}
  for group in ('actor', 'critic'):
    cfg.observations[group].terms['phase'] = ObservationTermCfg(func=observations.gait_phase)
    cfg.observations[group].terms['foot_control'] = ObservationTermCfg(func=observations.foot_control_state)
  cfg.observations['critic'].terms['base_lin_vel'] = ObservationTermCfg(func=observations.base_velocity)
  # Resolve site order explicitly wherever per-foot arrays are consumed.
  cfg.observations['critic'].terms['foot_height'].params['asset_cfg'] = SceneEntityCfg(
    'robot', site_names=('FL', 'FR', 'RL', 'RR'), preserve_order=True)
  cfg.rewards['pose'].func = rewards.mode_posture
  for name, func, weight, params in (
    ('track_linear_velocity', rewards.track_linear_velocity, 1.,
      {'std': .5, 'tripod_std': .15, 'tripod_weight': 2.}),
    ('track_angular_velocity', rewards.track_angular_velocity, 1., {}),
    ('stand_still', rewards.stand_still, -1., {}),
    ('foot_gait', rewards.feet_gait, .5, {}),
    ('foot_clearance', rewards.foot_clearance, -1., {}),
    ('foot_slip', rewards.foot_slip, -.25, {}),
    ('soft_landing', rewards.soft_landing, -.001, {}),
    ('support_contact', rewards.support_contact, .5, {}),
    ('fr_position_tracking', rewards.fr_position_tracking, 2.5, {'std': .05}),
    ('fr_ground_contact', rewards.fr_ground_contact, -1., {}),
    ('base_height', rewards.base_height, .5, {}),
  ):
    cfg.rewards[name] = RewardTermCfg(func=func, weight=weight, params=params)
  cfg.terminations['fell_over'].params['limit_angle'] = math.radians(60.)
  cfg.terminations['landing_failed'] = TerminationTermCfg(func=rewards.landing_failed)
  cfg.metrics['manipulation_fraction'] = MetricsTermCfg(func=rewards.manipulation_fraction)
  cfg.curriculum = {} if play else {
    'manipulation': CurriculumTermCfg(func=curriculums.manipulation_curriculum)}
  # Start with smaller disturbances while retaining the locomotion randomizations.
  cfg.events['base_com'].params['ranges'] = {axis: (-.015, .015) for axis in range(3)}
  if 'push_robot' in cfg.events:
    cfg.events['push_robot'].params['velocity_range'] = {
      'x': (-.15, .15), 'y': (-.15, .15), 'yaw': (-.2, .2)}
  return cfg
