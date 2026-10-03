"""Fused quadruped/tripod/biped task on the stand spec snapshot.

One configuration wires the two source tasks together: the loco_pedipulation
phase machine and rewards (LOCO group), the Pedipulation rear-stand rewards
(BIPED group), and shared terms. The command term switches the active group by
commanded FR target height with a 10 cm overlap band; see doc/LEG_MANIP.md.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

from src.tasks.pedipulation.mdp import events as pedi_events
from src.tasks.pedipulation.mdp.actions import PedipulationPositionActionCfg
from src.tasks.pedipulation.mdp.terminations import base_contact
from src.tasks.pedipulation.robot import NONFOOT_BODIES, get_stand_robot_cfg
from src.tasks.velocity import mdp as velocity_mdp

from .constants import STAGE_BIPED
from .mdp import curriculums, observations, rewards, terminations, events
from .mdp.commands import LegManipCommandCfg

FOOT_NAMES = ('FL', 'FR', 'RL', 'RR')


def _contact_sensor(name, bodies):
  return ContactSensorCfg(name=name,
    primary=ContactMatch(mode='body', pattern=tuple(bodies), entity='robot'),
    fields=('force', 'found'), reduce='netforce', num_slots=1)


def leg_manip_env_cfg(play=False):
  reward_terms = {}
  for name, func, weight, params in (
    # Shared terms: active in both groups.
    ('fr_position_tracking', rewards.fr_position_tracking, 2.5, {'std': .05}),
    ('fr_ground_contact', rewards.fr_ground_contact, -1., {}),
    ('is_terminated', rewards.is_terminated, -200., {}),
    ('joint_acc_l2', rewards.joint_acc_l2, -2.5e-7, {}),
    ('action_rate_l2', rewards.action_rate_l2, -.05, {}),
    ('joint_pos_limits', rewards.joint_pos_limits, -10., {}),
    ('torques', rewards.torques, -.0002, {}),
    ('collision', rewards.collision, -2., {}),
    ('alive', rewards.alive, 1., {}),
    # LOCO group: quadruped and tripod, weighted by 1 - w.
    # Preserve tripod's effective weight: 2.5 * 0.8 = 2.0.
    ('track_linear_velocity', rewards.track_linear_velocity, 2.5,
      {'std': .5, 'tripod_std': .15, 'tripod_weight': .8, 'min_std': .05, 'speed_ratio': .75}),
    ('track_angular_velocity', rewards.track_angular_velocity, 1., {}),
    ('pose', rewards.mode_posture, 1., {
      'asset_cfg': SceneEntityCfg('robot', joint_names='.*'),
      'command_name': 'twist',
      'std_standing': {r'.*(FR|FL|RR|RL)_hip_joint.*': .05,
                       r'.*(FR|FL|RR|RL)_thigh_joint.*': .1,
                       r'.*(FR|FL|RR|RL)_calf_joint.*': .15},
      'std_walking': {r'.*(FR|FL|RR|RL)_hip_joint.*': .15,
                      r'.*(FR|FL|RR|RL)_thigh_joint.*': .35,
                      r'.*(FR|FL|RR|RL)_calf_joint.*': .5},
      'std_running': {r'.*(FR|FL|RR|RL)_hip_joint.*': .15,
                      r'.*(FR|FL|RR|RL)_thigh_joint.*': .35,
                      r'.*(FR|FL|RR|RL)_calf_joint.*': .5},
      'walking_threshold': .1,
      'running_threshold': 1.5,
    }),
    ('stand_still', rewards.stand_still, -1., {}),
    ('foot_gait', rewards.feet_gait, .5, {}),
    ('foot_clearance', rewards.foot_clearance, -1., {}),
    ('foot_slip', rewards.foot_slip, -.25, {}),
    ('soft_landing', rewards.soft_landing, -.001, {}),
    ('support_contact', rewards.support_contact, .5, {}),
    ('base_height', rewards.base_height, .5, {}),
    ('body_orientation_l2', rewards.body_orientation_l2, -1.,
      {'asset_cfg': SceneEntityCfg('robot', body_names=('base_link',))}),
    ('body_ang_vel', rewards.body_ang_vel, -.05,
      {'asset_cfg': SceneEntityCfg('robot', body_names=('base_link',))}),
    ('angular_momentum', rewards.angular_momentum, -.025, {'sensor_name': 'robot/root_angmom'}),
    # BIPED group: rear-leg stand, weighted by w. biped_base_height writes the
    # per-environment height gate and must stay first in this group.
    ('biped_base_height', rewards.biped_base_height, 1.5, {}),
    ('biped_tracking_lin_vel', rewards.biped_tracking_lin_vel, 2.5,
      {'min_std': .05, 'speed_ratio': .75}),
    ('biped_tracking_ang_vel', rewards.biped_tracking_ang_vel, 2.5, {}),
    ('biped_lin_vel_z', rewards.biped_lin_vel_z, .3, {}),
    ('biped_ang_vel_xy', rewards.biped_ang_vel_xy, .2, {}),
    ('handstand_orientation', rewards.handstand_orientation, -1., {}),
    ('handstand_feet_on_air', rewards.handstand_feet_on_air, .4, {}),
    ('handstand_feet_height_exp', rewards.handstand_feet_height_exp, 5.,
      {'asset_cfg': SceneEntityCfg('robot', site_names=('stand_goal',), preserve_order=True)}),
    ('biped_feet_clearance', rewards.biped_feet_clearance, .4,
      {'asset_cfg': SceneEntityCfg('robot', site_names=('RL', 'RR'), preserve_order=True)}),
    ('biped_contact', rewards.biped_contact, .3, {}),
    ('biped_ang_xz', rewards.biped_ang_xz, -.5, {}),
    ('symmetric_joints', rewards.symmetric_joints, -.1, {}),
    ('orientation_symmetry', rewards.orientation_symmetry, -.5, {}),
    ('feet_height_symmetry', rewards.feet_height_symmetry, -.2,
      {'asset_cfg': SceneEntityCfg('robot', site_names=('FL', 'FR'), preserve_order=True)}),
    ('default_pos_front', rewards.default_pos_front, -.1, {}),
    ('default_pos_rear', rewards.default_pos_rear, -.1, {}),
    ('default_hip_pos', rewards.default_hip_pos, -.1, {}),
    ('default_pos_reward_FL', rewards.default_pos_reward_FL, .5, {}),
    ('default_pos_reward_FR', rewards.default_pos_reward_FR, .5, {}),
  ):
    reward_terms[name] = RewardTermCfg(func=func, weight=weight, params=params)

  event_terms = {
    'physics': EventTermCfg(func=pedi_events.randomize_physics, mode='startup'),
    'reset_base': EventTermCfg(func=velocity_mdp.reset_root_state_uniform, mode='reset',
      params={'pose_range': {'x': (-.5, .5), 'y': (-.5, .5), 'z': (0., 0.), 'yaw': (-3.14, 3.14)},
              'velocity_range': {}}),
    'reset_robot_joints': EventTermCfg(func=velocity_mdp.reset_joints_by_offset, mode='reset',
      params={'position_range': (0., 0.), 'velocity_range': (0., 0.),
              'asset_cfg': SceneEntityCfg('robot', joint_names=('.*',))}),
    'push_robot': EventTermCfg(func=events.push_robot, mode='interval',
      interval_range_s=(10., 15.),
      params={'velocity_range': {'x': (-.15, .15), 'y': (-.15, .15), 'yaw': (-.2, .2)}}),
  }
  if play:
    event_terms.pop('push_robot')

  foot_geoms = tuple(f'{name}_foot_collision' for name in FOOT_NAMES)
  sensors = (
    ContactSensorCfg(name='feet_ground_contact',
      primary=ContactMatch(mode='geom', pattern=foot_geoms, entity='robot'),
      secondary=ContactMatch(mode='body', pattern='terrain'),
      fields=('found', 'force'), reduce='netforce', num_slots=1, track_air_time=True),
    ContactSensorCfg(name='nonfoot_ground_touch',
      primary=ContactMatch(mode='geom', pattern=r'.*_collision\d*$', entity='robot',
                           exclude=foot_geoms),
      secondary=ContactMatch(mode='body', pattern='terrain'),
      fields=('found', 'force'), reduce='none', num_slots=1, history_length=4),
    _contact_sensor('front_contact', ('FL_foot', 'FR_foot')),
    _contact_sensor('rear_contact', ('RL_foot', 'RR_foot')),
    _contact_sensor('nonfoot_contact', NONFOOT_BODIES),
    _contact_sensor('base_contact', ('base_link',)),
  )

  return ManagerBasedRlEnvCfg(
    seed=42,
    scene=SceneCfg(num_envs=1 if play else 4096, env_spacing=3.,
      entities={'robot': get_stand_robot_cfg()},
      terrain=TerrainEntityCfg(terrain_type='plane', collisions=(CollisionCfg(
        geom_names_expr=('terrain',), friction=(1., .005, .0001),
        contype=1, conaffinity=1, condim=3, priority=0),)),
      sensors=sensors),
    actions={'joint_pos': PedipulationPositionActionCfg(entity_name='robot')},
    observations={
      'actor': ObservationGroupCfg(terms={
        'policy': ObservationTermCfg(func=observations.policy_state, params={'add_noise': not play}),
      }),
      'critic': ObservationGroupCfg(terms={
        'policy': ObservationTermCfg(func=observations.shared_policy_state),
        'base_lin_vel': ObservationTermCfg(func=observations.base_velocity),
        'domain_parameters': ObservationTermCfg(func=observations.domain_parameters),
        'foot_contact': ObservationTermCfg(func=velocity_mdp.foot_contact,
          params={'sensor_name': 'feet_ground_contact'}),
        'gait_phase': ObservationTermCfg(func=observations.gait_phase),
        'foot_control': ObservationTermCfg(func=observations.foot_control_state),
        'foot_height': ObservationTermCfg(func=velocity_mdp.foot_height,
          params={'asset_cfg': SceneEntityCfg('robot', site_names=FOOT_NAMES, preserve_order=True)}),
        'foot_air_time': ObservationTermCfg(func=velocity_mdp.foot_air_time,
          params={'sensor_name': 'feet_ground_contact'}),
        'foot_contact_forces': ObservationTermCfg(func=velocity_mdp.foot_contact_forces,
          params={'sensor_name': 'feet_ground_contact'}),
      }),
    },
    commands={'twist': LegManipCommandCfg(debug_vis=play,
      initial_stage=STAGE_BIPED if play else 0, curriculum_enabled=not play)},
    events=event_terms,
    rewards=reward_terms,
    terminations={
      'time_out': TerminationTermCfg(func=velocity_mdp.time_out, time_out=True),
      'base_contact': TerminationTermCfg(func=base_contact),
      'illegal_contact': TerminationTermCfg(func=velocity_mdp.illegal_contact,
        params={'sensor_name': 'nonfoot_ground_touch', 'force_threshold': 10.}),
      'fell_over': TerminationTermCfg(func=terminations.fell_over,
        params={'limit_angle': math.radians(60.)}),
      'landing_failed': TerminationTermCfg(func=rewards.landing_failed),
    },
    metrics={
      'manipulation_fraction': MetricsTermCfg(func=rewards.manipulation_fraction),
      'biped_fraction': MetricsTermCfg(func=rewards.biped_fraction),
    },
    curriculum={} if play else {
      'manipulation': CurriculumTermCfg(func=curriculums.manipulation_curriculum)},
    sim=SimulationCfg(nconmax=96, njmax=512, contact_sensor_maxmatch=128,
      mujoco=MujocoCfg(timestep=.005, iterations=20, ls_iterations=10)),
    decimation=4,
    episode_length_s=1e9 if play else 20.,
    scale_rewards_by_dt=True,
    viewer=ViewerConfig(entity_name='robot', body_name='base_link',
      origin_type=ViewerConfig.OriginType.ASSET_BODY, distance=2., elevation=-10.),
  )
