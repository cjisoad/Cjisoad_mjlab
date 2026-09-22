"""Go2 rear-leg stand expressed entirely through mjlab manager configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

from .constants import REWARD_WEIGHTS
from .mdp import events, observations, rewards, terminations
from .mdp.actions import PedipulationPositionActionCfg
from .mdp.commands import PedipulationCommandCfg
from .robot import get_stand_robot_cfg, NONFOOT_BODIES


def _contact_sensor(name, bodies):
  return ContactSensorCfg(name=name,
    primary=ContactMatch(mode='body', pattern=tuple(bodies), entity='robot'),
    fields=('force', 'found'), reduce='netforce', num_slots=1)


def pedipulation_env_cfg(play=False):
  reward_terms = {}
  for name, weight in sorted(REWARD_WEIGHTS.items()):
    params = {}
    if name == 'handstand_feet_height_exp':
      params['asset_cfg'] = SceneEntityCfg('robot', site_names=('stand_goal',), preserve_order=True)
    elif name == 'default_pos_reward_FR':
      params['asset_cfg'] = SceneEntityCfg('robot', site_names=('FR',), preserve_order=True)
    elif name == 'feet_height_symmetry':
      params['asset_cfg'] = SceneEntityCfg('robot', site_names=('FL', 'FR'), preserve_order=True)
    elif name == 'feet_clearance':
      params['asset_cfg'] = SceneEntityCfg('robot', site_names=('RL', 'RR'), preserve_order=True)
    reward_terms[name] = RewardTermCfg(func=getattr(rewards, name), weight=weight, params=params)

  event_terms = {
    'physics': EventTermCfg(func=events.randomize_physics, mode='startup'),
    'reset_robot': EventTermCfg(func=events.reset_robot, mode='reset'),
    'push_robot': EventTermCfg(func=events.push_robot, mode='step'),
  }
  if play:
    event_terms.pop('push_robot')

  return ManagerBasedRlEnvCfg(
    seed=1,
    scene=SceneCfg(num_envs=1 if play else 4096, env_spacing=3.,
      entities={'robot': get_stand_robot_cfg()},
      terrain=TerrainEntityCfg(terrain_type='plane', collisions=(CollisionCfg(
        geom_names_expr=('terrain',), friction=(1., .005, .0001),
        contype=1, conaffinity=1, condim=3, priority=0),)),
      sensors=(
        _contact_sensor('front_contact', ('FL_foot', 'FR_foot')),
        _contact_sensor('rear_contact', ('RL_foot', 'RR_foot')),
        _contact_sensor('nonfoot_contact', NONFOOT_BODIES),
        _contact_sensor('base_contact', ('base_link',)),
      )),
    actions={'joint_pos': PedipulationPositionActionCfg(entity_name='robot')},
    observations={
      'actor': ObservationGroupCfg(terms={
        'policy': ObservationTermCfg(func=observations.policy_state, params={'add_noise': not play}),
      }),
      'critic': ObservationGroupCfg(terms={
        'linear_velocity': ObservationTermCfg(func=observations.base_velocity),
        'policy': ObservationTermCfg(func=observations.shared_policy_state),
        'domain_parameters': ObservationTermCfg(func=observations.domain_parameters),
        'foot_contact': ObservationTermCfg(func=observations.foot_contact),
      }),
    },
    commands={'stand': PedipulationCommandCfg(resampling_time_range=(10., 10.), debug_vis=play)},
    events=event_terms,
    rewards=reward_terms,
    terminations={
      'base_contact': TerminationTermCfg(func=terminations.base_contact),
      'time_out': TerminationTermCfg(func=terminations.time_out, time_out=True),
    },
    curriculum={},
    sim=SimulationCfg(nconmax=96, njmax=512, contact_sensor_maxmatch=128,
      mujoco=MujocoCfg(timestep=.005, iterations=20, ls_iterations=10)),
    decimation=4,
    episode_length_s=20.,
    scale_rewards_by_dt=True,
    viewer=ViewerConfig(entity_name='robot', body_name='base_link',
      origin_type=ViewerConfig.OriginType.ASSET_BODY, distance=2., elevation=-10.),
  )
