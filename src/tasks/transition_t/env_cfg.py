"""Go2 one-shot rise tracking with the source locomotion action convention."""
from copy import deepcopy

from mjlab.envs import mdp
from mjlab.managers import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from src.assets.robots.unitree_go2.go2_constants import INIT_STATE
from src.tasks.pedipulation.mdp.actions import PedipulationPositionActionCfg
from src.tasks.pedipulation.robot import get_stand_robot_cfg
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from .commands import TRACKED_BODY_NAMES, TransitionCommandCfg
from . import terminations


# Support-leg command dynamics stay unchanged for the frozen endpoint teachers.
TRANSITION_TARGET_VELOCITY_LIMITS = (None,)*3+(2.,)*3+(None,)*6
TRANSITION_TARGET_ACCELERATION_LIMITS = (None,)*3+(30.,)*3+(None,)*6


def _ground_sensor(name, bodies):
  return ContactSensorCfg(name=name,
    primary=ContactMatch(mode='body', pattern=bodies, entity='robot'),
    secondary=ContactMatch(mode='geom', pattern='terrain'),
    fields=('found', 'force'), reduce='netforce', num_slots=1, global_frame=True)


def transition_env_cfg(play=False):
  cfg = make_tracking_env_cfg()
  robot = get_stand_robot_cfg()
  robot.init_state.joint_pos = deepcopy(INIT_STATE.joint_pos)
  robot.articulation.soft_joint_pos_limit_factor = .98
  for actuator in robot.articulation.actuators:
    actuator.armature = .02 if 'calf' in actuator.target_names_expr[0] else .01
  cfg.scene.entities = {'robot': robot}
  cfg.scene.num_envs = 1 if play else 4096
  cfg.scene.env_spacing = 3.
  cfg.scene.sensors = (
    ContactSensorCfg(name='self_collision',
      primary=ContactMatch(mode='subtree', pattern='base_link', entity='robot'),
      secondary=ContactMatch(mode='subtree', pattern='base_link', entity='robot'),
      fields=('found', 'force'), reduce='none', num_slots=1, history_length=4),
    _ground_sensor('feet_ground_contact', tuple(f'{leg}_foot' for leg in ('FL','FR','RL','RR'))),
    _ground_sensor('base_ground_contact', ('base_link','Head_upper','Head_lower')),
  )
  cfg.actions = {'joint_pos': PedipulationPositionActionCfg(
    entity_name='robot', scale=.25, delay=False,
    target_velocity_limits=TRANSITION_TARGET_VELOCITY_LIMITS,
    target_acceleration_limits=TRANSITION_TARGET_ACCELERATION_LIMITS)}
  cfg.commands = {'motion': TransitionCommandCfg(
    pose_range=dict(x=(-.02,.02), y=(-.02,.02), z=(-.01,.01),
                    roll=(-.08,.08), pitch=(-.08,.08), yaw=(-.15,.15)),
    velocity_range=dict(x=(-.2,.2), y=(-.2,.2), z=(-.1,.1),
                        roll=(-.3,.3), pitch=(-.3,.3), yaw=(-.3,.3)),
    debug_vis=play)}
  for group in ('actor', 'critic'):
    terms = cfg.observations[group].terms
    terms['base_lin_vel'].func = mdp.base_lin_vel
    terms['base_lin_vel'].params = {}
    terms['base_ang_vel'].func = mdp.base_ang_vel
    terms['base_ang_vel'].params = {}
  cfg.events.pop('push_robot')
  cfg.events['base_com'].params['asset_cfg'].body_names = ('base_link',)
  cfg.events['base_com'].params['ranges'] = {axis: (-.01,.01) for axis in range(3)}
  # Encoder bias is an observation perturbation; the source custom PD action
  # does not consume the generic actuator encoder correction.
  cfg.events['foot_friction'].params['asset_cfg'].geom_names = '.*_foot_collision'
  cfg.events['foot_friction'].params['ranges'] = (.6, 1.2)
  cfg.rewards['motion_global_root_pos'].params['std'] = .15
  cfg.rewards['motion_body_pos'].params['std'] = .15
  cfg.terminations = {
    'time_out': TerminationTermCfg(func=mdp.time_out, time_out=True),
    'anchor_pos': TerminationTermCfg(func=terminations.tracking_height_failure),
    'anchor_ori': TerminationTermCfg(func=terminations.tracking_orientation_failure),
    'ee_body_pos': TerminationTermCfg(func=terminations.tracking_foot_height_failure),
    'physical_failure': TerminationTermCfg(func=terminations.physical_failure),
  }
  cfg.viewer.body_name = 'base_link'
  cfg.viewer.distance = 2.5
  cfg.viewer.elevation = -10.
  cfg.episode_length_s = 8.
  cfg.sim.nconmax = 96
  cfg.sim.njmax = 512
  cfg.sim.contact_sensor_maxmatch = 1024
  cfg.sim.mujoco.iterations = 30
  cfg.sim.mujoco.ls_iterations = 10
  if play:
    cfg.observations['actor'].enable_corruption = False
    cfg.events = {}
    command = cfg.commands['motion']
    command.sampling_mode = 'start'
    command.pose_range = {}
    command.velocity_range = {}
    command.joint_position_range = (0., 0.)
  return cfg
