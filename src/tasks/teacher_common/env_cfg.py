"""Independent source rewards with task-specific teacher physics settings."""
from copy import deepcopy
from dataclasses import asdict
import re

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from src.tasks.leg_manip.env_cfg import leg_manip_env_cfg
from src.tasks.leg_manip.mdp import observations
from src.tasks.pedipulation.env_cfg import pedipulation_env_cfg
from src.tasks.loco_pedipulation.env_cfg import loco_pedipulation_env_cfg
from .commands import PedipulationTeacherCommandCfg, LocoPedipulationTeacherCommandCfg
from .rewards import pedipulation_reward
from .anchor_frame import anchor_degenerate_fraction


def pedipulation_teacher_env_cfg(play=False):
  cfg = pedipulation_env_cfg(play=play)
  source_command = cfg.commands['stand']
  cfg.commands = {'twist': PedipulationTeacherCommandCfg(**asdict(source_command))}
  cfg.metrics['anchor_degenerate_fraction'] = MetricsTermCfg(func=anchor_degenerate_fraction)
  cfg.observations['actor'].terms['policy'] = ObservationTermCfg(
    func=observations.policy_state, params={'add_noise': not play})
  cfg.observations['critic'].terms['policy'].func = observations.shared_policy_state
  cfg.observations['critic'].terms['linear_velocity'].func = observations.base_velocity
  for name, term in cfg.rewards.items():
    term.func = pedipulation_reward
    term.params = {**term.params, 'reward_name': name}
  return cfg


def loco_pedipulation_teacher_env_cfg(play=False):
  cfg = loco_pedipulation_env_cfg(play=play)
  common = leg_manip_env_cfg(play=play)
  source_robot = cfg.scene.entities['robot']
  source_com_ranges = cfg.events['base_com'].params['ranges']
  source_command, source_push = cfg.commands['twist'], cfg.events.get('push_robot')
  cfg.commands = {'twist': LocoPedipulationTeacherCommandCfg(**asdict(source_command))}
  cfg.metrics['anchor_degenerate_fraction'] = MetricsTermCfg(func=anchor_degenerate_fraction)
  cfg.scene.entities['robot'] = deepcopy(common.scene.entities['robot'])
  # Restore the original loco action/observation joint zero and its fixed
  # rotor inertias, while retaining the standing spec and PD actuator type.
  robot = cfg.scene.entities['robot']
  robot.init_state.joint_pos = deepcopy(source_robot.init_state.joint_pos)
  for actuator in robot.articulation.actuators:
    name, = actuator.target_names_expr
    source_actuator = next(reference for reference in source_robot.articulation.actuators
      if any(re.fullmatch(pattern, name) for pattern in reference.target_names_expr))
    actuator.armature = source_actuator.armature
  cfg.scene.sensors = deepcopy(common.scene.sensors)
  cfg.actions = deepcopy(common.actions)
  # Keep the custom-action DR event. Generic Go2 actuator/encoder events
  # cannot be applied again to this standing action.
  cfg.events = {name: deepcopy(term) for name, term in common.events.items() if name != 'push_robot'}
  cfg.events['physics'].params.update(
    com_range=source_com_ranges[0], armature_range=None)
  if source_push is not None:
    cfg.events['push_robot'] = source_push
  cfg.observations['actor'].terms['policy'] = ObservationTermCfg(
    func=observations.policy_state, params={'add_noise': not play})
  cfg.observations['critic'].terms['policy'].func = observations.shared_policy_state
  # Source loco critic uses unscaled exact velocity; keep its original layout.
  return cfg
