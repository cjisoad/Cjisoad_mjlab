"""Independent source rewards, shared leg_manip actor/action/physical model."""
from copy import deepcopy
from dataclasses import asdict

from mjlab.managers.observation_manager import ObservationTermCfg
from src.tasks.leg_manip.env_cfg import leg_manip_env_cfg
from src.tasks.leg_manip.mdp import observations
from src.tasks.pedipulation.env_cfg import pedipulation_env_cfg
from src.tasks.loco_pedipulation.env_cfg import loco_pedipulation_env_cfg
from .commands import PedipulationTeacherCommandCfg, LocoPedipulationTeacherCommandCfg
from .rewards import pedipulation_reward


def pedipulation_teacher_env_cfg(play=False):
  cfg = pedipulation_env_cfg(play=play)
  source_command = cfg.commands['stand']
  cfg.commands = {'twist': PedipulationTeacherCommandCfg(**asdict(source_command))}
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
  source_command, source_push = cfg.commands['twist'], cfg.events.get('push_robot')
  cfg.commands = {'twist': LocoPedipulationTeacherCommandCfg(**asdict(source_command))}
  cfg.scene.entities['robot'] = deepcopy(common.scene.entities['robot'])
  cfg.scene.sensors = deepcopy(common.scene.sensors)
  cfg.actions = deepcopy(common.actions)
  # Shared physical randomizations must match the student model. Do not run
  # generic Go2 actuator/encoder DR again on the custom standing action.
  cfg.events = {name: deepcopy(term) for name, term in common.events.items() if name != 'push_robot'}
  if source_push is not None:
    cfg.events['push_robot'] = source_push
  cfg.observations['actor'].terms['policy'] = ObservationTermCfg(
    func=observations.policy_state, params={'add_noise': not play})
  cfg.observations['critic'].terms['policy'].func = observations.shared_policy_state
  # Source loco critic uses unscaled exact velocity; keep its original layout.
  return cfg
