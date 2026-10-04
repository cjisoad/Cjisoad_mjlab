"""Read-only command alias for the unchanged pedipulation reward functions."""
from src.tasks.pedipulation.mdp import rewards


class _CommandView:
  def __init__(self, manager):
    self.manager = manager

  def get_term(self, name):
    return self.manager.get_term('twist' if name == 'stand' else name)

  def get_command(self, name):
    return self.manager.get_command('twist' if name == 'stand' else name)

  def __getattr__(self, name):
    return getattr(self.manager, name)


class PedipulationRewardView:
  def __init__(self, env):
    self.env = env
    self.command_manager = _CommandView(env.command_manager)

  def __getattr__(self, name):
    return getattr(self.env, name)


def source_reward_function(name):
  return getattr(rewards, name)


def pedipulation_reward(env, reward_name, **params):
  return source_reward_function(reward_name)(PedipulationRewardView(env), **params)
