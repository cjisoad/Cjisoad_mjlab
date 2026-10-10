"""Direction overrides of the approved transition_t environment."""
from dataclasses import fields
from mjlab.managers import RewardTermCfg
from src.tasks.transition_t.env_cfg import transition_env_cfg
from .commands import DownMotionCfg
from .rewards import SustainedTripodReward


def down_env_cfg(play=False):
  cfg = transition_env_cfg(play=play)
  old = cfg.commands['motion']
  values = {field.name: getattr(old, field.name) for field in fields(old) if field.name != 'motion_file'}
  cfg.commands['motion'] = DownMotionCfg(**values)
  cfg.rewards['standing_hold'] = RewardTermCfg(func=SustainedTripodReward, weight=.1)
  return cfg
