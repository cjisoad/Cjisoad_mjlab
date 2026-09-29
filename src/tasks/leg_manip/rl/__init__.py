"""Fused-task runner: curriculum persistence, manipulation logging, fused PPO."""

from dataclasses import replace

from src.tasks.loco_pedipulation.rl import LocoPedipulationOnPolicyRunner
from src.tasks.pedipulation.rl.config import pedipulation_ppo_runner_cfg


def leg_manip_ppo_runner_cfg():
  cfg = pedipulation_ppo_runner_cfg()
  cfg.seed = 42
  cfg.experiment_name = 'leg_manip'
  cfg.clip_actions = 10.
  cfg.algorithm = replace(cfg.algorithm,
                          class_name='src.tasks.leg_manip.rl.ppo:LegManipPPO')
  return cfg


class LegManipOnPolicyRunner(LocoPedipulationOnPolicyRunner):
  """Inherit curriculum save/load and the manipulation metrics logger."""
