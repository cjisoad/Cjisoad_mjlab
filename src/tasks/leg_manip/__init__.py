"""Leg Manip: fused quadruped/tripod/biped locomotion with FR pedipulation."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import leg_manip_env_cfg
from .rl import LegManipOnPolicyRunner, leg_manip_ppo_runner_cfg

register_mjlab_task(
  task_id='leg_manip',
  env_cfg=leg_manip_env_cfg(),
  play_env_cfg=leg_manip_env_cfg(play=True),
  rl_cfg=leg_manip_ppo_runner_cfg(),
  runner_cls=LegManipOnPolicyRunner,
)
