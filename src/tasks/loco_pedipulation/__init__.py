"""Quadruped locomotion with optional right-front pedipulation."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import loco_pedipulation_env_cfg
from .rl import LocoPedipulationOnPolicyRunner, loco_pedipulation_ppo_runner_cfg

register_mjlab_task(
  task_id='loco_pedipulation',
  env_cfg=loco_pedipulation_env_cfg(),
  play_env_cfg=loco_pedipulation_env_cfg(play=True),
  rl_cfg=loco_pedipulation_ppo_runner_cfg(),
  runner_cls=LocoPedipulationOnPolicyRunner,
)
