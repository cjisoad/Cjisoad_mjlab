"""Pedipulation: rear-leg standing with a gravity-aligned velocity frame."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import pedipulation_env_cfg
from .rl import PedipulationOnPolicyRunner, pedipulation_ppo_runner_cfg

register_mjlab_task(
  task_id='Pedipulation',
  env_cfg=pedipulation_env_cfg(),
  play_env_cfg=pedipulation_env_cfg(play=True),
  rl_cfg=pedipulation_ppo_runner_cfg(),
  runner_cls=PedipulationOnPolicyRunner,
)
