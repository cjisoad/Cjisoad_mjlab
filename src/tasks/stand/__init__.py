"""Rear-leg support task (named go2_handstand in the source repository)."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import stand_env_cfg
from .rl import StandOnPolicyRunner, stand_ppo_runner_cfg

register_mjlab_task(
  task_id='Unitree-Go2-RearStand',
  env_cfg=stand_env_cfg(),
  play_env_cfg=stand_env_cfg(play=True),
  rl_cfg=stand_ppo_runner_cfg(),
  runner_cls=StandOnPolicyRunner,
)
