"""Independent Go2 BeyondMimic rise tracking task."""
from mjlab.tasks.registry import register_mjlab_task
from .env_cfg import transition_env_cfg
from .rl import TransitionOnPolicyRunner, transition_ppo_runner_cfg

register_mjlab_task(
  task_id='Transition-Go2-v0',
  env_cfg=transition_env_cfg(),
  play_env_cfg=transition_env_cfg(play=True),
  rl_cfg=transition_ppo_runner_cfg(),
  runner_cls=TransitionOnPolicyRunner,
)
