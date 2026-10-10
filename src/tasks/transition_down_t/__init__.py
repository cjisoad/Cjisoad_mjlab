"""Independent Go2 biped-to-tripod BeyondMimic transition."""
from mjlab.tasks.registry import register_mjlab_task
from .env_cfg import down_env_cfg
from .rl import DownOnPolicyRunner, down_ppo_runner_cfg

register_mjlab_task(task_id='Transition-Down-Go2-v0',env_cfg=down_env_cfg(),
  play_env_cfg=down_env_cfg(play=True),rl_cfg=down_ppo_runner_cfg(),runner_cls=DownOnPolicyRunner)
