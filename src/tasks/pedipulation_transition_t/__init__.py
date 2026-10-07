"""Independent tripod-to-biped reference tracking teacher."""
from mjlab.tasks.registry import register_mjlab_task
from .env_cfg import transition_env_cfg
from .rl import TransitionOnPolicyRunner, transition_ppo_runner_cfg

register_mjlab_task(task_id='pedipulation_transition_t',
  env_cfg=transition_env_cfg(),play_env_cfg=transition_env_cfg(play=True),
  rl_cfg=transition_ppo_runner_cfg(),runner_cls=TransitionOnPolicyRunner)
