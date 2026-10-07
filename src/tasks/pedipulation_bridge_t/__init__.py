"""Frozen quadruped → learned bridge → frozen biped, with real handoff credit."""
from mjlab.tasks.registry import register_mjlab_task
from .env_cfg import bridge_env_cfg
from .rl import BridgeOnPolicyRunner, bridge_ppo_runner_cfg

register_mjlab_task(task_id='pedipulation_bridge_t',
  env_cfg=bridge_env_cfg(), play_env_cfg=bridge_env_cfg(play=True),
  rl_cfg=bridge_ppo_runner_cfg(), runner_cls=BridgeOnPolicyRunner)
