"""Independent reviewed V2 landing task; V1 checkpoints remain reproducible."""
from mjlab.tasks.registry import register_mjlab_task
from .env_cfg import landing_env_cfg
from .rl import LandingRunner,landing_ppo_cfg
register_mjlab_task(task_id='Transition-Down-Go2-v2',env_cfg=landing_env_cfg(),
    play_env_cfg=landing_env_cfg(play=True),rl_cfg=landing_ppo_cfg(),runner_cls=LandingRunner)
