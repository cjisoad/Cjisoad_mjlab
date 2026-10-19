"""Online FSM-to-GRU distillation with a teacher-independent deployment world."""
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import legmanip_distil_env_cfg
from .rl import LegManipDistillationRunner, legmanip_distil_runner_cfg

register_mjlab_task(
    task_id='legmanip_distil',
    env_cfg=legmanip_distil_env_cfg(),
    play_env_cfg=legmanip_distil_env_cfg(play=True),
    rl_cfg=legmanip_distil_runner_cfg(),
    runner_cls=LegManipDistillationRunner,
)
