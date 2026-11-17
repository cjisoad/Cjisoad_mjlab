"""Registry entry for fresh Go2 world-model guided student distillation."""
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.legmanip_distil.supermimic.config import SuperMimicRunnerCfg
from src.tasks.legmanip_distil.supermimic.env import supermimic_env_cfg
from src.tasks.legmanip_distil.supermimic.training import SuperMimicRunner

register_mjlab_task(task_id='legmanip_distil_supermimic',
    env_cfg=supermimic_env_cfg(), play_env_cfg=supermimic_env_cfg(play=True),
    rl_cfg=SuperMimicRunnerCfg(), runner_cls=SuperMimicRunner)
