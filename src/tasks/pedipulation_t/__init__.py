"""Independent rear-standing teacher with leg_manip's policy interface."""
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.teacher_common.env_cfg import pedipulation_teacher_env_cfg
from src.tasks.teacher_common.rl import TeacherOnPolicyRunner, teacher_ppo_runner_cfg

register_mjlab_task(task_id='pedipulation_t',
  env_cfg=pedipulation_teacher_env_cfg(),
  play_env_cfg=pedipulation_teacher_env_cfg(play=True),
  rl_cfg=teacher_ppo_runner_cfg('pedipulation_t'), runner_cls=TeacherOnPolicyRunner)
