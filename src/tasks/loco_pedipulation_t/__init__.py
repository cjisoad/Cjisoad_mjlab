"""Independent quadruped/tripod teacher with leg_manip's policy interface."""
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.teacher_common.env_cfg import loco_pedipulation_teacher_env_cfg
from src.tasks.teacher_common.rl import TeacherOnPolicyRunner, teacher_ppo_runner_cfg

register_mjlab_task(task_id='loco_pedipulation_t',
  env_cfg=loco_pedipulation_teacher_env_cfg(),
  play_env_cfg=loco_pedipulation_teacher_env_cfg(play=True),
  rl_cfg=teacher_ppo_runner_cfg('loco_pedipulation_t'), runner_cls=TeacherOnPolicyRunner)
