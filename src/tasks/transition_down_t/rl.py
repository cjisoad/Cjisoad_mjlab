"""Original PPO with a strict, portable reverse-transition contract."""
from src.tasks.transition_t.rl import TransitionOnPolicyRunner, transition_ppo_runner_cfg
from src.tasks.transition_t.entry_rl import _recipe_value
from .data import ALIGNMENT, REVIEW_SHA256, TEACHER_SHA256
from .env_cfg import down_env_cfg
from .standing import standing_hold_policy


def down_ppo_runner_cfg():
  cfg=transition_ppo_runner_cfg()
  cfg.class_name='DownOnPolicyRunner'
  cfg.experiment_name='transition_down_t'
  cfg.save_interval=100
  return cfg


def augmentation_recipe(cfg):
  group=cfg.observations['actor']
  return dict(events=_recipe_value(cfg.events),actor_corruption=group.enable_corruption,
    actor_noise={name:_recipe_value(term.noise) for name,term in group.terms.items()},
    pose_range=cfg.commands['motion'].pose_range,velocity_range=cfg.commands['motion'].velocity_range,
    joint_position_range=cfg.commands['motion'].joint_position_range)


class DownOnPolicyRunner(TransitionOnPolicyRunner):
  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    if augmentation_recipe(self.env.unwrapped.cfg) != augmentation_recipe(down_env_cfg()):
      raise ValueError('Training randomization differs from the approved recipe')
    return super().learn(num_learning_iterations, init_at_random_ep_len=False)

  def _contract(self):
    result=super()._contract()
    cfg=self.env.unwrapped.cfg
    result.update(task='transition_down_t',direction='biped_to_tripod',
      alignment=ALIGNMENT,support_feet=['FL','RL','RR'],airborne_feet=['FR'],
      teachers=dict(TEACHER_SHA256),approved_review_sha256=dict(REVIEW_SHA256),
      endpoint_limits=dict(orientation_rad=.25,height_m=.05,linear_speed_m_s=.3,
        angular_speed_rad_s=.8,joint_speed_norm=20.,support_force_n=5.,fr_center_height_m=.032,
        strict_world_feet_rms_m=.06),
      training_augmentation=augmentation_recipe(down_env_cfg()),
      training_objective=dict(episode_length_s=cfg.episode_length_s,
        scale_rewards_by_dt=cfg.scale_rewards_by_dt,rewards=_recipe_value(cfg.rewards),
        hold=standing_hold_policy(),initial_credit=.2,duration_gain=.8))
    return result

  def load(self,*args,allow_legacy_target_limiter=False,**kwargs):
    # No approved reverse policy predates this target-limiter contract.
    return super().load(*args,allow_legacy_target_limiter=False,**kwargs)
