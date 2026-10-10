"""Reuse weights-only initialization and strict resumable entry training."""
import torch
from src.tasks.transition_t.entry_rl import EntryOnPolicyRunner, _recipe_value
from src.tasks.transition_t.entry_bank import LEVELS, SPEED_EDGES, ERROR_EDGES
from .standing import standing_hold_policy
from .data import ALIGNMENT, TEACHER_SHA256
from .rl import DownOnPolicyRunner


class DownEntryOnPolicyRunner(EntryOnPolicyRunner,DownOnPolicyRunner):
  def resume(self, path):
    # Contact sensor caches are allocated during inference-mode PPO rollouts.
    # Their reset must run in that mode when resuming an already used world.
    with torch.inference_mode():
      return super().resume(path)

  def _reward_recipe(self):
    cfg=self.env.unwrapped.cfg
    return dict(episode_length_s=cfg.episode_length_s,scale_rewards_by_dt=cfg.scale_rewards_by_dt,
      rewards=_recipe_value(cfg.rewards),standing_hold=standing_hold_policy(),
      hold_shaping=dict(version=1,initial_credit=.2,duration_gain=.8,ramp_s=1.,
        maximum_raw_reward=1.,bad_sample_reward=0.))

  def _entry_recipe(self):
    return dict(version=1,task='transition_down_t_entry',direction='biped_to_tripod',
      bank_sha256=self.term.bank.sha256,teacher_sha256=self.term.bank.metadata['teacher_sha256'],
      endpoint_teachers=dict(TEACHER_SHA256),endpoint_support=['FL','RL','RR'],
      levels=list(LEVELS),speed_edges=list(SPEED_EDGES),error_edges=list(ERROR_EDGES),
      split='trajectory_id_modulo_5',sampling='uniform_occupied_bins_then_states',
      evaluation_sampling='uniform_occupied_bins_without_repeated_trajectories_per_pass',
      promotion=dict(min_attempts=64,min_distinct_trajectories=64,standing_rate=.8,
        consecutive_windows=2,standing_hold=standing_hold_policy()),
      alignment=ALIGNMENT,tracking_grace_for_entries=0.,collection_physics='nominal_no_domain_randomization',
      target_limiter=self._contract().get('target_limiter'),initialization=self._initialization_recipe(),
      training_augmentation=self._augmentation_recipe(),training_objective=self._reward_recipe())
