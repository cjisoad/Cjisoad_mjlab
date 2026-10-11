"""Checked scratch/resume recipe and deliberate stage-boundary reconstruction."""
import math

import torch

from .entry_rl import EntryOnPolicyRunner
from .rl import TransitionOnPolicyRunner
from .three_stage import STAGES


class ThreeStageOnPolicyRunner(EntryOnPolicyRunner):
  def _runner_recipe(self):
    cfg = self.cfg
    algorithm = cfg.get('algorithm', {})
    return dict(num_steps_per_env=cfg.get('num_steps_per_env'),
      clip_actions=cfg.get('clip_actions'), algorithm={key: algorithm.get(key) for key in (
        'num_learning_epochs','num_mini_batches','learning_rate','schedule','desired_kl',
        'gamma','lam','clip_param','entropy_coef','value_loss_coef','max_grad_norm','use_clipped_value_loss')})

  def _entry_recipe(self):
    recipe = super()._entry_recipe()
    recipe.update(version=8, curriculum='three-stage-v3-from-scratch', levels=list(STAGES),
      initial_weights='random', stage=self.term.cfg.stage,
      runner=self._runner_recipe(),
      promotion=dict(nominal=dict(attempts=64, successes=52),
        stage1=dict(attempts=64, successes=48, distinct_trajectories=64),
        final=dict(attempts=128, successes=103, distinct_trajectories=128),
        consecutive_windows=2, nominal_retention=True))
    return recipe

  def start_from_scratch(self):
    self._verify_bank_physics()
    self.seed_checkpoint_sha256 = ''
    self.current_learning_iteration = 0
    self.env.reset()

  def resume(self, path):
    return self._restore(path, stage_transition=False)

  def restore_stage_transition(self, path):
    return self._restore(path, stage_transition=True)

  def _restore(self, path, *, stage_transition):
    self._verify_bank_physics()
    saved = torch.load(path, map_location='cpu', weights_only=False)
    state = (saved.get('infos') or {}).get('transition_entry_training')
    if not isinstance(state, dict) or not isinstance(state.get('recipe'), dict):
      raise ValueError('Three-stage checkpoint metadata is missing')
    recipe, expected = dict(state['recipe']), self._entry_recipe()
    if stage_transition:
      old_stage = recipe.get('stage')
      if (type(old_stage) is not int or old_stage+1 != self.term.cfg.stage
          or state['course']['level'] != self.term.cfg.stage):
        raise ValueError('Only the immediately promoted stage can rebuild the environment')
      # All shared reference/bank/physics/objective contracts must still match.
      for key in ('stage', 'training_augmentation'):
        recipe[key] = expected[key]
    if recipe != expected:
      raise ValueError('Three-stage checkpoint recipe / bank differs from current training')
    updates = state.get('updates_completed')
    if type(updates) is not int or updates <= 0 or state.get('seed_checkpoint_sha256') != '':
      raise ValueError('Require a valid from-scratch update clock')
    self.term.course.load_state_dict(state['course'])
    if self.term.course.level != self.term.cfg.stage:
      raise ValueError('Checkpoint course and instantiated physics stage differ')
    info = TransitionOnPolicyRunner.load(self, str(path), map_location=self.device)
    rates = [group['lr'] for group in self.alg.optimizer.param_groups]
    if not rates or any(not math.isfinite(r) or r <= 0 or r != rates[0] for r in rates):
      raise ValueError('Invalid restored optimizer learning rates')
    self.alg.learning_rate = rates[0]
    self.current_learning_iteration = updates
    self.seed_checkpoint_sha256 = ''
    self.env.unwrapped._sim_step_counter = self.env.unwrapped.common_step_counter*self.env.unwrapped.cfg.decimation
    self.env.reset()
    return info
