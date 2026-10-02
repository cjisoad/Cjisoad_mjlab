"""Fused-task runner: curriculum persistence, manipulation logging, fused PPO."""

from dataclasses import replace

import torch

from mjlab.rl.runner import MjlabOnPolicyRunner

from src.tasks.loco_pedipulation.rl import LocoPedipulationOnPolicyRunner
from src.tasks.pedipulation.rl.config import pedipulation_ppo_runner_cfg
from .checkpoint import observation_layout, course_state, restore_course_state, validate_course_state
from .metrics import CourseLogger


def leg_manip_ppo_runner_cfg():
  cfg = pedipulation_ppo_runner_cfg()
  cfg.seed = 42
  cfg.experiment_name = 'leg_manip'
  cfg.clip_actions = 10.
  # Upload every periodic checkpoint to the wandb run.
  cfg.upload_model = True
  cfg.algorithm = replace(cfg.algorithm,
                          class_name='src.tasks.leg_manip.rl.ppo:LegManipPPO')
  return cfg


class LegManipOnPolicyRunner(LocoPedipulationOnPolicyRunner):
  """Use a versioned course and six-group logger without changing source tasks."""

  def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
    # Bypass the shared logger constructor: it assumes a two-row tracking ABI.
    MjlabOnPolicyRunner.__init__(self, env, train_cfg, log_dir, device)
    term = self.env.unwrapped.command_manager.get_term('twist')
    self.logger = CourseLogger(self.logger, term.training_metrics, self.is_distributed)

  def _curriculum_state(self):
    term = self.env.unwrapped.command_manager.get_term('twist')
    return course_state(term, self.logger.landing_totals, self.logger.switch_totals)

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    # Reject old stages before model, optimizer, iteration, or env state mutates.
    checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
    state = validate_course_state((checkpoint.get('infos') or {}).get('leg_manip_course'))
    infos = MjlabOnPolicyRunner.load(self, path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    term = self.env.unwrapped.command_manager.get_term('twist')
    if term.cfg.curriculum_enabled:
      restore_course_state(term, state)
    self.logger.restore_counts(state['landing_counts'], state.get('switch_counts'))
    return infos

  def save(self, path, infos=None):
    # VelocityOnPolicyRunner.save also exports ONNX deploy metadata, which
    # asserts mjlab's JointPositionAction; this task uses
    # PedipulationPositionAction, so persist through the plain mjlab save
    # (which still honors upload_model) and skip the ONNX metadata.
    MjlabOnPolicyRunner.save(self, path,
      {**(infos or {}), 'leg_manip_course': self._curriculum_state(),
       'leg_manip_observations': observation_layout()})
