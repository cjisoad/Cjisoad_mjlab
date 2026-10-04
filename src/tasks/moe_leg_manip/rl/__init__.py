"""Fused-task runner: curriculum persistence, manipulation logging, fused PPO."""

from dataclasses import asdict, dataclass

import torch

from mjlab.rl.runner import MjlabOnPolicyRunner

from src.tasks.loco_pedipulation.rl import LocoPedipulationOnPolicyRunner
from src.tasks.pedipulation.rl.config import PedipulationPpoAlgorithmCfg, pedipulation_ppo_runner_cfg
from .checkpoint import observation_layout, course_state, restore_course_state, validate_course_state
from .metrics import CourseLogger
from .stance_evaluation import evaluate_stances, apply_stance_evaluation


@dataclass
class LegManipPpoAlgorithmCfg(PedipulationPpoAlgorithmCfg):
  gradient_diagnostics_interval: int = 20
  gradient_diagnostics_max_samples: int = 512


def leg_manip_ppo_runner_cfg():
  cfg = pedipulation_ppo_runner_cfg()
  cfg.seed = 42
  cfg.experiment_name = 'leg_manip'
  cfg.clip_actions = 10.
  # Upload every periodic checkpoint to the wandb run.
  cfg.upload_model = True
  cfg.actor.distribution_cfg = {**cfg.actor.distribution_cfg,
    'class_name': 'src.tasks.moe_leg_manip.rl.distribution:StanceGaussianDistribution', 'init_std': .2}
  cfg.algorithm = LegManipPpoAlgorithmCfg(**{**asdict(cfg.algorithm),
    'class_name': 'src.tasks.moe_leg_manip.rl.ppo:LegManipPPO', 'entropy_coef': .001})
  return cfg


class LegManipOnPolicyRunner(LocoPedipulationOnPolicyRunner):
  """Use a versioned course and six-group logger without changing source tasks."""

  def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
    # Bypass the shared logger constructor: it assumes a two-row tracking ABI.
    MjlabOnPolicyRunner.__init__(self, env, train_cfg, log_dir, device)
    term = self.env.unwrapped.command_manager.get_term('twist')
    self.alg.gradient_command_term = term
    self.logger = CourseLogger(self.logger, term.training_metrics, self.is_distributed, self._iteration_end)

  def _iteration_end(self, it):
    term = self.env.unwrapped.command_manager.get_term('twist')
    iteration = it + 1
    diagnostics = getattr(self.alg, 'last_gradient_diagnostics', None)
    if diagnostics:
      if getattr(self.logger, 'writer', None) is not None:
        for key, value in diagnostics.items():
          self.logger.writer.add_scalar('GradConflict/' + key, value, it)
      cosine = diagnostics.get('quad_biped/policy/cosine')
      print(f'[grad_diag] iteration={iteration}, quad_biped_policy_cosine={cosine}, '
        f'quad_samples={diagnostics.get("samples/quad", 0)}, '
        f'biped_samples={diagnostics.get("samples/biped", 0)}', flush=True)
    if (term.cfg.curriculum_enabled and term.stage == 0
        and iteration > term.last_stance_eval_iteration and iteration % term.cfg.stance_eval_interval == 0):
      # Only rank0 simulates evaluation; all ranks apply the same course decision.
      results = None
      if not self.is_distributed or torch.distributed.get_rank() == 0:
        print(f'[stance_eval] iteration={iteration}, 32 trials per stance', flush=True)
        results = evaluate_stances(self.alg.actor, self.env.unwrapped.cfg, self.device,
          int(self.cfg.get('seed', 42)) + 100000 + iteration)
      if self.is_distributed:
        payload = [results]
        torch.distributed.broadcast_object_list(payload, src=0)
        results = payload[0]
      apply_stance_evaluation(term, results, iteration)
      print(f'[stance_eval] {results}; stage={term.stage}, streak={term.pass_streak}', flush=True)
      if getattr(self.logger, 'writer', None) is not None:
        for key, value in term.last_decision.items():
          self.logger.writer.add_scalar('StanceEval/' + key, value, it)
    self._set_exploration(iteration)

  def _set_exploration(self, iteration):
    term = self.env.unwrapped.command_manager.get_term('twist')
    if term.stage == 0:
      progress = 0.
    else:
      if term.walk_started_iteration < 0:
        term.walk_started_iteration = iteration
      progress = min(max((iteration - term.walk_started_iteration) / 1500., 0.), 1.)
    self.alg.actor.distribution.max_std = .3 + .7 * progress
    self.alg.entropy_coef = .001 + .009 * progress
    if getattr(self.logger, 'writer', None) is not None:
      self.logger.writer.add_scalar('StanceEval/std_cap', self.alg.actor.distribution.max_std, iteration - 1)
      self.logger.writer.add_scalar('StanceEval/entropy_coef', self.alg.entropy_coef, iteration - 1)
      std = self.alg.actor.distribution.std_param.detach().clamp(.02, self.alg.actor.distribution.max_std)
      for joint, value in enumerate(std.tolist()):
        self.logger.writer.add_scalar(f'StanceEval/action_std/joint_{joint}', value, iteration - 1)

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
    self._set_exploration(self.current_learning_iteration + 1)
    return infos

  def save(self, path, infos=None):
    # VelocityOnPolicyRunner.save also exports ONNX deploy metadata, which
    # asserts mjlab's JointPositionAction; this task uses
    # PedipulationPositionAction, so persist through the plain mjlab save
    # (which still honors upload_model) and skip the ONNX metadata.
    MjlabOnPolicyRunner.save(self, path,
      {**(infos or {}), 'leg_manip_course': self._curriculum_state(),
       'leg_manip_observations': observation_layout()})
