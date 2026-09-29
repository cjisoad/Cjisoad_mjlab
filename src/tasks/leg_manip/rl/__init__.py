"""Fused-task runner: curriculum persistence, manipulation logging, fused PPO."""

from dataclasses import replace

from mjlab.rl.runner import MjlabOnPolicyRunner

from src.tasks.loco_pedipulation.rl import LocoPedipulationOnPolicyRunner
from src.tasks.pedipulation.rl.config import pedipulation_ppo_runner_cfg


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
  """Inherit curriculum save/load and the manipulation metrics logger."""

  def save(self, path, infos=None):
    # VelocityOnPolicyRunner.save also exports ONNX deploy metadata, which
    # asserts mjlab's JointPositionAction; this task uses
    # PedipulationPositionAction, so persist through the plain mjlab save
    # (which still honors upload_model) and skip the ONNX metadata.
    MjlabOnPolicyRunner.save(self, path,
                             {**(infos or {}), 'loco_pedipulation': self._curriculum_state()})
