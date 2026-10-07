"""Apply exactly one selected controller with the original PD/delay history."""
from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse
from src.tasks.teacher_common.anchor_frame import to_anchor

from src.tasks.pedipulation.mdp.actions import PedipulationPositionAction, PedipulationPositionActionCfg
from .commands import WARM, PREFIX, BRIDGE, VERIFY, DONE
from .experts import FrozenTeacherPair, DEFAULT_QUADRUPED_CHECKPOINT, DEFAULT_BIPED_CHECKPOINT


@dataclass(kw_only=True)
class BridgePositionActionCfg(PedipulationPositionActionCfg):
  quadruped_checkpoint: str = str(DEFAULT_QUADRUPED_CHECKPOINT)
  biped_checkpoint: str = str(DEFAULT_BIPED_CHECKPOINT)

  def build(self,env):
    return BridgePositionAction(self,env)


class BridgePositionAction(PedipulationPositionAction):
  def __init__(self,cfg,env):
    super().__init__(cfg,env)
    self.experts=None

  def _ensure_experts(self):
    if self.experts is None:
      self.experts=FrozenTeacherPair(self._env,
        quadruped_checkpoint=self.cfg.quadruped_checkpoint,
        biped_checkpoint=self.cfg.biped_checkpoint)
    return self.experts

  @torch.inference_mode()
  def process_actions(self,actions):
    term=self._env.command_manager.get_term('twist')
    term.begin_step()
    ids=(term.pending_initial_impulse & (term.action_owner==BRIDGE)).nonzero().flatten()
    if len(ids):
      # A bounded training impulse changes momentum only, never pose/history.
      data=self._entity.data
      velocity=torch.cat((data.root_link_lin_vel_w[ids],data.root_link_ang_vel_w[ids]),-1)
      velocity+=term.initial_impulse[ids]
      self._entity.write_root_link_velocity_to_sim(velocity,env_ids=ids)
      term.start_linear_velocity[ids]=to_anchor(term.basis_w[ids],velocity[:,:3])
      term.start_angular_velocity[ids]=quat_apply_inverse(data.root_link_quat_w[ids],velocity[:,3:])
      term.pending_initial_impulse[ids]=False
    selected=actions.clamp(-10.,10.).clone()
    for name,mask in (('quadruped',(term.action_owner==WARM)|(term.action_owner==PREFIX)),
                      ('biped',(term.action_owner==VERIFY)|(term.action_owner==DONE))):
      ids=mask.nonzero().flatten()
      if len(ids):
        selected[ids]=self._ensure_experts().actions(name,term.command,env_ids=ids)
    # Source action applies only its original ±100 safety limit. Translated
    # expert actions may exceed10; clipping them here changes q_des.
    super().process_actions(selected)
    self.delay_steps[~term.disturbed]=0
    if term.course_stage>=2:
      maximum=round((self._env.cfg.decimation-1)*term.course.randomization_fraction)
      self.delay_steps.clamp_max_(maximum)
