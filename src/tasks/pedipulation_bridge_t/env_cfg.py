"""Frozen-teacher transition on the unchanged quadruped teacher physical ABI."""
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.tasks.teacher_common.env_cfg import loco_pedipulation_teacher_env_cfg
from .commands import BridgeCommandCfg
from .actions import BridgePositionActionCfg
from . import observations, rewards


def bridge_env_cfg(play=False):
  cfg=loco_pedipulation_teacher_env_cfg(play=play)
  cfg.scene.sensors=(*cfg.scene.sensors,ContactSensorCfg(name='bridge_self_contact',
    primary=ContactMatch(mode='geom',pattern='.*',entity='robot'),
    secondary=ContactMatch(mode='subtree',pattern='base_link',entity='robot'),
    fields=('force','found'),reduce='netforce',num_slots=1))
  cfg.commands={'twist':BridgeCommandCfg(debug_vis=play)}
  cfg.actions={'joint_pos':BridgePositionActionCfg(entity_name='robot')}
  cfg.observations['actor'].terms['policy']=ObservationTermCfg(
    func=observations.policy_state,params={'add_noise':not play})
  cfg.observations['critic'].terms['policy'].func=observations.shared_policy_state
  cfg.curriculum={}
  cfg.episode_length_s=15.
  # Command.advance classifies all failures before any reward or automatic
  # reset. Generic source fall/contact terms would lose tail attribution.
  cfg.terminations={'bridge_outcome':TerminationTermCfg(func=rewards.bridge_termination)}
  cfg.rewards={}
  for name,weight in (('height',3.),('orientation',5.),('support_pose',.5),
                      ('fl_position',.5),('fr_tracking',.5),('support',.5),('rise_progress',2.)):
    cfg.rewards[name]=RewardTermCfg(func=rewards.shaping,weight=weight,params={'name':name})
  for name,weight in (('action_change',-.025),('slip',-.25),
                      ('front_contact',-.5),('joint_limits',-10.),('torque',-.02),('collision',-2.)):
    cfg.rewards[name]=RewardTermCfg(func=rewards.penalty,weight=weight,params={'name':name})
  # Terminal success/failure is intentionally supplied to the last bridge
  # transition by the collector after the real frozen-biped validation window.
  cfg.metrics={name:MetricsTermCfg(func=rewards.bridge_metric,params={'name':name})
    for name in ('bridge_owner','source_rejected','handoff_success','handoff_failed','handoff_count')}
  return cfg
