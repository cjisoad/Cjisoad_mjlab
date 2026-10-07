"""Frozen-teacher transition on the unchanged quadruped teacher physical ABI."""
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg, ObservationGroupCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.tasks.teacher_common.env_cfg import loco_pedipulation_teacher_env_cfg
from .commands import BridgeCommandCfg
from .actions import BridgePositionActionCfg
from . import observations, rewards
from .physics import declare_tracking_physics


def bridge_env_cfg(play=False):
  cfg=loco_pedipulation_teacher_env_cfg(play=play)
  cfg.scene.sensors=(*cfg.scene.sensors,ContactSensorCfg(name='bridge_self_contact',
    primary=ContactMatch(mode='geom',pattern='.*',entity='robot'),
    secondary=ContactMatch(mode='subtree',pattern='base_link',entity='robot'),
    fields=('force','found'),reduce='netforce',num_slots=1))
  cfg.commands={'twist':BridgeCommandCfg(debug_vis=play,play=play,curriculum_enabled=not play)}
  cfg.actions={'joint_pos':BridgePositionActionCfg(entity_name='robot',delay=False)}
  cfg.observations['actor'].terms['policy']=ObservationTermCfg(
    func=observations.policy_state,params={'add_noise':not play})
  cfg.observations['critic']=ObservationGroupCfg(terms={
    'policy':ObservationTermCfg(func=observations.shared_policy_state),
    'privileged':ObservationTermCfg(func=observations.privileged_state)})
  cfg.events={'physics':EventTermCfg(func=declare_tracking_physics,mode='startup')}
  cfg.curriculum={}
  cfg.episode_length_s=24.
  # Command.advance classifies all failures before any reward or automatic
  # reset. Generic source fall/contact terms would lose tail attribution.
  cfg.terminations={'bridge_outcome':TerminationTermCfg(func=rewards.bridge_termination)}
  cfg.rewards={}
  for name,weight in (('height',3.),('orientation',3.),('support_pose',.5),
                      ('fl_height',.5),('fr_tracking',1.)):
    cfg.rewards[name]=RewardTermCfg(func=rewards.shaping,weight=weight,params={'name':name})
  for name,weight in (('action_change',-.03),('slip',-.15),('support_loss',-.7),
                      ('front_contact',-.5),('excessive_force',-.1),
                      ('joint_limits',-.5),('torque',-.02),('collision',-1.)):
    cfg.rewards[name]=RewardTermCfg(func=rewards.penalty,weight=weight,params={'name':name})
  # Terminal success/failure is intentionally supplied to the last bridge
  # transition by the collector after the real frozen-biped validation window.
  cfg.metrics={name:MetricsTermCfg(func=rewards.bridge_metric,params={'name':name})
    for name in ('bridge_owner','source_rejected','handoff_success','handoff_failed','handoff_count')}
  return cfg
