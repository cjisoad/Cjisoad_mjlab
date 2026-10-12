"""V3 from-scratch motion learning followed by genuine entry robustness."""
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from .entry_course import EntryCommand, EntryCommandCfg, entry_course_env_cfg
from .three_stage_rewards import knee_ground_cost, rise_height_reward, terminal_body_velocity_reward

# Default V3 timing uses the independent 1.25x central-rise revision.
# The original 3.44 s trace remains available beside it for reproducibility.
V3_MOTION = Path(__file__).parent/'data/tripod_to_biped_v3_speed125_train.npz'
STAGES = (
  dict(nominal_probability=.5, entry_probability=0., max_speed=.45, max_fr_error=.14, randomization=0.),
  dict(nominal_probability=.4, entry_probability=.4, max_speed=.45, max_fr_error=.14, randomization=.5),
  dict(nominal_probability=.2, entry_probability=.6, max_speed=1.2, max_fr_error=.45, randomization=1.),
)


@dataclass
class ThreeStageCurriculum:
  level: int = 0
  streak: int = 0
  evaluations: int = 0
  accepted: bool = False

  @property
  def settings(self):
    return STAGES[self.level]

  @staticmethod
  def _pass(report, source, count, required):
    if not isinstance(report, dict):
      return False
    attempts, successes = report.get('attempts'), report.get('successes')
    if (type(attempts) is not int or type(successes) is not int
        or attempts != count or not 0 <= successes <= attempts):
      return False
    return (report.get('initialization') == source and report.get('full_reference_from_start') is True
      and successes >= required and (source != 'entries'
        or report.get('distinct_trajectories', 0) >= count))

  def observe(self, report, *, nominal=None):
    self.evaluations += 1
    if self.level == 0:
      passed = self._pass(report, 'nominal', 64, 52)
    else:
      count, required = (64, 48) if self.level == 1 else (128, 103)
      passed = (self._pass(report, 'entries', count, required)
        and self._pass(nominal, 'nominal', 64, 52))
    self.streak = self.streak+1 if passed else 0
    if self.level == 2:
      self.accepted = passed
      return False
    if self.streak >= 2:
      self.level += 1
      self.streak = 0
      return True
    return False

  def state_dict(self):
    return dict(level=self.level, streak=self.streak, evaluations=self.evaluations, accepted=self.accepted)

  def load_state_dict(self, state):
    if (not isinstance(state, dict) or set(state) != {'level', 'streak', 'evaluations', 'accepted'}
        or any(type(state[k]) is not int or state[k] < 0 for k in ('level', 'streak', 'evaluations'))
        or state['level'] > 2 or type(state['accepted']) is not bool
        or (state['accepted'] and state['level'] != 2)):
      raise ValueError('Invalid three-stage curriculum state')
    self.level, self.streak, self.evaluations, self.accepted = (
      state[k] for k in ('level', 'streak', 'evaluations', 'accepted'))


@dataclass(kw_only=True)
class ThreeStageCommandCfg(EntryCommandCfg):
  stage: int = 0

  def build(self, env):
    return ThreeStageCommand(self, env)


class ThreeStageCommand(EntryCommand):
  levels = STAGES

  def make_course(self):
    return ThreeStageCurriculum(level=self.cfg.stage)

  def _update_metrics(self):
    super()._update_metrics()
    force = self._env.scene['knees_ground_contact'].data.force_history.norm(dim=-1)
    self.metrics['knee_contact_fraction'] = (force.amax(dim=1) > 5.).float().mean(-1)
    self.metrics['knee_force_peak'] = force.flatten(1).amax(-1)


def three_stage_env_cfg(bank_file, *, stage=0, num_envs=4096, split='train',
    motion_file=None, nominal_evaluation=False, evaluation=False):
  if type(stage) is not int or not 0 <= stage <= 2:
    raise ValueError('Three-stage course must be 0, 1 or 2')
  cfg = entry_course_env_cfg(bank_file, num_envs=num_envs, split=split,
    motion_file=motion_file or V3_MOTION)
  old = cfg.commands['motion']
  cfg.commands['motion'] = ThreeStageCommandCfg(**{f.name: getattr(old, f.name) for f in fields(old)}, stage=stage)
  cmd = cfg.commands['motion']
  cmd.debug_vis = False
  alpha = STAGES[stage]['randomization']
  cfg.observations['actor'].enable_corruption = alpha > 0 and not evaluation and not nominal_evaluation
  for term in cfg.observations['actor'].terms.values():
    if term.noise is not None:
      term.noise.n_min *= alpha
      term.noise.n_max *= alpha
  if alpha == 0:
    cfg.events = {}
    cmd.pose_range = dict(x=(-.005,.005), y=(-.005,.005), z=(-.002,.002),
      roll=(-.02,.02), pitch=(-.02,.02), yaw=(-.03,.03))
    cmd.velocity_range = dict(x=(-.03,.03), y=(-.03,.03), z=(-.02,.02),
      roll=(-.05,.05), pitch=(-.05,.05), yaw=(-.05,.05))
    cmd.joint_position_range = (-.01,.01)
  else:
    cfg.events['base_com'].params['ranges'] = {axis: (-.01*alpha,.01*alpha) for axis in range(3)}
    cfg.events['foot_friction'].params['ranges'] = (1.-.4*alpha, 1.+.2*alpha)
    for name, event in cfg.events.items():
      if name not in ('base_com', 'foot_friction'):
        lo, hi = event.params['bias_range']
        event.params['bias_range'] = (lo*alpha, hi*alpha)
    cmd.pose_range = {k: (a*alpha,b*alpha) for k,(a,b) in cmd.pose_range.items()}
    cmd.velocity_range = {k: (a*alpha,b*alpha) for k,(a,b) in cmd.velocity_range.items()}
    cmd.joint_position_range = tuple(v*alpha for v in cmd.joint_position_range)
  if nominal_evaluation:
    cfg.events = {}
    cmd.entry_probability = 0.
    cmd.nominal_probability = 1.
    # Small course-0 reset perturbations in every retention evaluation.
    cmd.pose_range = dict(x=(-.005,.005),y=(-.005,.005),z=(-.002,.002),
      roll=(-.02,.02),pitch=(-.02,.02),yaw=(-.03,.03))
    cmd.velocity_range = dict(x=(-.03,.03),y=(-.03,.03),z=(-.02,.02),
      roll=(-.05,.05),pitch=(-.05,.05),yaw=(-.05,.05))
    cmd.joint_position_range = (-.01,.01)
  cfg.scene.sensors += (ContactSensorCfg(name='knees_ground_contact',
    primary=ContactMatch(mode='geom', pattern=('RL_thigh.*_collision','RR_thigh.*_collision',
      'RL_calf_collision','RR_calf_collision'), entity='robot'),
    secondary=ContactMatch(mode='geom', pattern='terrain'), fields=('found','force'),
    reduce='netforce', num_slots=1, history_length=4, global_frame=True),)
  cfg.rewards['knee_ground'] = RewardTermCfg(func=knee_ground_cost, weight=-.2)
  cfg.rewards['rise_height'] = RewardTermCfg(func=rise_height_reward, weight=.5,
    params=dict(command_name='motion', std=.06, activation_height=.02, ramp_height=.04))
  for key, angular in (('motion_body_lin_vel',False), ('motion_body_ang_vel',True)):
    cfg.rewards[key].func = terminal_body_velocity_reward
    cfg.rewards[key].params['angular'] = angular
  return cfg
