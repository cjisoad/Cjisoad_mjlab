"""Teacher PPO and persistence without mixed-task curriculum assumptions."""
from dataclasses import dataclass

import torch
from rsl_rl.algorithms import PPO
from mjlab.rl.runner import MjlabOnPolicyRunner
from src.tasks.pedipulation.rl.ppo import PedipulationPPO
from src.tasks.pedipulation.rl.config import PedipulationPpoAlgorithmCfg, pedipulation_ppo_runner_cfg
from src.tasks.pedipulation.rl.symmetry import mirror_actions
from src.tasks.leg_manip.rl.symmetry import mirror_actor_observations
from src.tasks.leg_manip.mdp.foot_workspace import build_foot_workspace
from src.tasks.loco_pedipulation.rl import loco_pedipulation_ppo_runner_cfg
from src.tasks.loco_pedipulation.rl.metrics import ManipulationLogger
from .commands import PedipulationTeacherCommand, LocoPedipulationTeacherCommand
from .anchor_frame import ANCHOR_CONTRACT


def teacher_symmetry_loss(actor, observations):
  """Retain stand symmetry after translating the no-operation command zero."""
  nominal = observations.new_tensor(build_foot_workspace().nominal_biped_offset)
  stance = ((observations[..., 9:12]-nominal).abs().amax(-1) <= 1e-6)
  enabled = stance | (observations[..., 9:12] == 0).all(-1)
  mirrored = mirror_actor_observations(observations)
  # nominal.dy is nonzero because standing straightens the front hip.
  # Mirroring that absolute offset would accidentally request FR operation.
  mirrored[..., 9:12] = torch.where(stance[..., None], nominal, mirrored[..., 9:12])
  original = actor(observations)
  reflected = mirror_actions(actor(mirrored))
  error = (original-reflected).square().mean(-1)
  return (error*enabled).sum()/enabled.sum().clamp_min(1)


class PedipulationTeacherPPO(PedipulationPPO):
  _symmetry_loss = staticmethod(teacher_symmetry_loss)

  def __init__(self, *args, sym_coef=1., **kwargs):
    if kwargs.get('symmetry_cfg') is not None:
      raise ValueError('Teacher PPO supplies source symmetry itself')
    PPO.__init__(self, *args, **kwargs)
    if self.actor.is_recurrent or self.critic.is_recurrent:
      raise ValueError('Teacher PPO requires feed-forward models')
    if self.actor.obs_dim != 51 or tuple(self.actor.obs_groups) != ('actor',):
      raise ValueError('Aligned teacher requires the 51D actor group')
    if self.critic.obs_dim != 89 or self.actor.distribution.output_dim != 12:
      raise ValueError('Standing teacher requires source critic89/action12')
    if self.actor.obs_normalization or self.critic.obs_normalization:
      raise ValueError('Aligned teachers do not normalize observations')
    self.sym_coef = sym_coef


@dataclass
class TeacherPpoAlgorithmCfg(PedipulationPpoAlgorithmCfg):
  class_name: str = 'src.tasks.teacher_common.rl:PedipulationTeacherPPO'


def teacher_ppo_runner_cfg(task):
  if task == 'pedipulation_t':
    cfg = pedipulation_ppo_runner_cfg()
    from dataclasses import asdict
    params = asdict(cfg.algorithm)
    params.pop('class_name')
    cfg.algorithm = TeacherPpoAlgorithmCfg(**params)
  elif task == 'loco_pedipulation_t':
    cfg = loco_pedipulation_ppo_runner_cfg()
  else:
    raise ValueError(f'Unknown teacher task {task}')
  cfg.class_name = 'TeacherOnPolicyRunner'
  cfg.experiment_name = task
  cfg.clip_actions = 10.
  return cfg


class TeacherOnPolicyRunner(MjlabOnPolicyRunner):
  def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
    super().__init__(env, train_cfg, log_dir, device)
    self.teacher_term = env.unwrapped.command_manager.get_term('twist')
    if isinstance(self.teacher_term, PedipulationTeacherCommand):
      self.teacher_task = 'pedipulation_t'
    elif isinstance(self.teacher_term, LocoPedipulationTeacherCommand):
      self.teacher_task = 'loco_pedipulation_t'
    else:
      raise ValueError('Teacher runner requires an aligned teacher command')
    if self.teacher_task == 'loco_pedipulation_t':
      self.logger = ManipulationLogger(self.logger, self.teacher_term.training_metrics, self.is_distributed)

  def _contract(self):
    from src.tasks.leg_manip.constants import BIPED_LOW, TRIPOD_HIGH, BASE_VELOCITY_SCALE
    action = self.env.unwrapped.action_manager.get_term('joint_pos')
    return {'version': 2, 'task': self.teacher_task, 'actor_dim': self.alg.actor.obs_dim,
      'critic_dim': self.alg.critic.obs_dim, 'anchor': ANCHOR_CONTRACT,
      'command': 'vx_vy_wz_FR_offset_from_quadruped_zero',
      'foot_zero': self.teacher_term.zero.cpu().tolist(),
      'overlap_dz': [BIPED_LOW, TRIPOD_HIGH],
      'joint_names': [action._entity.joint_names[i] for i in action.joint_ids.tolist()],
      'default_angles': action.default_angles[0].cpu().tolist(),
      'action_scale': action.cfg.scale, 'clip_actions': self.env.clip_actions,
      'control_delay': action.cfg.delay, 'base_velocity_scale': BASE_VELOCITY_SCALE}

  def save(self, path, infos=None):
    info = {**(infos or {}), 'teacher_contract': self._contract()}
    if self.teacher_task == 'loco_pedipulation_t':
      term = self.teacher_term
      info['loco_pedipulation'] = {'stage': term.stage, 'stage_started': term.stage_started,
        'window': term.curriculum_window.clone(), 'episodes': term.curriculum_episodes,
        'landing_counts': self.logger.landing_totals.clone()}
    # VelocityOnPolicyRunner assumes a generic JointPositionAction in its ONNX
    # metadata. Save custom-action checkpoints with the base implementation.
    super().save(path, info)

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    contract = (checkpoint.get('infos') or {}).get('teacher_contract')
    if contract != self._contract():
      raise ValueError('Checkpoint is not an aligned teacher with this task/command/action contract')
    infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    term = self.teacher_term
    state = (infos or {}).get('loco_pedipulation')
    if state and self.teacher_task == 'loco_pedipulation_t' and term.cfg.curriculum_enabled:
      term.stage = state['stage']
      term.stage_started = state['stage_started']
      term.curriculum_window.copy_(state['window'])
      term.curriculum_episodes = state['episodes']
      self.logger.restore_counts(state['landing_counts'])
    return infos
