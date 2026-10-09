"""Frozen aligned teachers queried in their original joint coordinates.

The bridge action term owns PD, delay, and motor offsets. These helpers only
rebase raw actions; applying the returned action once preserves the teacher's
physical joint-position target. Teacher clipping happens before this rebase.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from src.tasks.teacher_common.anchor_frame import ANCHOR_CONTRACT, anchor_basis, to_anchor


JOINT_NAMES = tuple(f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR')
                    for joint in ('hip','thigh','calf'))
QUADRUPED_DEFAULT_ANGLES = (-.1,.9,-1.8,.1,.9,-1.8,-.1,.9,-1.8,.1,.9,-1.8)
BIPED_DEFAULT_ANGLES = (.1,.8,-1.5,-.1,.8,-1.5,.1,1.,-1.5,-.1,1.,-1.5)
COMMON_FOOT_ZERO = (.17782151699066162,-.17260202765464783,-.3002205789089203)
ACTION_SCALE = .25
TEACHER_CLIP = 10.
_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QUADRUPED_CHECKPOINT = _ROOT / 'outputs/new_teachers_review_20261006/quadruped/checkpoints/model_9900.pt'
DEFAULT_BIPED_CHECKPOINT = _ROOT / 'outputs/new_teachers_review_20261006/biped/checkpoints/model_14600.pt'
_TASKS = {'quadruped':'loco_pedipulation_t', 'biped':'pedipulation_t'}


def _finite(value, name):
  if not torch.isfinite(value).all():
    raise ValueError(f'{name} contains nonfinite values')


def common_previous_to_teacher(common_previous, common_zero, teacher_zero, scale=ACTION_SCALE):
  """Express the last common action as the same delayed teacher PD target.

  Do not clip this observation: previous actions from the other controller can
  legitimately fall outside the teacher's own output bounds.
  """
  if scale <= 0:
    raise ValueError('action scale must be positive')
  return common_previous + (common_zero-teacher_zero)/scale


def teacher_action_to_common(teacher_raw, teacher_zero, common_zero, scale=ACTION_SCALE,
                             teacher_clip=TEACHER_CLIP, common_clip=None):
  """Clip in teacher coordinates, then preserve the physical PD target.

  A caller with a common action limit may request validation. An incompatible
  limit raises instead of silently changing the expert's physical target.
  The bridge applies these actions after its policy wrapper's clipping.
  """
  if scale <= 0 or teacher_clip <= 0:
    raise ValueError('action scale and teacher clip must be positive')
  _finite(teacher_raw,'teacher action')
  common = teacher_raw.clamp(-teacher_clip,teacher_clip) + (teacher_zero-common_zero)/scale
  _finite(common,'common action')
  if common_clip is not None:
    if common_clip <= 0:
      raise ValueError('common action clip must be positive')
    if (common.abs() > common_clip+1e-6).any():
      raise ValueError('translated teacher target exceeds common action clip; '
                       'apply expert actions after policy clipping or increase the common limit')
  return common


def validate_teacher_contract(contract, expected_task, *, actor_dim=51):
  """Require the approved observation, command, action, and nominal-pose ABI."""
  if expected_task not in _TASKS.values():
    raise ValueError(f'unknown teacher task {expected_task!r}')
  if not isinstance(contract,Mapping):
    raise ValueError('teacher_contract is missing or invalid')
  biped = expected_task == 'pedipulation_t'
  if actor_dim not in ((51,) if biped else (51,54)):
    raise ValueError('Standing teachers require actor51; locomotion teachers require actor51 or actor54')
  required = {'version':2, 'task':expected_task, 'actor_dim':actor_dim,
    'critic_dim':89 if biped else 95, 'anchor':ANCHOR_CONTRACT,
    'command':'vx_vy_wz_FR_offset_from_quadruped_zero',
    'joint_names':list(JOINT_NAMES), 'action_scale':ACTION_SCALE,
    'clip_actions':TEACHER_CLIP, 'control_delay':True, 'base_velocity_scale':2.}
  for name,expected in required.items():
    value = contract.get(name)
    if value != expected or (name == 'control_delay' and value is not True):
      raise ValueError(f'teacher_contract {name}: expected {expected!r}, got {value!r}')
  vectors = {'foot_zero':COMMON_FOOT_ZERO,
    'default_angles':BIPED_DEFAULT_ANGLES if biped else QUADRUPED_DEFAULT_ANGLES,
    'overlap_dz':(.25,.35)}
  for name,expected in vectors.items():
    try:
      actual = torch.as_tensor(contract.get(name),dtype=torch.float64)
      target = torch.tensor(expected,dtype=torch.float64)
      valid = actual.shape == target.shape and torch.allclose(actual,target,atol=1e-6,rtol=0.)
    except (TypeError,ValueError,RuntimeError):
      valid = False
    if not valid:
      raise ValueError(f'teacher_contract {name} does not match approved teacher coordinates')


class FrozenTeacherActor(nn.Module):
  """Frozen 51/54→512→256→128→12 ELU mean; bridge pairs default to51."""
  def __init__(self, actor_state, *, input_dim=51):
    super().__init__()
    if input_dim not in (51,54):
      raise ValueError('Frozen teacher input dimension must be51 or54')
    self.input_dim=input_dim
    if not isinstance(actor_state,Mapping):
      raise ValueError('checkpoint actor_state_dict is missing')
    # Construction must not consume the bridge training RNG stream.
    with torch.random.fork_rng(devices=[]):
      self.mlp = nn.Sequential(nn.Linear(input_dim,512),nn.ELU(),nn.Linear(512,256),
        nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,12))
    required = set(self.state_dict()) | {'distribution.std_param'}
    if set(actor_state) != required:
      raise ValueError('checkpoint actor_state_dict is not the approved unnormalized ELU actor')
    std = actor_state['distribution.std_param']
    if not isinstance(std,torch.Tensor) or std.shape != (12,):
      raise ValueError('checkpoint distribution.std_param must have 12 entries')
    for name,value in actor_state.items():
      if not isinstance(value,torch.Tensor):
        raise ValueError(f'checkpoint actor_state_dict {name} must be a tensor')
      _finite(value,f'checkpoint actor_state_dict {name}')
    try:
      self.load_state_dict({name:value for name,value in actor_state.items()
                           if name != 'distribution.std_param'},strict=True)
    except RuntimeError as error:
      raise ValueError(f'checkpoint actor architecture differs from {input_dim}/512/256/128/12') from error
    self.requires_grad_(False)
    self.eval()

  def train(self, mode=True):
    """Keep teachers in evaluation mode even when a parent is trained."""
    return super().train(False)

  @torch.inference_mode()
  def forward(self, observations):
    if observations.ndim != 2 or observations.shape[-1] != self.input_dim:
      raise ValueError(f'teacher observations must have shape [N, {self.input_dim}]')
    _finite(observations,'teacher observations')
    output = self.mlp(observations)
    _finite(output,'teacher output')
    return output


class FrozenTeacherPair:
  """Query frozen experts from the current simulation state or a selected batch.

  Provide an environment, or ``robot`` and ``action`` explicitly. Commands are
  six physical values: vx, vy, wz, FR dx, dy, dz from the shared quadruped zero.
  Queries accept full-environment commands or commands already selected by
  ``env_ids``. They do not mutate state, action history, noise, or motor offsets.
  """
  def __init__(self, env=None, *, robot=None, action=None,
               quadruped_checkpoint=DEFAULT_QUADRUPED_CHECKPOINT,
               biped_checkpoint=DEFAULT_BIPED_CHECKPOINT,
               command_name='twist', action_name='joint_pos', common_clip=None):
    self.env = env
    if env is not None:
      robot = env.scene['robot'] if robot is None else robot
      action = env.action_manager.get_term(action_name) if action is None else action
    if robot is None or action is None:
      raise ValueError('provide an environment or both robot and action')
    self.robot,self.action = robot,action
    self.command_name,self.common_clip = command_name,common_clip
    if action.cfg.scale != ACTION_SCALE:
      raise ValueError('common action scale must equal teacher scale .25')
    self.device = action.default_angles.device
    self.joint_ids = torch.as_tensor(action.joint_ids,device=self.device,dtype=torch.long)
    ordered = tuple(robot.joint_names[i] for i in self.joint_ids.tolist())
    if ordered != JOINT_NAMES:
      raise ValueError(f'common action joint order does not match teachers: {ordered!r}')
    self.actors,self.contracts,self.teacher_zeros = {},{},{}
    for name,path in (('quadruped',quadruped_checkpoint),('biped',biped_checkpoint)):
      checkpoint = torch.load(Path(path).expanduser(),map_location='cpu',weights_only=False)
      if not isinstance(checkpoint,Mapping):
        raise ValueError('teacher checkpoint must be a mapping')
      contract = (checkpoint.get('infos') or {}).get('teacher_contract')
      validate_teacher_contract(contract,_TASKS[name])
      self.actors[name] = FrozenTeacherActor(checkpoint.get('actor_state_dict')).to(self.device)
      self.contracts[name] = dict(contract)
      self.teacher_zeros[name] = action.default_angles.new_tensor(contract['default_angles'])

  def _teacher(self, name):
    if name not in self.actors:
      raise ValueError(f'teacher must be quadruped or biped, got {name!r}')
    return self.actors[name],self.teacher_zeros[name]

  def _ids(self, env_ids):
    if env_ids is None or isinstance(env_ids,slice):
      return slice(None) if env_ids is None else env_ids
    return torch.as_tensor(env_ids,device=self.device).reshape(-1)

  def _common_zero(self, ids):
    zero = self.action.default_angles
    return zero if zero.ndim == 1 else zero[ids]

  def observations(self, teacher, command=None, env_ids=None):
    """Build clean original actor51 channels using the actual current state."""
    _,teacher_zero = self._teacher(teacher)
    ids = self._ids(env_ids)
    data = self.robot.data
    position = data.joint_pos[ids][:,self.joint_ids]
    if command is None:
      if self.env is None:
        raise ValueError('commands are required when constructed without an environment')
      command = self.env.command_manager.get_command(self.command_name)
    command = torch.as_tensor(command,device=self.device,dtype=position.dtype)
    if command.ndim != 2 or command.shape[-1] != 6:
      raise ValueError('teacher commands must have shape [N, 6]')
    if command.shape[0] == data.joint_pos.shape[0]:
      command = command[ids]
    if command.shape[0] != position.shape[0]:
      raise ValueError('teacher command batch does not match selected environments')
    basis = anchor_basis(data.root_link_quat_w[ids],data.gravity_vec_w[ids]
      if data.gravity_vec_w.ndim == 2 and data.gravity_vec_w.shape[0] > 1 else data.gravity_vec_w)
    previous = common_previous_to_teacher(self.action.raw_action[ids],
      self._common_zero(ids),teacher_zero)
    original = torch.cat((data.root_link_ang_vel_b[ids]*.25,data.projected_gravity_b[ids],
      command*command.new_tensor((2.,2.,.25,1.,1.,1.)),position-teacher_zero,
      data.joint_vel[ids][:,self.joint_ids]*.05,previous),-1)
    # Source clamps its first 48 channels before appending base velocity.
    obs = torch.cat((original.clamp(-100.,100.),
                     to_anchor(basis,data.root_link_lin_vel_w[ids])*2.),-1)
    _finite(obs,'teacher observations')
    return obs

  def actor_actions(self, teacher, observations):
    """Return deterministic original actor means before teacher clipping."""
    actor,_ = self._teacher(teacher)
    return actor(observations)

  @torch.inference_mode()
  def actions(self, teacher, command=None, env_ids=None):
    """Return rebased raw actions for the selected environments."""
    _,teacher_zero = self._teacher(teacher)
    observations = self.observations(teacher,command,env_ids)
    raw = self.actor_actions(teacher,observations)
    return teacher_action_to_common(raw,teacher_zero,self._common_zero(self._ids(env_ids)),
                                   common_clip=self.common_clip)
