"""Explicit migration of original leg_manip checkpoints to velocity inputs."""

from copy import deepcopy
import math

import torch

from ..constants import (
  ACTOR_OBS_DIM, BASE_VELOCITY_NOISE, BASE_VELOCITY_SCALE, CRITIC_OBS_DIM, LEGACY_ACTOR_DIM,
)


def observation_layout():
  return {'version': 2, 'actor_dim': ACTOR_OBS_DIM, 'critic_dim': CRITIC_OBS_DIM,
    'velocity_indices': [LEGACY_ACTOR_DIM, ACTOR_OBS_DIM],
    'velocity_quantity': 'root_link_lin_vel_w', 'velocity_frame': 'leg_manip_anchor',
    'velocity_scale': BASE_VELOCITY_SCALE, 'actor_velocity_noise_mps': BASE_VELOCITY_NOISE,
    'critic_velocity_noise_mps': 0.}


def migrate_velocity_checkpoint(checkpoint):
  """Preserve initial actor/value outputs; start fresh Adam and promotion windows."""
  actor_weight = checkpoint['actor_state_dict']['mlp.0.weight']
  critic_weight = checkpoint['critic_state_dict']['mlp.0.weight']
  if actor_weight.shape[-1] != LEGACY_ACTOR_DIM or critic_weight.shape[-1] != CRITIC_OBS_DIM:
    raise ValueError('Migration requires an original 48D actor / 131D leg_manip critic')
  infos = checkpoint.get('infos') or {}
  if 'loco_pedipulation' not in infos or 'env_state' not in infos:
    raise ValueError('Migration requires persisted leg_manip curriculum and environment state')
  migrated = deepcopy(checkpoint)
  migrated['actor_state_dict']['mlp.0.weight'] = torch.cat((actor_weight.clone(),
    actor_weight.new_zeros(actor_weight.shape[0], ACTOR_OBS_DIM - LEGACY_ACTOR_DIM)), dim=-1)
  migrated['critic_state_dict']['mlp.0.weight'][:, LEGACY_ACTOR_DIM:ACTOR_OBS_DIM] /= BASE_VELOCITY_SCALE
  # Parameter shapes changed; old Adam moments cannot be restored as-is.
  migrated['optimizer_state_dict']['state'] = {}
  state = migrated['infos']['loco_pedipulation']
  state['window'].zero_()
  state['episodes'] = 0
  state['stage_started'] = migrated['infos']['env_state']['common_step_counter']
  migrated['infos']['leg_manip_observations'] = observation_layout()
  return migrated


def course_state(term, landing_counts, switch_counts=None):
  """Persist the versioned course independently of shared source-task state."""
  from ..constants import COURSE_NAME, COURSE_VERSION
  state = {'version': COURSE_VERSION, 'name': COURSE_NAME, 'stage': term.stage,
    'stage_started': term.stage_started, 'window': term.curriculum_window.clone(),
    'switches': term.switch_window.clone(), 'episodes': term.curriculum_episodes,
    'failures': term.curriculum_failures, 'pass_streak': term.pass_streak,
    'velocity_level': term.velocity_level, 'landing_counts': landing_counts.clone(),
    'group_survival': term.group_survival.clone(), 'last_decision': dict(term.last_decision)}
  if switch_counts is not None:
    state['switch_counts'] = switch_counts.clone()
  return state


def validate_course_state(state):
  from ..constants import COURSE_NAME, COURSE_VERSION, STAGE_BIPED
  if not isinstance(state, dict) or state.get('version') != COURSE_VERSION or state.get('name') != COURSE_NAME:
    raise ValueError('Checkpoint has no compatible leg_manip stance course version; start a fresh course run')
  for key, upper in (('stage', STAGE_BIPED), ('velocity_level', 3),
                     ('stage_started', None), ('episodes', None), ('failures', None), ('pass_streak', None)):
    value = state.get(key)
    if not isinstance(value, int) or value < 0 or (upper is not None and value > upper):
      raise ValueError(f'Invalid course checkpoint {key}: {value}')
  for key, shape in (('window', (4, 3)), ('switches', (2, 2)), ('landing_counts', (4,)), ('group_survival', (4, 2))):
    value = state.get(key)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all() or (value < 0).any():
      raise ValueError(f'Invalid course checkpoint buffer {key}')
  for key, value, shape in (('switch_counts', state.get('switch_counts'), (2, 5)),
                            ('landing_counts', state['landing_counts'], (4,))):
    if key == 'switch_counts' and key not in state:
      continue
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.isfinite(value).all() or (value < 0).any():
      raise ValueError(f'Invalid course checkpoint {key}')
    if (value[..., 1:].sum(-1) > value[..., 0]).any():
      raise ValueError(f'Course checkpoint {key} has more outcomes than attempts')
  if (state['window'][:, 1] > state['window'][:, 0]).any() or (state['switches'][:, 1] > state['switches'][:, 0]).any():
    raise ValueError('Course checkpoint successes exceed samples or attempts')
  visits = state['group_survival']
  if (visits[:, 1] > visits[:, 0]).any():
    raise ValueError('Course checkpoint group failures exceed episodes')
  decision = state.get('last_decision')
  if not isinstance(decision, dict) or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in decision.values()):
    raise ValueError('Invalid course checkpoint last_decision')
  return state


def restore_course_state(term, state):
  validate_course_state(state)
  for attr, key in (('stage', 'stage'), ('stage_started', 'stage_started'),
                    ('curriculum_episodes', 'episodes'), ('curriculum_failures', 'failures'),
                    ('pass_streak', 'pass_streak'), ('velocity_level', 'velocity_level')):
    setattr(term, attr, state[key])
  term.curriculum_window.copy_(state['window'])
  term.switch_window.copy_(state['switches'])
  term.group_survival.copy_(state['group_survival'])
  term.last_decision = dict(state['last_decision'])
