"""Left/right regularization for target-free 48D actor observations."""

from collections.abc import Callable

import torch


_JOINT_INDICES = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
_JOINT_SIGNS = (-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1)
_OBS_INDICES = tuple(range(12)) + tuple(
  offset + index for offset in (12, 24, 36) for index in _JOINT_INDICES
)
_OBS_SIGNS = (-1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1) + _JOINT_SIGNS * 3


def _reflect(values: torch.Tensor, indices: tuple, signs: tuple) -> torch.Tensor:
  if values.shape[-1] != len(indices):
    raise ValueError(f"Expected last dimension {len(indices)}, got {values.shape[-1]}")
  return values[..., list(indices)] * values.new_tensor(signs)


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
  """Reflect FL/FR/RL/RR hip-thigh-calf actions, reversing hip signs."""
  return _reflect(actions, _JOINT_INDICES, _JOINT_SIGNS)


def mirror_actor_observations(observations: torch.Tensor) -> torch.Tensor:
  """Reflect body angular velocity, gravity, commands, joints and actions."""
  return _reflect(observations, _OBS_INDICES, _OBS_SIGNS)


def exact_symmetry_loss(
  actor: Callable[[torch.Tensor], torch.Tensor], observations: torch.Tensor
) -> torch.Tensor:
  """Mirror only target-free samples; a right-only goal has no left counterpart."""
  original = actor(observations)
  reflected = mirror_actions(actor(mirror_actor_observations(observations)))
  enabled = (observations[..., 9:12] == 0).all(-1)
  error = (original - reflected).square().mean(-1)
  return (error * enabled).sum() / enabled.sum().clamp_min(1)
