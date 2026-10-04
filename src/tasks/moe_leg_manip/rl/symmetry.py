"""Source left/right symmetry extended with horizontal anchor base velocity."""

import torch

from src.tasks.pedipulation.rl.symmetry import (
  exact_symmetry_loss as source_symmetry_loss,
  mirror_actor_observations as source_mirror,
)
from ..constants import ACTOR_OBS_DIM, LEGACY_ACTOR_DIM


def mirror_actor_observations(observations):
  if observations.shape[-1] != ACTOR_OBS_DIM:
    raise ValueError(f'Expected {ACTOR_OBS_DIM} actor observations')
  return torch.cat((source_mirror(observations[..., :LEGACY_ACTOR_DIM]),
    observations[..., LEGACY_ACTOR_DIM:] * observations.new_tensor((1., -1., 1.))), dim=-1)


def exact_symmetry_loss(actor, observations):
  return source_symmetry_loss(actor, observations, mirror_observations=mirror_actor_observations)
