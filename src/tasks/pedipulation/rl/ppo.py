"""Preserve source PPO details while reusing the rsl_rl 5 update loop."""

from __future__ import annotations

from itertools import chain
from typing import Any

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules.distribution import GaussianDistribution
from tensordict import TensorDict

from .symmetry import exact_symmetry_loss


class PedipulationGaussianDistribution(GaussianDistribution):
  """Direct per-action standard deviation and the source KL expression."""

  def kl_divergence(self, old_params, new_params) -> torch.Tensor:
    old_mean, old_std = old_params
    new_mean, new_std = new_params
    return (
      torch.log(new_std / old_std + 1e-5)
      + (old_std.square() + (old_mean - new_mean).square()) / (2 * new_std.square())
      - 0.5
    ).sum(dim=-1)


class PedipulationPPO(PPO):
  """Source symmetry objective and joint actor/critic gradient clipping.

  rsl_rl 5 has no auxiliary-loss hook and its mirror loss detaches one actor
  branch. During ``update`` only, the instance's optimizer gradient reset is
  wrapped to accumulate the full symmetry gradient before PPO's backward pass.
  A step pre-hook clips the combined actor/critic gradient as in the source.
  All hooks and instance attributes are restored in ``finally``.
  """

  def __init__(self, *args: Any, sym_coef: float = 1.0, **kwargs: Any) -> None:
    if kwargs.get("symmetry_cfg") is not None:
      raise ValueError("PedipulationPPO supplies its own symmetry loss; symmetry_cfg must be None")
    super().__init__(*args, **kwargs)
    if self.actor.is_recurrent or self.critic.is_recurrent:
      raise ValueError("PedipulationPPO requires the source feed-forward policy")
    if self.actor.obs_dim != 45 or tuple(self.actor.obs_groups) != ("actor",):
      raise ValueError("PedipulationPPO requires one 45-dimensional actor observation group")
    if self.critic.obs_dim != 86:
      raise ValueError("PedipulationPPO requires 86 critic observations")
    if self.actor.obs_normalization or self.critic.obs_normalization:
      raise ValueError("The source stand policy does not normalize observations")
    if self.actor.distribution.output_dim != 12:
      raise ValueError("PedipulationPPO requires 12 canonical joint actions")
    self.sym_coef = sym_coef

  def update(self) -> dict[str, float]:
    batch_observations: TensorDict | None = None
    symmetry_losses: list[float] = []
    original_zero_grad = self.optimizer.zero_grad
    had_zero_grad_override = "zero_grad" in self.optimizer.__dict__
    previous_zero_grad_override = self.optimizer.__dict__.get("zero_grad")
    max_grad_norm = self.max_grad_norm

    def capture_batch(module, args, kwargs, output):
      nonlocal batch_observations
      if kwargs.get("stochastic_output", False):
        batch_observations = args[0]

    def zero_grad_with_symmetry(*args, **kwargs):
      original_zero_grad(*args, **kwargs)
      if batch_observations is None:
        raise RuntimeError("rsl_rl PPO changed: no actor minibatch before gradient reset")

      def actor_mean(actor_observations: torch.Tensor) -> torch.Tensor:
        # Only the actor reads this copy. Privileged observations need no mirror.
        mirrored_batch = batch_observations.clone(recurse=False)
        mirrored_batch["actor"] = actor_observations
        return self.actor(mirrored_batch)

      symmetry_loss = exact_symmetry_loss(actor_mean, batch_observations["actor"])
      (self.sym_coef * symmetry_loss).backward()
      symmetry_losses.append(symmetry_loss.detach().item())

    def clip_joint_gradient(optimizer, args, kwargs):
      torch.nn.utils.clip_grad_norm_(
        chain(self.actor.parameters(), self.critic.parameters()), max_grad_norm
      )

    actor_hook = self.actor.register_forward_hook(capture_batch, with_kwargs=True)
    step_hook = self.optimizer.register_step_pre_hook(clip_joint_gradient)
    try:
      self.optimizer.zero_grad = zero_grad_with_symmetry
      # Inherited per-network clipping becomes a no-op before the joint clip.
      self.max_grad_norm = float("inf")
      metrics = super().update()
      expected_updates = self.num_learning_epochs * self.num_mini_batches
      if len(symmetry_losses) != expected_updates:
        raise RuntimeError("rsl_rl PPO changed: unexpected number of gradient resets")
      metrics["symmetry"] = sum(symmetry_losses) / expected_updates
      return metrics
    finally:
      self.max_grad_norm = max_grad_norm
      if had_zero_grad_override:
        self.optimizer.zero_grad = previous_zero_grad_override
      else:
        del self.optimizer.zero_grad
      step_hook.remove()
      actor_hook.remove()
