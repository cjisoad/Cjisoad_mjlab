"""Packed, complete transition episodes; frozen expert steps never enter PPO."""
import torch
from rsl_rl.storage import RolloutStorage


class BridgeRolloutStorage(RolloutStorage):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.collection_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.closed = torch.zeros_like(self.collection_mask)
    self.valid = torch.zeros(self.num_transitions_per_env, self.num_envs,
      dtype=torch.bool, device=self.device)

  def add_transition(self, transition):
    ids = self.collection_mask.nonzero().flatten()
    if not len(ids):
      return
    if self.closed[ids].any():
      raise RuntimeError('Cannot collect into a closed transition episode')
    rows = self.lengths[ids]
    if (rows >= self.num_transitions_per_env).any():
      raise OverflowError('Packed bridge episode exceeded configured capacity')
    for key in self.observations.keys():
      self.observations[key][rows, ids] = transition.observations[key][ids]
    for name in ('actions', 'values', 'rewards', 'dones', 'actions_log_prob'):
      target = getattr(self, name)
      source = getattr(transition, name).reshape(self.num_envs, *target.shape[2:])
      target[rows, ids] = source[ids].to(target.dtype)
    if self.distribution_params is None:
      self.distribution_params = tuple(torch.zeros(self.num_transitions_per_env,
        *p.shape, device=self.device) for p in transition.distribution_params)
    for target, source in zip(self.distribution_params, transition.distribution_params):
      target[rows, ids] = source[ids]
    self.valid[rows, ids] = True
    self.lengths[ids] += 1
    self.step = int(self.lengths.max())

  def finish(self, env_ids, feedback, tail_steps, *, gamma, discount_tail=True):
    """Close each attempt, including source-only rejections (which store nothing)."""
    if self.closed[env_ids].any():
      raise RuntimeError('Outcome was already attributed for this attempt')
    self.closed[env_ids] = True
    present = self.lengths[env_ids] > 0
    ids = env_ids[present]
    rows = self.lengths[ids] - 1
    measured = feedback[present]
    if discount_tail: measured = measured * gamma ** tail_steps[present]
    self.rewards[rows, ids, 0] += measured
    self.dones[rows, ids, 0] = 1

  def compute_closed_returns(self, gamma, lam, normalize=True):
    active = self.lengths > 0
    if (~self.closed & active).any():
      raise RuntimeError('Cannot compute GAE with unresolved target handoffs')
    advantage = torch.zeros(self.num_envs, 1, device=self.device)
    next_value = torch.zeros_like(advantage)
    for step in reversed(range(self.step)):
      valid = self.valid[step, :, None]
      live = 1. - self.dones[step].float()
      delta = self.rewards[step] + gamma * live * next_value - self.values[step]
      current = delta + gamma * lam * live * advantage
      advantage = torch.where(valid, current, advantage)
      next_value = torch.where(valid, self.values[step], next_value)
      self.advantages[step] = torch.where(valid, current, 0.)
      self.returns[step] = torch.where(valid, current + self.values[step], 0.)
    if normalize and self.valid.any():
      a = self.advantages[..., 0][self.valid]
      # A singleton episode still produces finite gradients.
      self.advantages[..., 0][self.valid] = (a-a.mean()) / (a.std(unbiased=a.numel()>1)+1e-8)

  def mini_batch_generator(self, num_mini_batches, num_epochs=8):
    indices = self.valid.flatten().nonzero().flatten()
    if len(indices) < num_mini_batches:
      raise RuntimeError('Too few transition samples for requested PPO minibatches')
    flat_obs = self.observations.flatten(0, 1)
    for _ in range(num_epochs):
      shuffled = indices[torch.randperm(len(indices), device=self.device)]
      for ids in torch.tensor_split(shuffled, num_mini_batches):
        yield self.Batch(observations=flat_obs[ids],
          actions=self.actions.flatten(0, 1)[ids],
          values=self.values.flatten(0, 1)[ids],
          returns=self.returns.flatten(0, 1)[ids],
          advantages=self.advantages.flatten(0, 1)[ids],
          old_actions_log_prob=self.actions_log_prob.flatten(0, 1)[ids],
          old_distribution_params=tuple(p.flatten(0, 1)[ids] for p in self.distribution_params))

  def clear(self):
    super().clear()
    self.lengths.zero_()
    self.valid.zero_()
    self.closed.zero_()
    self.collection_mask.zero_()
