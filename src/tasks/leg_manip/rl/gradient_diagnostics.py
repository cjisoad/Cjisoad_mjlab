"""Read-only, sampled shared-actor gradients from the completed PPO rollout."""

import torch

from ..mdp.commands import HOLD, QUAD


def classify_command_groups(term):
  """Label the pre-action command context; -1 excludes reward transitions."""
  groups = torch.full_like(term.mode, -1, dtype=torch.long)
  quad = (term.phase == QUAD) & ~term.mode & (term.group_weight <= .01)
  biped = (term.phase == HOLD) & term.mode & (term.group_weight >= .99)
  moving = term.command[:, :3].norm(dim=-1) > .05
  groups[quad] = 2 * moving[quad].long()
  groups[biped] = 1 + 2 * moving[biped].long()
  return groups


def _gradient_vectors(ppo, observations, actions, old_log_prob, advantages):
  """Match the PPO/symmetry actor objectives without sampling or .backward()."""
  parameters = tuple(p for p in ppo.actor.mlp.parameters() if p.requires_grad)
  distribution = ppo.actor.distribution
  previous_distribution = distribution._distribution
  try:
    mean = ppo.actor(observations, stochastic_output=False)
    distribution.update(mean)
    log_prob = ppo.actor.get_output_log_prob(actions)
    ratio = torch.exp(log_prob - old_log_prob)
    surrogate = torch.maximum(-advantages * ratio,
      -advantages * ratio.clamp(1. - ppo.clip_param, 1. + ppo.clip_param)).mean()

    def flatten(loss, retain_graph=False):
      gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
      return torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1)
                        for p, g in zip(parameters, gradients)])

    if ppo.sym_coef:
      def actor_mean(states):
        batch = observations.clone(recurse=False)
        batch['actor'] = states
        return ppo.actor(batch, stochastic_output=False)
      symmetry = ppo._symmetry_loss(actor_mean, observations['actor'])
      policy = flatten(surrogate, retain_graph=True)
      total = flatten(surrogate + ppo.sym_coef * symmetry)
    else:
      policy = flatten(surrogate)
      total = policy
    return {'policy': policy, 'actor_total': total}
  finally:
    distribution._distribution = previous_distribution


class GradientConflictDiagnostics:
  """Compare group-mean gradients; keep a cumulative valid negative fraction."""

  PAIRS = (
    ('quad_biped', 'quad', 'biped'),
    ('quad_stand_biped_stand', 'quad_stand', 'biped_stand'),
    ('quad_moving_biped_moving', 'quad_moving', 'biped_moving'),
    ('quad_stand_quad_moving', 'quad_stand', 'quad_moving'),
    ('biped_stand_biped_moving', 'biped_stand', 'biped_moving'),
  )

  def __init__(self, interval=20, max_samples=512):
    if interval < 0 or max_samples < 2:
      raise ValueError('Gradient diagnostics require interval>=0 and max_samples>=2')
    self.interval, self.max_samples = interval, max_samples
    self.groups = []
    self.updates = 0
    self.counts = {}

  @property
  def due(self):
    return self.interval > 0 and self.updates % self.interval == 0

  def capture(self, term):
    if self.due:
      self.groups.append(classify_command_groups(term).detach().clone())

  def evaluate(self, ppo):
    metrics = {}
    try:
      if not self.due or not self.groups:
        return metrics
      storage = ppo.storage
      if len(self.groups) != storage.step:
        raise RuntimeError('Gradient mode labels are not aligned with the PPO rollout')
      if ppo.normalize_advantage_per_mini_batch or ppo.is_multi_gpu:
        raise ValueError('Gradient diagnostics currently require global advantages and a single GPU')
      labels = torch.stack(self.groups).flatten()
      size = labels.numel()
      observations = storage.observations.flatten(0, 1)[:size]
      actions = storage.actions.flatten(0, 1)[:size]
      log_prob = storage.actions_log_prob.flatten()[:size]
      advantages = storage.advantages.flatten()[:size]
      masks = {'quad': (labels == 0) | (labels == 2), 'biped': (labels == 1) | (labels == 3)}
      masks.update({name: labels == i for i, name in enumerate(
        ('quad_stand', 'biped_stand', 'quad_moving', 'biped_moving'))})
      # A private generator leaves training's CPU/CUDA sampling streams untouched.
      generator = torch.Generator(device=actions.device).manual_seed(1729 + self.updates)
      vectors = {}
      with torch.enable_grad():
        for name, mask in masks.items():
          indices = mask.nonzero().flatten()
          metrics['samples/' + name] = float(indices.numel())
          if indices.numel() < 2:
            continue
          if indices.numel() > self.max_samples:
            indices = indices[torch.randperm(indices.numel(), device=indices.device,
              generator=generator)[:self.max_samples]]
          metrics['used_samples/' + name] = float(indices.numel())
          vectors[name] = _gradient_vectors(ppo, observations[indices], actions[indices],
            log_prob[indices], advantages[indices])
      for pair, first, second in self.PAIRS:
        for objective in ('policy', 'actor_total'):
          prefix = pair + '/' + objective + '/'
          metrics[prefix + 'valid'] = 0.
          if first not in vectors or second not in vectors:
            continue
          a, b = vectors[first][objective], vectors[second][objective]
          norm_a, norm_b = a.norm(), b.norm()
          metrics[prefix + 'first_norm'] = float(norm_a)
          metrics[prefix + 'second_norm'] = float(norm_b)
          if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise FloatingPointError('Nonfinite actor gradient in diagnostic')
          if float(norm_a) <= 1e-12 or float(norm_b) <= 1e-12:
            continue
          cosine = float((torch.dot(a, b) / (norm_a * norm_b)).clamp(-1., 1.))
          count, negatives = self.counts.get(prefix, (0, 0))
          count += 1; negatives += int(cosine < 0.)
          self.counts[prefix] = count, negatives
          metrics[prefix + 'valid'] = 1.
          metrics[prefix + 'cosine'] = cosine
          metrics[prefix + 'negative_fraction'] = negatives / count
          metrics[prefix + 'measurements'] = float(count)
      return metrics
    finally:
      self.groups.clear()
      self.updates += 1
