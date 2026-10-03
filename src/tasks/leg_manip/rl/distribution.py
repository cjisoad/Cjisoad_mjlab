"""Bound the actual Gaussian distribution, keeping PPO likelihoods consistent."""
from torch.distributions import Normal
from src.tasks.pedipulation.rl.ppo import PedipulationGaussianDistribution


class StanceGaussianDistribution(PedipulationGaussianDistribution):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.max_std = .3

  def update(self, mlp_output):
    if self.std_type != 'scalar':
      raise ValueError('Stance course requires direct scalar std parameters')
    std = self.std_param.clamp(.02, self.max_std).expand_as(mlp_output)
    self._distribution = Normal(mlp_output, std)
