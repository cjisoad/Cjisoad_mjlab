"""Learned dense actor experts with the existing leg_manip Gaussian policy."""
from dataclasses import asdict, dataclass

from mjlab.rl import RslRlModelCfg
from rsl_rl.models import MLPModel

from .vendor.ac_moe_gated import GatedMoENet


@dataclass
class DenseMoEActorCfg(RslRlModelCfg):
  class_name: str = 'src.tasks.moe_leg_manip.rl.moe:DenseMoEActor'
  num_experts: int = 4
  gate_hidden_dims: tuple[int, ...] = (128,)
  top_k: int = -1
  use_explicit_expert: bool = False


class DenseMoEActor(MLPModel):
  """Change only the action mean network; keep per-joint Gaussian std and PPO."""

  def __init__(self, obs, obs_groups, obs_set, output_dim,
               hidden_dims=(512, 256, 128), activation='elu', obs_normalization=False,
               distribution_cfg=None, num_experts=4, gate_hidden_dims=(128,),
               top_k=-1, use_explicit_expert=False):
    if use_explicit_expert or top_k not in (None, -1) or num_experts != 4:
      raise ValueError('This experiment requires four learned experts with dense routing')
    super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
      obs_normalization, dict(distribution_cfg) if distribution_cfg is not None else None)
    self.mlp = GatedMoENet(self.obs_dim, output_dim, hidden_dims,
      gate_hidden_dims=list(gate_hidden_dims), activation=activation, num_experts=num_experts,
      top_k=-1, use_shared_layers=False, use_gate_loss=False, use_load_balance_loss=False)

  def _export(self, build):
    # Router caches can hold non-leaf tensors from the latest PPO forward pass.
    names = ('_last_gate_weights', '_last_unmasked_gate_weights', '_last_component_outputs')
    caches = {name: getattr(self.mlp, name) for name in names}
    try:
      for name, value in caches.items(): setattr(self.mlp, name, value.new_empty(0))
      return build()
    finally:
      for name, value in caches.items(): setattr(self.mlp, name, value)

  def as_jit(self):
    return self._export(super().as_jit)

  def as_onnx(self, verbose):
    return self._export(lambda: super(DenseMoEActor, self).as_onnx(verbose))


def moe_leg_manip_ppo_runner_cfg():
  from . import leg_manip_ppo_runner_cfg
  cfg = leg_manip_ppo_runner_cfg()
  cfg.actor = DenseMoEActorCfg(**{**asdict(cfg.actor),
    'class_name': DenseMoEActorCfg.class_name})
  cfg.experiment_name = 'moe_leg_manip'
  return cfg


def checkpoint_uses_moe(checkpoint):
  """Support legacy MLP and new MoE keyboard playback without altering models."""
  return any(name.startswith('mlp.experts.') for name in checkpoint.get('actor_state_dict', {}))
