"""Original Go2_Handstand PPO hyperparameters in mjlab's configuration API."""

from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class StandPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = f"{__package__}.ppo:StandPPO"
  sym_coef: float = 1.0


def stand_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    seed=1,
    class_name="StandOnPolicyRunner",
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": f"{__package__}.ppo:StandGaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128), activation="elu", obs_normalization=False,
    ),
    algorithm=StandPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      normalize_advantage_per_mini_batch=False,
      sym_coef=1.0,
    ),
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    experiment_name="go2_stand",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=15000,
    clip_actions=100.0,
    logger="wandb",
    upload_model=False,
  )
