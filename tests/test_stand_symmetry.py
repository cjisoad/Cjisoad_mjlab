"""CPU-only regression tests for the source Go2 stand PPO semantics."""

from __future__ import annotations

import copy
from dataclasses import asdict
import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage


RL_DIR = Path(__file__).resolve().parents[1] / "src/tasks/stand/rl"
SPEC = importlib.util.spec_from_file_location(
  "_stand_test_rl", RL_DIR / "__init__.py", submodule_search_locations=[str(RL_DIR)]
)
assert SPEC is not None and SPEC.loader is not None
RL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RL
SPEC.loader.exec_module(RL)

from _stand_test_rl.config import stand_ppo_runner_cfg  # noqa: E402
from _stand_test_rl.ppo import StandGaussianDistribution, StandPPO  # noqa: E402
from _stand_test_rl.runner import StandOnPolicyRunner  # noqa: E402
from _stand_test_rl.symmetry import (  # noqa: E402
  exact_symmetry_loss,
  mirror_actions,
  mirror_actor_observations,
)


OBS_PERMUTATION = (
  -0.0001, 1, -2, 3, -4, 5, 6, -7, -8,
  -12, 13, 14, -9, 10, 11, -18, 19, 20, -15, 16, 17,
  -24, 25, 26, -21, 22, 23, -30, 31, 32, -27, 28, 29,
  -36, 37, 38, -33, 34, 35, -42, 43, 44, -39, 40, 41,
)
ACTION_PERMUTATION = (-3, 4, 5, -0.0001, 1, 2, -9, 10, 11, -6, 7, 8)


def source_matrix(permutation):
  matrix = torch.zeros(len(permutation), len(permutation), dtype=torch.float64)
  for index, value in enumerate(permutation):
    matrix[int(abs(value)), index] = -1.0 if value < 0 else 1.0
  return matrix


def observations(num_envs=4):
  return TensorDict(
    {"actor": torch.randn(num_envs, 45), "critic": torch.randn(num_envs, 86)},
    batch_size=[num_envs],
  )


def algorithm():
  torch.manual_seed(11)
  obs = observations()
  groups = {"actor": ["actor"], "critic": ["critic"]}
  actor = MLPModel(
    obs, groups, "actor", 12, hidden_dims=(8, 7),
    distribution_cfg={
      "class_name": StandGaussianDistribution, "init_std": 1.0, "std_type": "scalar",
    },
  )
  critic = MLPModel(obs, groups, "critic", 1, hidden_dims=(8, 7))
  storage = RolloutStorage("rl", 4, 2, obs, [12], "cpu")
  alg = StandPPO(
    actor, critic, storage, num_learning_epochs=1, num_mini_batches=1,
    schedule="fixed", max_grad_norm=0.05, entropy_coef=0.01, sym_coef=1.0,
  )
  with torch.no_grad():
    for _ in range(2):
      alg.act(obs)
      obs = observations()
      alg.process_env_step(obs, torch.randn(4), torch.zeros(4), {})
    alg.compute_returns(obs)
  return alg


class CpuObservationEnv:
  """Exercise runner integration without allocating a physics simulator."""

  num_envs = 4
  num_actions = 12
  device = "cpu"
  cfg = {}
  common_step_counter = 0

  @property
  def unwrapped(self):
    return self

  def get_observations(self):
    return observations(self.num_envs)

  def step(self, actions):
    self.common_step_counter += 1
    return self.get_observations(), -actions.square().mean(-1), torch.zeros(self.num_envs), {}


class StandSymmetryTests(unittest.TestCase):
  def test_mirrors_match_source_matrices_and_are_involutions(self):
    for width, mirror, permutation in (
      (45, mirror_actor_observations, OBS_PERMUTATION),
      (12, mirror_actions, ACTION_PERMUTATION),
    ):
      values = torch.arange(1, 1 + width * 3, dtype=torch.float64).reshape(3, width)
      torch.testing.assert_close(mirror(values), values @ source_matrix(permutation))
      torch.testing.assert_close(mirror(mirror(values)), values)
    self.assertEqual(mirror_actor_observations(torch.ones(1, 45))[0, 0].item(), -1)
    self.assertEqual(mirror_actions(torch.ones(1, 12))[0, 3].item(), -1)

  def test_wrong_dimensions_fail_explicitly(self):
    with self.assertRaises(ValueError):
      mirror_actor_observations(torch.zeros(2, 48))
    with self.assertRaises(ValueError):
      mirror_actions(torch.zeros(2, 13))

  def test_symmetry_gradients_match_both_source_branches(self):
    torch.manual_seed(9)
    model = torch.nn.Sequential(torch.nn.Linear(45, 16), torch.nn.ELU(), torch.nn.Linear(16, 12)).double()
    reference = copy.deepcopy(model)
    values = torch.randn(5, 45, dtype=torch.float64)
    actual = exact_symmetry_loss(model, values)
    expected = (
      reference(values)
      - reference(values @ source_matrix(OBS_PERMUTATION)) @ source_matrix(ACTION_PERMUTATION)
    ).square().mean()
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    for actual_param, expected_param in zip(model.parameters(), reference.parameters()):
      torch.testing.assert_close(actual_param.grad, expected_param.grad)

    detached = copy.deepcopy(reference)
    detached.zero_grad()
    detached_loss = (
      detached(values)
      - mirror_actions(detached(mirror_actor_observations(values))).detach()
    ).square().mean()
    detached_loss.backward()
    self.assertTrue(any(
      not torch.allclose(full.grad, one_sided.grad)
      for full, one_sided in zip(model.parameters(), detached.parameters())
    ))

  def test_kl_keeps_source_epsilon_for_adaptive_learning_rate(self):
    distribution = StandGaussianDistribution(12)
    old = (torch.randn(4, 12), torch.rand(4, 12) + 0.5)
    new = (torch.randn(4, 12), torch.rand(4, 12) + 0.5)
    expected = (
      torch.log(new[1] / old[1] + 1e-5)
      + (old[1].square() + (old[0] - new[0]).square()) / (2 * new[1].square())
      - 0.5
    ).sum(-1)
    torch.testing.assert_close(distribution.kl_divergence(old, new), expected)
    self.assertTrue(torch.all(distribution.kl_divergence(old, old) > 0))

  def test_full_cpu_update_matches_source_gradient_and_joint_clipping(self):
    alg = algorithm()
    actor, critic = copy.deepcopy(alg.actor), copy.deepcopy(alg.critic)
    torch.manual_seed(123)
    batch = next(alg.storage.mini_batch_generator(1, 1))
    actor(batch.observations, stochastic_output=True)
    mean = actor.output_mean
    entropy = actor.output_entropy
    ratio = torch.exp(actor.get_output_log_prob(batch.actions) - batch.old_actions_log_prob.squeeze())
    advantage = batch.advantages.squeeze()
    surrogate = torch.maximum(
      -advantage * ratio, -advantage * ratio.clamp(1 - alg.clip_param, 1 + alg.clip_param)
    ).mean()
    value = critic(batch.observations)
    clipped_value = batch.values + (value - batch.values).clamp(-alg.clip_param, alg.clip_param)
    value_loss = torch.maximum((value - batch.returns).square(), (clipped_value - batch.returns).square()).mean()
    mirrored = batch.observations.clone()
    mirrored["actor"] = mirrored["actor"] @ source_matrix(OBS_PERMUTATION).float()
    symmetry = (mean - actor(mirrored) @ source_matrix(ACTION_PERMUTATION).float()).square().mean()
    loss = surrogate + alg.value_loss_coef * value_loss - alg.entropy_coef * entropy.mean() + alg.sym_coef * symmetry
    loss.backward()
    expected_params = list(actor.parameters()) + list(critic.parameters())
    torch.nn.utils.clip_grad_norm_(expected_params, alg.max_grad_norm)
    before = [parameter.detach().clone() for parameter in alg.actor.parameters()]

    torch.manual_seed(123)
    metrics = alg.update()

    actual_params = list(alg.actor.parameters()) + list(alg.critic.parameters())
    for actual_param, expected_param in zip(actual_params, expected_params):
      torch.testing.assert_close(actual_param.grad, expected_param.grad, atol=1e-7, rtol=1e-5)
    self.assertAlmostEqual(metrics["symmetry"], symmetry.item(), places=6)
    self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
    self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, alg.actor.parameters())))
    self.assertEqual(alg.max_grad_norm, 0.05)
    self.assertNotIn("zero_grad", alg.optimizer.__dict__)
    self.assertEqual(len(alg.actor._forward_hooks), 0)
    self.assertEqual(len(alg.optimizer._optimizer_step_pre_hooks), 0)

  def test_adapter_restores_scoped_changes_when_update_fails(self):
    alg = algorithm()
    with patch("_stand_test_rl.ppo.exact_symmetry_loss", side_effect=RuntimeError("test failure")):
      with self.assertRaisesRegex(RuntimeError, "test failure"):
        alg.update()
    self.assertEqual(alg.max_grad_norm, 0.05)
    self.assertNotIn("zero_grad", alg.optimizer.__dict__)
    self.assertEqual(len(alg.actor._forward_hooks), 0)
    self.assertEqual(len(alg.optimizer._optimizer_step_pre_hooks), 0)

  def test_configuration_matches_original_training_contract(self):
    cfg = stand_ppo_runner_cfg()
    self.assertEqual((cfg.seed, cfg.num_steps_per_env, cfg.max_iterations), (1, 24, 15000))
    self.assertEqual(cfg.actor.hidden_dims, (512, 256, 128))
    self.assertEqual(cfg.critic.hidden_dims, (512, 256, 128))
    self.assertFalse(cfg.actor.obs_normalization)
    self.assertFalse(cfg.critic.obs_normalization)
    self.assertEqual(cfg.clip_actions, 100.0)
    self.assertEqual(cfg.algorithm.sym_coef, 1.0)
    self.assertEqual(cfg.algorithm.entropy_coef, 0.01)
    self.assertEqual(cfg.algorithm.num_learning_epochs, 5)
    self.assertEqual(cfg.algorithm.num_mini_batches, 4)
    self.assertEqual(cfg.algorithm.gamma, 0.99)
    self.assertEqual(cfg.algorithm.lam, 0.95)

  def test_configured_runner_cpu_iteration_checkpoint_and_policy_export(self):
    torch.manual_seed(1)
    env = CpuObservationEnv()
    with redirect_stdout(io.StringIO()):
      runner = StandOnPolicyRunner(env, asdict(stand_ppo_runner_cfg()), device="cpu")
      initial = [parameter.detach().clone() for parameter in runner.alg.actor.parameters()]
      runner.learn(1)
    self.assertEqual(env.common_step_counter, 24)
    self.assertEqual(runner.alg.storage.step, 0)
    self.assertTrue(any(
      not torch.equal(before, after) for before, after in zip(initial, runner.alg.actor.parameters())
    ))
    self.assertTrue(all(torch.isfinite(parameter).all() for parameter in runner.alg.actor.parameters()))
    self.assertEqual(runner.alg.actor.distribution.std_param.shape, (12,))
    self.assertIsInstance(runner.alg.actor.obs_normalizer, torch.nn.Identity)
    self.assertIsInstance(runner.alg.critic.obs_normalizer, torch.nn.Identity)
    with tempfile.TemporaryDirectory() as directory:
      checkpoint = str(Path(directory) / "model.pt")
      runner.save(checkpoint)
      saved_weights = [parameter.detach().clone() for parameter in runner.alg.actor.parameters()]
      with torch.no_grad():
        for parameter in runner.alg.actor.parameters():
          parameter.add_(1)
      env.common_step_counter = 0
      runner.load(checkpoint, map_location="cpu")
      self.assertEqual(env.common_step_counter, 24)
      for restored, saved in zip(runner.alg.actor.parameters(), saved_weights):
        torch.testing.assert_close(restored, saved)
      runner.export_policy_to_jit(directory)
      exported = torch.jit.load(str(Path(directory) / "policy.pt"), map_location="cpu")
      obs = env.get_observations()
      torch.testing.assert_close(exported(obs["actor"]), runner.alg.actor(obs))


if __name__ == "__main__":
  unittest.main()
