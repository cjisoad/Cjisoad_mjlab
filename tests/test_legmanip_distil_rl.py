"""CPU tests of the recurrent RSL-RL distillation contract, without physics."""

from copy import deepcopy
from dataclasses import asdict
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict


@pytest.fixture(autouse=True, scope="module")
def single_torch_thread():
  old = torch.get_num_threads()
  torch.set_num_threads(1)
  yield
  torch.set_num_threads(old)


def observations(count=4, value=0.0):
  return TensorDict({"student": torch.full((count, 48), value)}, batch_size=[count])


def test_default_config_is_distillation_only_and_matches_approved_recipe():
  from mjlab.rl import RslRlBaseRunnerCfg
  from src.tasks.legmanip_distil.rl import LegManipDistillationRunnerCfg

  cfg = LegManipDistillationRunnerCfg()
  assert isinstance(cfg, RslRlBaseRunnerCfg)
  assert cfg.experiment_name == "legmanip_distil"
  assert (cfg.num_steps_per_env, cfg.max_iterations, cfg.save_interval) == (32, 60000, 500)
  assert cfg.logger == "tensorboard"
  assert cfg.clip_actions is None
  assert cfg.obs_groups == {"student": ("student",)}
  assert not {"actor", "critic", "ppo"}.intersection(asdict(cfg))
  alg = cfg.algorithm
  assert (alg.num_learning_epochs, alg.num_mini_batches, alg.gradient_length) == (8, 2, 32)
  assert (alg.optimizer, alg.learning_rate, alg.weight_decay) == ("adamw", 3e-4, 0.01)
  assert (alg.loss_type, alg.max_grad_norm) == ("l1", 1.0)


def test_student_has_exact_architecture_and_unbounded_deterministic_output():
  from rsl_rl.models import RNNModel
  from src.tasks.legmanip_distil.rl import make_student

  model = make_student(num_envs=4)
  assert isinstance(model, RNNModel)
  assert isinstance(model.rnn.rnn, torch.nn.GRU)
  assert (model.rnn.rnn.input_size, model.rnn.rnn.hidden_size, model.rnn.rnn.num_layers) == (48, 256, 1)
  assert sum(p.numel() for p in model.parameters()) == 335244
  assert isinstance(model.obs_normalizer, torch.nn.Identity)
  assert model.distribution is None
  assert not any(isinstance(m, torch.nn.Tanh) for m in model.modules())
  with torch.no_grad():
    for p in model.parameters():
      p.zero_()
    linear = [m for m in model.mlp.modules() if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linear] == [(256, 256), (256, 128), (128, 12)]
    linear[-1].bias.fill_(2.0)
    output = model(observations())
  assert output.shape == (4, 12)
  torch.testing.assert_close(output, torch.full_like(output, 2.0))
  assert model.output_std.shape == (12,)
  assert model.output_std.eq(0).all()


def test_sequence_replay_resets_at_done_without_touching_live_memory():
  from src.tasks.legmanip_distil.rl import make_student

  torch.manual_seed(17)
  model = make_student(num_envs=2)
  with torch.inference_mode():
    model(observations(2, 0.2))
  live_hidden = model.get_hidden_state()
  preserved = live_hidden.clone()
  sequence = torch.randn(4, 2, 48, requires_grad=True)
  dones = torch.tensor([[False, False], [True, False], [False, False], [False, False]])
  initial_hidden = torch.randn(1, 2, 256)
  prediction, final_hidden = model.forward_sequence(sequence, dones, initial_hidden)
  prefix, _ = model.forward_sequence(sequence[:2], dones[:2], initial_hidden)
  fresh, _ = model.forward_sequence(sequence[2:, :1], torch.zeros(2, 1, dtype=torch.bool))
  torch.testing.assert_close(prediction[:2], prefix)
  torch.testing.assert_close(prediction[2:, :1], fresh)
  assert final_hidden.shape == (1, 2, 256)
  assert model.get_hidden_state() is live_hidden
  torch.testing.assert_close(model.get_hidden_state(), preserved)
  prediction[2:, 0].sum().backward()
  assert sequence.grad[:2, 0].eq(0).all()
  assert sequence.grad[2:, 0].abs().sum() > 0


@pytest.mark.parametrize("field,value", [("obs_normalization", True), ("distribution_cfg", {}), ("rnn_type", "lstm")])
def test_student_rejects_architecture_contract_changes(field, value):
  from src.tasks.legmanip_distil.rl import make_student

  with pytest.raises(ValueError):
    make_student(cfg={field: value})


class TensorTeacher(torch.nn.Module):
  """Small frozen teacher standing in for the external FSM/policy assets."""

  loaded = True

  def __init__(self, env=None, **kwargs):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.tensor(1.0))
    self.control_mask = None
    self.value = 1.0
    self.calls = 0

  def forward(self, obs):
    self.calls += 1
    return torch.ones_like(obs["student"][:, :12]) * self.weight * self.value

  def set_control_mask(self, mask):
    self.control_mask = mask.clone()

  def reset(self, dones=None):
    pass

  def get_hidden_state(self):
    return None

  def detach_hidden_state(self, dones=None):
    pass

  def metadata(self):
    return {
      "teacher": "tensor_fixture", "loaded": self.loaded,
      "checkpoints": {"quad": {"path": "/test/policy.pt", "sha256": "unchanged"}},
    }


def make_algorithm(*, count=4, steps=4, **kwargs):
  from rsl_rl.storage import RolloutStorage
  from src.tasks.legmanip_distil.rl import LegManipDistillation, make_student

  storage = RolloutStorage("distillation", count, steps, observations(count), [12], "cpu")
  options = dict(num_learning_epochs=2, num_mini_batches=2, gradient_length=steps)
  options.update(kwargs)
  return LegManipDistillation(make_student(count), TensorTeacher(), storage, **options)


def collect_rollout(algorithm, *, done_step=None):
  count = algorithm.storage.num_envs
  with torch.inference_mode():
    for step in range(algorithm.storage.num_transitions_per_env):
      obs = observations(count, float(step + 1))
      algorithm.act(obs)
      dones = torch.zeros(count, dtype=torch.bool)
      if step == done_step:
        dones[::2] = True
      algorithm.process_env_step(obs, torch.zeros(count), dones, {})


def test_l1_loss_sums_actions_and_means_all_frames():
  from src.tasks.legmanip_distil.rl.algorithm import sequence_l1_loss

  prediction = torch.zeros(4, 2, 12, requires_grad=True)
  target = torch.ones_like(prediction)
  target[1:] = 2.0
  loss = sequence_l1_loss(prediction, target)
  assert loss.item() == pytest.approx(21.0)
  loss.backward()
  assert prediction.grad.ne(0).all()


@pytest.mark.parametrize("completed", [0, 999, 1000, 1001, 3000])
def test_beta_uses_completed_updates(completed):
  algorithm = make_algorithm()
  algorithm.num_updates = completed
  expected = (1.0 - 1.0 / 2000) ** max(completed - 1000, 0)
  assert algorithm.beta == pytest.approx(expected)


def test_control_choice_persists_until_episode_reset_and_student_always_advances():
  from rsl_rl.algorithms import Distillation

  algorithm = make_algorithm(teacher_probability=1.0)
  assert isinstance(algorithm, Distillation)
  assert algorithm.teacher_loaded
  assert not any(p.requires_grad for p in algorithm.teacher.parameters())
  obs = observations(value=0.5)
  with torch.inference_mode():
    action = algorithm.act(obs)
    hidden1 = algorithm.student.get_hidden_state().clone()
    labels = algorithm.transition.privileged_actions.clone()
    torch.testing.assert_close(action, labels)
    algorithm.process_env_step(obs, torch.zeros(4), torch.zeros(4, dtype=torch.bool), {})
    algorithm.teacher_probability = 0.0
    algorithm.act(obs)
    assert algorithm.teacher.control_mask.tolist() == [True, True, True, True]
    assert not torch.equal(hidden1, algorithm.student.get_hidden_state())
    algorithm.process_env_step(obs, torch.zeros(4), torch.tensor([False, True, False, True]), {})
    actual = algorithm.act(obs)
    assert algorithm.teacher.control_mask.tolist() == [True, False, True, False]
    teacher = algorithm.transition.privileged_actions.clone()
    assert not torch.equal(actual[1::2], teacher[1::2])
    torch.testing.assert_close(actual[::2], teacher[::2])
    algorithm.process_env_step(obs, torch.zeros(4), torch.zeros(4, dtype=torch.bool), {})
  torch.testing.assert_close(algorithm.storage.actions[2], actual)
  torch.testing.assert_close(algorithm.storage.privileged_actions[2], teacher)
  assert algorithm.teacher.calls == 3


def test_control_sampling_consumes_rng_before_teacher_previews_action_delay(monkeypatch):
  algorithm = make_algorithm(teacher_probability=0.5)
  real_forward = algorithm.teacher.forward
  preview_rng = []

  def preview(obs):
    labels = real_forward(obs)
    # The real FSM previews the next delay draw, then restores this RNG state.
    preview_rng.append(torch.get_rng_state().clone())
    return labels

  monkeypatch.setattr(algorithm.teacher, "forward", preview)
  with torch.inference_mode():
    algorithm.act(observations())
  torch.testing.assert_close(torch.get_rng_state(), preview_rng[-1], rtol=0, atol=0)


def test_update_uses_every_frame_and_keeps_live_state_separate(monkeypatch):
  algorithm = make_algorithm(learning_rate=0.0, teacher_probability=1.0)
  with torch.no_grad():
    for parameter in algorithm.student.parameters():
      parameter.zero_()
  with torch.inference_mode():
    for step in range(4):
      algorithm.teacher.value = 1.0 if step == 0 else 0.0
      obs = observations(value=float(step + 1))
      algorithm.act(obs)
      algorithm.process_env_step(obs, torch.zeros(4), torch.zeros(4, dtype=torch.bool), {})
  live_hidden = algorithm.student.get_hidden_state()
  preserved = live_hidden.clone()
  sequences = []
  real_forward = algorithm.student.forward_sequence

  def record(observations, dones, hidden_state=None):
    sequences.append(observations.detach().clone())
    return real_forward(observations, dones, hidden_state)

  monkeypatch.setattr(algorithm.student, "forward_sequence", record)
  result = algorithm.update()
  assert result["behavior"] == pytest.approx(3.0)
  assert len(sequences) == 4  # Two epochs, two independent environment batches.
  for batch in sequences:
    assert batch.shape == (4, 2, 48)
    torch.testing.assert_close(batch[:, 0, 0], torch.arange(1, 5, dtype=torch.float))
  assert algorithm.student.get_hidden_state() is live_hidden
  torch.testing.assert_close(algorithm.student.get_hidden_state(), preserved)
  assert algorithm.num_updates == 1
  assert algorithm.storage.step == 0


def test_update_changes_gru_weights_and_preserves_continuing_rollout_memory():
  algorithm = make_algorithm()
  collect_rollout(algorithm, done_step=1)
  before = algorithm.student.rnn.rnn.weight_ih_l0.detach().clone()
  hidden_before = algorithm.student.get_hidden_state().clone()
  result = algorithm.update()
  assert torch.isfinite(torch.tensor(result["behavior"]))
  assert not torch.equal(before, algorithm.student.rnn.rnn.weight_ih_l0)
  torch.testing.assert_close(algorithm.student.get_hidden_state(), hidden_before)
  with torch.inference_mode():
    algorithm.act(observations(value=0.6))
  torch.testing.assert_close(algorithm.rollout_initial_hidden, hidden_before)
  assert algorithm.teacher.weight.grad is None
  assert isinstance(algorithm.optimizer, torch.optim.AdamW)


@pytest.mark.parametrize("options,match", [
  ({"count": 3}, "divisible"),
  ({"num_mini_batches": 0}, "mini"),
  ({"gradient_length": 2}, "gradient_length"),
  ({"teacher_probability": 1.1}, "probability"),
  ({"num_learning_epochs": 0}, "epochs"),
])
def test_algorithm_rejects_invalid_sequence_configuration(options, match):
  with pytest.raises(ValueError, match=match):
    make_algorithm(**options)


def test_algorithm_requires_a_complete_rollout():
  algorithm = make_algorithm()
  with pytest.raises(ValueError, match="complete"):
    algorithm.update()


def test_resume_restores_optimizer_schedule_rng_but_reinitializes_episode_memory():
  algorithm = make_algorithm(teacher_probability=0.3)
  collect_rollout(algorithm)
  algorithm.update()
  algorithm.num_updates = 1400
  saved = deepcopy(algorithm.save())
  assert saved["completed_updates"] == 1400
  assert "teacher_state_dict" not in saved
  expected_random = (torch.rand(3), np.random.random(3), random.random())
  restored = make_algorithm()
  collect_rollout(restored)
  assert restored.student.get_hidden_state() is not None
  assert restored.load(saved, load_cfg=None, strict=True)
  assert restored.num_updates == 1400
  assert restored.beta == pytest.approx(0.3)
  assert restored.optimizer.state_dict()["state"]
  torch.testing.assert_close(restored.student.state_dict(), algorithm.student.state_dict())
  torch.testing.assert_close(torch.rand(3), expected_random[0])
  np.testing.assert_array_equal(np.random.random(3), expected_random[1])
  assert random.random() == expected_random[2]
  assert restored.student.get_hidden_state() is None
  assert restored.storage.step == 0
  assert not restored.teacher_control_mask.any()
  with torch.inference_mode():
    restored.act(observations())
  assert restored.rollout_initial_hidden.eq(0).all()


class TensorEnv:
  device = "cpu"
  num_envs = 4
  num_actions = 12
  max_episode_length = 20
  cfg = {}

  def __init__(self):
    self.unwrapped = self
    self.episode_length_buf = torch.zeros(4, dtype=torch.long)
    self.observed_initial_lengths = None
    self.reset_count = 0
    self.sensor_history = None

  def get_observations(self):
    return TensorDict({"student": self.episode_length_buf.float()[:, None].expand(-1, 48).clone()}, batch_size=[4])

  def step(self, action):
    if self.sensor_history is None:
      # mjlab contact sensors allocate some buffers lazily during rollout.
      self.sensor_history = torch.ones(4)
    if self.observed_initial_lengths is None:
      self.observed_initial_lengths = self.episode_length_buf.clone()
    self.episode_length_buf += 1
    dones = self.episode_length_buf >= 3
    self.episode_length_buf[dones] = 0
    return self.get_observations(), torch.zeros(4), dones, {}

  def reset(self):
    self.reset_count += 1
    if self.sensor_history is not None:
      self.sensor_history.zero_()
    self.episode_length_buf.zero_()
    return self.get_observations(), {}


def small_runner_cfg():
  from src.tasks.legmanip_distil.rl import LegManipDistillationRunnerCfg

  cfg = LegManipDistillationRunnerCfg()
  cfg.num_steps_per_env = 4
  cfg.algorithm.gradient_length = 4
  cfg.algorithm.num_learning_epochs = 1
  cfg.save_interval = 1
  return asdict(cfg)


def test_constructor_uses_real_distillation_storage_and_injected_teacher_without_mutating_cfg():
  from src.tasks.legmanip_distil.rl import LegManipDistillation

  env = TensorEnv()
  cfg = small_runner_cfg()
  before = deepcopy(cfg)
  algorithm = LegManipDistillation.construct_algorithm(env.get_observations(), env, cfg, "cpu", teacher_factory=TensorTeacher)
  assert cfg == before
  assert algorithm.storage.training_type == "distillation"
  assert not hasattr(algorithm.storage, "values")
  assert set(algorithm.storage.observations.keys()) == {"student"}
  assert algorithm.teacher_loaded


def test_runner_logs_saves_resumes_and_exports_student_without_teacher(tmp_path, monkeypatch):
  from rsl_rl.runners import DistillationRunner
  from src.tasks.legmanip_distil.rl import LegManipDistillation, LegManipDistillationRunner, load_student

  construct = LegManipDistillation.construct_algorithm
  monkeypatch.setattr(LegManipDistillation, "construct_algorithm", staticmethod(
    lambda obs, env, cfg, device: construct(obs, env, cfg, device, teacher_factory=TensorTeacher)
  ))
  # Contracts are pure manifests; this fixture avoids importing physics helpers.
  monkeypatch.setitem(sys.modules, "src.tasks.legmanip_distil.observations", SimpleNamespace(
    observation_contract=lambda: {"dimension": 48, "scales": "fixed"},
    action_contract=lambda env: {"dimension": 12, "control_dt": 0.02},
  ))
  env = TensorEnv()
  runner = LegManipDistillationRunner(env, small_runner_cfg(), str(tmp_path), "cpu")
  assert isinstance(runner, DistillationRunner)
  runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
  assert env.sensor_history.is_inference()
  assert env.observed_initial_lengths.eq(0).all()
  assert runner.current_learning_iteration == runner.alg.num_updates == 1
  checkpoint = tmp_path / "model_1.pt"
  assert checkpoint.is_file()
  saved = torch.load(checkpoint, weights_only=False)
  assert saved["iter"] == saved["completed_updates"] == 1
  metadata = saved["infos"]["legmanip_distil"]
  assert metadata["version"] == 1
  assert metadata["student_cfg"]["input_dim"] == 48
  assert metadata["training_cfg"]["algorithm"]["loss_type"] == "l1"
  assert metadata["teacher"]["loaded"]
  artifact = tmp_path / "student.pt"
  assert artifact.is_file()
  # Disable construction of every teacher before loading the exported policy.
  monkeypatch.setattr(TensorTeacher, "__init__", lambda *a, **k: pytest.fail("Student loading must not construct a teacher."))
  student, student_metadata = load_student(artifact, num_envs=4)
  assert student_metadata["version"] == 1
  with torch.no_grad():
    runner.alg.student.reset()
    torch.testing.assert_close(student(observations()), runner.alg.student(observations()))
  assert list(tmp_path.glob("events.out.tfevents.*"))
  teacher_manifest = runner.alg.teacher.metadata()
  changed_manifest = deepcopy(teacher_manifest)
  changed_manifest["checkpoints"]["quad"]["sha256"] = "different"
  monkeypatch.setattr(runner.alg.teacher, "metadata", lambda: changed_manifest)
  with pytest.raises(ValueError, match="teacher"):
    runner.load(str(checkpoint))
  moved_manifest = deepcopy(teacher_manifest)
  moved_manifest["checkpoints"]["quad"]["path"] = "/moved/policy.pt"
  monkeypatch.setattr(runner.alg.teacher, "metadata", lambda: moved_manifest)
  # Resume the existing runner: physical state and recurrent episode state reset.
  runner.load(str(checkpoint))
  assert env.reset_count == 1
  assert env.sensor_history.eq(0).all()
  assert runner.current_learning_iteration == runner.alg.num_updates == 1
  assert runner.alg.student.get_hidden_state() is None
  runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
  assert runner.current_learning_iteration == runner.alg.num_updates == 2
  assert (tmp_path / "model_2.pt").is_file()


@pytest.mark.parametrize("version,input_dim", [(2, 48), (1, 47)])
def test_student_loader_rejects_incompatible_artifacts(tmp_path, version, input_dim):
  from src.tasks.legmanip_distil.rl import load_student, make_student

  student = make_student()
  cfg = dict(student.model_cfg, input_dim=input_dim)
  path = tmp_path / "invalid.pt"
  torch.save({
    "format": "legmanip_distil.student", "version": version, "student_cfg": cfg,
    "student_state_dict": student.state_dict(),
    "infos": {"legmanip_distil": {"version": version, "student_cfg": cfg}},
  }, path)
  with pytest.raises(ValueError):
    load_student(path)
