"""Compare migrated rewards with AST-extracted Isaac Gym source on CPU.

No Isaac Gym modules are imported. Set GO2_STAND_SOURCE to the original
Go2_Handstand.py, or pass --source when running this test file directly.
The source-oracle tests skip explicitly when the original checkout is absent.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RELATIVE = Path("legged_gym/envs/Go2_Stand/Go2_Handstand/Go2_Handstand.py")
DEFAULT_SOURCE = Path("/home/eden/project/My_unitree_go2_gym") / SOURCE_RELATIVE
ACTIVE_NAMES = (
  "action_rate", "alive", "ang_vel_xy", "ang_xz", "base_height", "collision",
  "contact", "default_hip_pos", "default_pos", "default_pos_reward", "dof_acc",
  "dof_pos_limits", "feet_clearance", "feet_height_symmetry",
  "handstand_feet_height_exp", "handstand_feet_on_air", "handstand_orientation",
  "lin_vel_z", "orientation_symmetry", "symmetric_joints", "torques",
  "tracking_ang_vel", "tracking_lin_vel",
)
JOINT_NAMES = tuple(
  f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")
)


def load_file(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def compile_source_oracle(source_path):
  source = source_path.read_text()
  tree = ast.parse(source, filename=str(source_path))
  robot = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Go2_Handstand_Robot")
  helper_names = {"_get_phase", "_get_gait_phase", "_prepare_reward_function", "compute_reward"}
  methods = [
    node for node in robot.body if isinstance(node, ast.FunctionDef)
    and (node.name.startswith("_reward_") or node.name in helper_names)
  ]
  oracle_class = ast.ClassDef(name="SourceRewardOracle", bases=[], keywords=[], body=methods, decorator_list=[])
  module = ast.fix_missing_locations(ast.Module(body=[oracle_class], type_ignores=[]))
  namespace = {"torch": torch}
  exec(compile(module, str(source_path), "exec"), namespace)

  cfg_path = source_path.with_name("Go2_Handstand_Config.py")
  cfg_tree = ast.parse(cfg_path.read_text(), filename=str(cfg_path))
  cfg_class = next(node for node in cfg_tree.body if isinstance(node, ast.ClassDef) and node.name == "Go2_Handstand_Cfg_Yu")
  rewards = next(node for node in cfg_class.body if isinstance(node, ast.ClassDef) and node.name == "rewards")
  exec(compile(ast.Module(body=[rewards], type_ignores=[]), str(cfg_path), "exec"), namespace)
  init_state = next(node for node in cfg_class.body if isinstance(node, ast.ClassDef) and node.name == "init_state")
  desired_node = next(
    node for node in init_state.body if isinstance(node, ast.Assign)
    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "descire_joint_angles"
  )
  desired = ast.literal_eval(desired_node.value)
  return namespace["SourceRewardOracle"], namespace["rewards"], desired, hashlib.sha256(source.encode()).hexdigest()


def fixture(oracle_class, reward_cfg, desired_angles, gate=0.9):
  count = 6
  generator = torch.Generator(device="cpu").manual_seed(617)

  def random(*shape):
    return torch.randn(*shape, generator=generator)

  source = oracle_class()
  source.num_envs = count
  source.num_dof = 12
  source.device = "cpu"
  source.dt = 0.02
  source.cfg = SimpleNamespace(rewards=reward_cfg)
  source.base_lin_vel = random(count, 3) * 0.3
  source.base_ang_vel = random(count, 3) * 0.5
  source.commands = random(count, 4) * 0.25
  source.projected_gravity = random(count, 3)
  source.projected_gravity /= source.projected_gravity.norm(dim=-1, keepdim=True)
  source.target_gravity = torch.tensor([-1.0, 0.0, 0.0])
  source.dof_pos = random(count, 12) * 1.1
  source.dof_vel = random(count, 12) * 0.8
  source.last_dof_vel = random(count, 12) * 0.7
  source.actions = random(count, 12)
  source.last_actions = random(count, 12)
  source.torques = random(count, 12) * 4
  source.descire_joint_pos = torch.tensor([desired_angles[name] for name in JOINT_NAMES])
  source.soft_dof_pos_limits = torch.stack((-torch.linspace(0.1, 1.0, 12), torch.linspace(0.2, 1.3, 12)), -1)
  source.root_states = torch.zeros(count, 13)
  source.root_states[:, 2] = torch.tensor([0.52, 0.20, 0.42, 0.60, 0.80, 0.50])
  source.measured_heights = torch.zeros(count, 1)
  source.feet_indices = torch.arange(4)
  source.feet_name_reward_indices = torch.tensor([0, 1])
  source.contact_foot_indices = torch.tensor([2, 3])
  source.penalised_contact_indices = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])
  source.contact_forces = random(count, 12, 3) * 0.12
  source.contact_forces[:, :4] = 0.0
  source.contact_forces[:, :4, 2] = torch.tensor([
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0],
    [1.001, 0.0, 1.001, 0.0],
    [0.0, 1.001, 0.0, 1.001],
    [0.0, 0.0, 2.0, 2.0],
    [0.0, 0.0, -2.0, 0.0],
  ])
  source.contact_forces[5, 0, 0] = 1.001
  source.contact_forces[:, 4] = torch.tensor([0.1, 0.0, 0.0])
  source.rigid_state = torch.zeros(count, 12, 13)
  source.rigid_state[:, :4, :3] = random(count, 4, 3) * 0.1
  source.rigid_state[:, :4, 2] += torch.tensor([0.67, 0.67, 0.02, 0.02])
  source.feet_pos = source.rigid_state[:, :2, :3]
  source.episode_length_buf = torch.tensor([0, 20, 40, 60, 80, 120])
  source.rew_hanstand = torch.tensor(gate)
  source.base_euler_xyz = torch.tensor([
    [-3.0, 0.4, 0.2], [-0.7, -0.3, 1.0], [0.0, 0.0, 0.0],
    [0.7, 0.5, -0.2], [2.3, -0.8, 0.3], [3.0, 0.9, -1.0],
  ])
  roll, pitch, yaw = source.base_euler_xyz.unbind(-1)
  cr, cp, cy = (angle.div(2).cos() for angle in (roll, pitch, yaw))
  sr, sp, sy = (angle.div(2).sin() for angle in (roll, pitch, yaw))
  quat_wxyz = torch.stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                          cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), -1)
  joint_ids = torch.tensor([4, 1, 10, 7, 2, 11, 6, 0, 9, 3, 8, 5])
  joint_limits = torch.empty(count, 12, 2)
  joint_limits[:, joint_ids] = source.soft_dof_pos_limits
  state = SimpleNamespace(
    height_score=source.rew_hanstand.clone(), joint_pos=source.dof_pos,
    joint_vel=source.dof_vel, previous_joint_vel=source.last_dof_vel,
    raw_action=source.actions, previous_action=source.last_actions,
    desired_angles=source.descire_joint_pos, joint_ids=joint_ids,
  )
  data = SimpleNamespace(
    root_link_pos_w=source.root_states[:, :3], root_link_lin_vel_b=source.base_lin_vel,
    root_link_ang_vel_b=source.base_ang_vel, projected_gravity_b=source.projected_gravity,
    root_link_quat_w=quat_wxyz, site_pos_w=source.rigid_state[:, :4, :3],
    soft_joint_pos_limits=joint_limits, actuator_force=source.torques,
  )
  env = SimpleNamespace(
    num_envs=count, device="cpu", step_dt=source.dt, episode_length_buf=source.episode_length_buf,
    action_manager=SimpleNamespace(get_term=lambda name: state),
    command_manager=SimpleNamespace(get_command=lambda name: source.commands[:, :3]),
    scene={
      "robot": SimpleNamespace(data=data),
      "front_contact": SimpleNamespace(data=SimpleNamespace(force=source.contact_forces[:, :2])),
      "rear_contact": SimpleNamespace(data=SimpleNamespace(force=source.contact_forces[:, 2:4])),
      "nonfoot_contact": SimpleNamespace(data=SimpleNamespace(force=source.contact_forces[:, 4:])),
    },
  )
  return source, env, state


class StandRewardParityTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    source_path = Path(os.environ.get("GO2_STAND_SOURCE", str(DEFAULT_SOURCE))).expanduser()
    if source_path.is_dir():
      source_path /= SOURCE_RELATIVE
    if not source_path.is_file():
      raise unittest.SkipTest(
        f"Original Go2 source unavailable at {source_path}; set GO2_STAND_SOURCE or use --source"
      )
    cls.oracle_class, cls.reward_cfg, cls.desired_angles, cls.source_sha256 = compile_source_oracle(source_path)
    cls.rewards = load_file("stand_reward_parity_terms", ROOT / "src/tasks/stand/mdp/rewards.py")
    cls.constants = load_file("stand_reward_parity_constants", ROOT / "src/tasks/stand/constants.py")
    cls.scales = {
      name: getattr(cls.reward_cfg.scales, name)
      for name in dir(cls.reward_cfg.scales)
      if not name.startswith("_") and getattr(cls.reward_cfg.scales, name) != 0
    }

  def make_fixture(self, gate=0.9):
    return fixture(self.oracle_class, self.reward_cfg, self.desired_angles, gate)

  def migrated(self, name, env):
    kwargs = {}
    if name in ("feet_height_symmetry", "handstand_feet_height_exp"):
      kwargs["asset_cfg"] = SimpleNamespace(name="robot", site_ids=[0, 1])
    elif name == "feet_clearance":
      kwargs["asset_cfg"] = SimpleNamespace(name="robot", site_ids=[2, 3])
    return getattr(self.rewards, name)(env, **kwargs)

  def assert_reward_equal(self, actual, expected):
    torch.testing.assert_close(
      actual, torch.as_tensor(expected, dtype=actual.dtype).expand_as(actual), atol=1e-5, rtol=1e-6,
      msg=lambda message: f"{message}\nSource SHA256: {self.source_sha256}",
    )

  def test_active_scales_match_source(self):
    self.assertEqual(tuple(self.scales), ACTIVE_NAMES)
    self.assertEqual(self.constants.REWARD_WEIGHTS, self.scales)
    self.assertEqual(len(self.scales), 23)

  def test_alphabetical_order_preserves_previous_gate_for_ang_xz(self):
    for old_gate, current_height in ((0.0, 0.52), (0.9, 0.20)):
      with self.subTest(old_gate=old_gate, current_height=current_height):
        source, env, state = self.make_fixture(old_gate)
        source.root_states[:, 2] = current_height
        source.reward_scales = dict(self.scales)
        source._prepare_reward_function()
        self.assertEqual(tuple(source.reward_names), ACTIVE_NAMES)
        source.rew_buf = torch.zeros(source.num_envs)
        source.compute_reward()
        total = torch.zeros(env.num_envs)
        terms = {}
        for name, weight in self.scales.items():
          terms[name] = self.migrated(name, env)
          total += terms[name] * weight * env.step_dt
          self.assert_reward_equal(terms[name] * weight * env.step_dt, source.episode_sums[name])
        self.assert_reward_equal(total, source.rew_buf)
        self.assert_reward_equal(state.height_score, source.rew_hanstand)
        self.assertEqual(torch.count_nonzero(terms["ang_xz"]).item(), 0 if old_gate == 0 else 5)
        self.assertEqual(torch.count_nonzero(terms["tracking_lin_vel"]).item(), 6 if current_height == 0.52 else 0)

  def test_height_gate_is_batch_global_and_strict(self):
    source, env, state = self.make_fixture()
    for heights, expected_on in (
      ([0.52, 0.20, 0.20, 0.20, 0.20, 0.20], False),
      ([0.52, 0.52, 0.52, 0.52, 0.52, 0.20], True),
    ):
      source.root_states[:, 2] = torch.tensor(heights)
      self.assert_reward_equal(self.rewards.base_height(env), source._reward_base_height())
      self.assert_reward_equal(self.rewards.tracking_lin_vel(env), source._reward_tracking_lin_vel())
      self.assertEqual(bool(torch.all(self.rewards.tracking_lin_vel(env) > 0)), expected_on)
      self.assertEqual(state.height_score.ndim, 0)
    for gate, expected_on in ((0.70, False), (0.70001, True)):
      source.rew_hanstand = torch.tensor(gate)
      state.height_score = torch.tensor(gate)
      self.assert_reward_equal(self.rewards.tracking_ang_vel(env), source._reward_tracking_ang_vel())
      self.assertEqual(bool(torch.all(self.rewards.tracking_ang_vel(env) > 0)), expected_on)

  def test_gait_half_cycle_boundary_rewards_both_rear_feet(self):
    source, env, state = self.make_fixture()
    source.rigid_state[:, 2:4, 2] = 0.02
    # At the real control dt, float32 rounding puts step 40 just below phase .5.
    for dt, step, expected in ((0.02, 40, 1.0), (0.1, 8, 2.0)):
      with self.subTest(dt=dt, step=step):
        source.dt = env.step_dt = dt
        source.episode_length_buf[:] = step
        actual = self.migrated("feet_clearance", env)
        self.assert_reward_equal(actual, source._reward_feet_clearance())
        torch.testing.assert_close(actual, torch.full((6,), expected), atol=1e-6, rtol=1e-6)


def reward_test(name):
  def test(self):
    for gate in (0.0, 0.70, 0.9):
      with self.subTest(gate=gate):
        source, env, state = self.make_fixture(gate)
        actual = self.migrated(name, env)
        expected = getattr(source, f"_reward_{name}")()
        self.assert_reward_equal(actual, expected)
        self.assertEqual(actual.shape, (source.num_envs,))
  return test


for _name in ACTIVE_NAMES:
  setattr(StandRewardParityTests, f"test_source_{_name}", reward_test(_name))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--source", type=Path)
  args, remaining = parser.parse_known_args()
  if args.source is not None:
    os.environ["GO2_STAND_SOURCE"] = str(args.source)
  unittest.main(argv=[str(Path(__file__)), *remaining])
