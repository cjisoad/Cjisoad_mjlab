"""Collector contracts without requiring a GPU or loading trained actors."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tempfile
import unittest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/collect_transition_entries.py'


def collector():
  assert SCRIPT.is_file(), 'real transition entry collector is missing'
  spec = importlib.util.spec_from_file_location('collect_transition_entries', SCRIPT)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_cli_defaults_and_rejects_empty_collection():
  module = collector()
  args = module.parse_args([])
  assert args.teacher_checkpoint.name == 'model_14999.pt'
  assert args.transition_checkpoint.name == 'model_30000.pt'
  assert args.num_envs == 256
  for option in ('--num-envs', '--batches', '--seconds', '--sample-interval'):
    with unittest.TestCase().assertRaises(SystemExit):
      module.parse_args([option, '0'])


def test_targets_cover_requested_velocity_range_and_leg_workspace():
  module = collector()
  targets = module.sample_targets(256, np.random.default_rng(7))
  assert targets.shape == (256, 6)
  lower = np.array([-.5, -.4, -.8, -.3, -.12, .28])
  upper = np.array([1., .4, .8, .3, .12, .35])
  assert np.all(targets >= lower) and np.all(targets <= upper)
  assert targets[:, 0].min() <= -.49 and targets[:, 0].max() >= .99
  assert targets[:, 1].min() <= -.39 and targets[:, 1].max() >= .39
  assert targets[:, 2].min() <= -.79 and targets[:, 2].max() >= .79


def test_collection_reserves_near_reference_low_speed_trajectories():
  targets = collector().sample_targets(256, np.random.default_rng(7))
  low_speed = np.linalg.norm(targets[:, :2], axis=1) < .2
  near_fr = np.linalg.norm(targets[:, 3:5]-[.10947, .01145], axis=1) < .06
  assert (low_speed & near_fr).sum() >= 50


def fake_world():
  quat = torch.tensor([[2**-.5, 0., 0., 2**-.5]])
  data = SimpleNamespace(root_link_pos_w=torch.tensor([[10., 20., .6]]),
    root_link_quat_w=quat, root_link_lin_vel_w=torch.tensor([[3., 4., 5.]]),
    root_link_ang_vel_w=torch.tensor([[6., 7., 8.]]),
    joint_pos=torch.zeros(1, 12), joint_vel=torch.ones(1, 12))
  # Reference base at z=.4, FR .2m in front and z=.1. Yaw alignment
  # puts FR at [10,20.2,.1], preserving reference height, not actual .6.
  motion = SimpleNamespace(motion_anchor_body_index=0,
    cfg=SimpleNamespace(body_names=('base_link', 'FR_foot')),
    motion=SimpleNamespace(body_pos_w=torch.tensor([[[0., 0., .4], [.2, 0., .1]]]),
      body_quat_w=torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]])))
  term = SimpleNamespace(foot_pos_anchor=torch.tensor([[.2, -.1, .02]]),
    zero=torch.tensor([.2, -.1, -.3]), command=torch.zeros(1, 6))
  data.body_link_pos_w=torch.tensor([[[10., 20.2, .25]]])
  robot = SimpleNamespace(data=data, find_bodies=lambda *a, **kw: ([0], ['FR_foot']))
  action = SimpleNamespace(raw_action=torch.ones(1, 12)*2,
    previous_action=torch.ones(1, 12), previous_joint_vel=torch.ones(1, 12)*9,
    target_limiter=SimpleNamespace(position=torch.ones(1, 12)*.5, velocity=torch.ones(1, 12)*.2))
  scene = type('Scene', (dict,), {})({'robot': robot})
  scene.env_origins = torch.tensor([[10., 20., 0.]])
  env = SimpleNamespace(scene=scene, command_manager=SimpleNamespace(
    get_term=lambda name: motion if name == 'motion' else term),
    action_manager=SimpleNamespace(get_term=lambda name: action))
  return env


def test_snapshot_uses_world_velocity_and_aligned_reference_first_frame():
  module = collector()
  env = fake_world()
  history = torch.stack([torch.ones(1, 12)*v for v in (2, 1, -3)], dim=1)
  result = module.capture_snapshot(env, history, torch.tensor([123]))
  np.testing.assert_allclose(result['root_state'][0, :3], [0., 0., .6])
  np.testing.assert_allclose(result['root_state'][0, 7:], [3., 4., 5., 6., 7., 8.])
  np.testing.assert_allclose(result['fr_error'], [.15], atol=1e-6)
  np.testing.assert_allclose(result['fr_offset'], [[0., 0., .32]], atol=1e-6)
  np.testing.assert_array_equal(result['manager_actions'], history.numpy())
  assert result['actual_speed'][0] == 5.
  env.action_manager.get_term('joint_pos').raw_action.fill_(999.)
  assert result['raw_action'][0, 0] == 2.  # Snapshot owns its memory.


def test_snapshot_preserves_limiter_position_and_velocity():
  env = fake_world()
  result = collector().capture_snapshot(env, torch.zeros(1, 3, 12), torch.tensor([123]))
  np.testing.assert_array_equal(result['limiter_position'], np.full((1, 12), .5, np.float32))
  np.testing.assert_array_equal(result['limiter_velocity'], np.full((1, 12), .2, np.float32))
  env.action_manager.get_term('joint_pos').target_limiter.position.zero_()
  assert result['limiter_position'][0, 0] == .5


def test_valid_entries_excludes_failures_and_nonfinite_but_keeps_moving_errors():
  module = collector()
  one = module.capture_snapshot(fake_world(), torch.zeros(1, 3, 12), torch.tensor([1]))
  data = {key: np.repeat(value, 4, axis=0) for key, value in one.items()}
  data['actual_speed'][:] = [0., 1., 2., 3.]
  data['fr_error'][:] = [.5, .7, .9, 1.]
  data['joint_vel'][2, 0] = np.nan
  data['fr_offset'][3, 2] = .1
  np.testing.assert_array_equal(module.valid_entries(data, np.array([False, True, False, False])),
    [True, False, False, False])


def test_bank_roundtrip_is_pickle_free_and_reports_missing_speed_bins():
  module = collector()
  sample = module.capture_snapshot(fake_world(), torch.zeros(1, 3, 12), torch.tensor([123]))
  output = Path(tempfile.mkdtemp()) / 'entries.npz'
  report = module.save_bank(output, [sample], {'schema_version': 1})
  with np.load(output, allow_pickle=False) as archive:
    assert set(archive.files) == set(sample) | {'metadata_json'}
    metadata = json.loads(archive['metadata_json'].item())
    assert metadata['schema_version'] == 1
    assert metadata['coverage']['states'] == 1
    assert metadata['coverage']['trajectories'] == 1
    assert metadata['coverage']['missing_speed_bins']
    assert metadata['coverage'] == report
    assert all(archive[key].dtype != object for key in archive.files)
  with unittest.TestCase().assertRaisesRegex(ValueError, 'No valid'):
    module.save_bank(output, [], {'schema_version': 1})


def test_physics_digest_is_stable_and_detects_nominal_model_changes():
  module = collector()
  names = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
    'body_pos', 'body_quat', 'jnt_range', 'jnt_axis', 'geom_type',
    'geom_size', 'geom_pos', 'geom_quat', 'geom_friction', 'dof_armature', 'dof_damping')
  model = SimpleNamespace(**{name: np.ones(2) for name in names})
  before = module.compiled_physics_sha256(model)
  assert before == module.compiled_physics_sha256(model)
  model.body_mass[0] = 3.
  assert before != module.compiled_physics_sha256(model)


class CollectorTests(unittest.TestCase):
  pass

for _name, _test in list(globals().items()):
  if _name.startswith('test_'):
    setattr(CollectorTests, _name, lambda self, fn=_test: fn())
del _name, _test

if __name__ == '__main__':
  unittest.main()
